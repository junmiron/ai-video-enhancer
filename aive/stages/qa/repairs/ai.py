"""AI restore models for the artifact-QA repairs (M4.x): SCUNet + deepDeband-f.

Two learned models run behind the same targeted-repair interface as the CV
fixes:

* ``scunet`` — Swin-Conv-UNet blind denoiser (cszn/SCUNet, Apache-2.0 arch;
  color-real-psnr checkpoint). Used by the flicker repair as a per-frame
  stabilizing pass on flagged static regions: in a static region every frame's
  input differs only by SR hallucination noise, so an independent denoiser maps
  them all to ~the same clean estimate and the temporal variance collapses.
* ``deepdeband_f`` — pix2pix unet_256 debander (BSD-3 + MIT arch; TAS mirror).
  Used by the banding repair on flagged plateau regions.

Backend chain mirrors :class:`~aive.stages.upscale.stage.SRRunner`: TensorRT
fp16 engine per exact padded size (auto-fallback to default precision inside
``build_engine``, then to PyTorch-CUDA fp16 when no engine can be built). The
installed onnxruntime has no CUDA EP, so the PyTorch fallback is required and
the arch code is vendored in this package.

Full-frame inference is the default; when the estimated activation memory for
one frame exceeds 75% of free VRAM (nvidia-smi-aware), the runner switches to
tiled inference reusing :mod:`aive.stages.upscale.tiling` at scale=1 — with an
RTX 5090 shared by an LLM server this path is expected to engage often, so it
is a first-class mode, not an emergency valve.

Runners are cached per (model, width, height) for the process lifetime;
weights auto-download from the manifest (``task: restore``) on first use with
clear logging. Any failure degrades to ``None`` (cached) so repairs fall back
to their CV paths without retrying per frame.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RestoreModelSpec:
    """Static properties of a restore model family (scale is always 1)."""

    name: str
    pad_mod: int  # input dims must be multiples of this (model-internal padding unit)
    min_dim: int  # minimum engine dim in px
    bytes_per_pixel: float  # conservative fp16 peak-activation estimate, per INPUT pixel


RESTORE_SPECS: dict[str, RestoreModelSpec] = {
    # SCUNet: internal ReplicationPad to a multiple of 64; window attention needs
    # H/8 >= 8 after three downsamples -> min dim 64. Heavier per pixel than the
    # U-Net (attention score tensors scale with heads x p^2).
    "scunet": RestoreModelSpec(name="scunet", pad_mod=64, min_dim=64, bytes_per_pixel=3000.0),
    # deepDeband-f: unet_256 skip connections require even sizes at every level
    # -> dims must be multiples of 256 (the wrapper reflect-pads internally).
    "deepdeband_f": RestoreModelSpec(name="deepdeband_f", pad_mod=256, min_dim=256, bytes_per_pixel=1500.0),
}


def spec_for_restore(name: str) -> RestoreModelSpec:
    if name not in RESTORE_SPECS:
        raise ValueError(f"Unknown restore model '{name}'. Known: {list(RESTORE_SPECS)}")
    return RESTORE_SPECS[name]


# -- arch construction / weight loading ----------------------------------------


def build_restore(name: str, device: str = "cuda", dtype=None):
    """Construct a fresh restore model for `name` (weights NOT loaded)."""
    if name == "scunet":
        from ._scunet_arch import SCUNet

        # Pinned to the scunet_color_real_psnr checkpoint: config=[4]*7 verified
        # with a strict state-dict load (540 keys, 17.95M params). The class
        # default [2]*7 does NOT match this file.
        model = SCUNet(in_nc=3, config=[4] * 7, dim=64)
    elif name == "deepdeband_f":
        from ._deepdeband_arch import DeepDebandF

        model = DeepDebandF()
    else:
        raise ValueError(f"Unknown restore model '{name}'. Known: {list(RESTORE_SPECS)}")
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def load_restore_state_dict(weights_path: str | Path) -> dict:
    """Load a restore checkpoint from disk (plain state dicts for both models)."""
    import torch

    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)!r}")
    return state


def apply_restore_state(model, name: str, state: dict) -> None:
    """Load restore weights into the arch; fatal on ANY mismatch.

    Unlike SR (which tolerates training-only extras), both restore checkpoints
    were verified as exact matches — running with random or partially-loaded
    weights would silently corrupt frames, so missing AND unexpected keys are
    errors here.
    """
    if name == "deepdeband_f":
        # The checkpoint is a plain UnetGenerator256 state dict (keys prefixed
        # 'model.' — the generator's own attribute); the wrapper holds it at .net.
        state = {"net." + k: v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        sample = ", ".join((result.missing_keys + result.unexpected_keys)[:6])
        raise RuntimeError(
            f"Restore weights do not match the arch ({name}): "
            f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected; e.g. {sample}"
        )


def load_restore(name: str, weights_path: str | Path, device: str = "cuda", dtype=None):
    """Load restore model `name` from a checkpoint file (eval mode)."""
    model = build_restore(name, device=device, dtype=dtype)
    apply_restore_state(model, name, load_restore_state_dict(weights_path))
    log.info("Loaded %s weights from %s", name, weights_path)
    return model.eval()


# -- runner ---------------------------------------------------------------------


class RestoreRunner:
    """Runs one restore model at a fixed (padded) resolution on TRT or CUDA.

    Either full-frame (default when it fits in VRAM) or tiled with feathered
    blending — decided once at construction from current free VRAM. Output is
    the same size as input (scale 1), HWC float32 in [0,1].
    """

    def __init__(
        self,
        weights_path: str | Path,
        width: int,
        height: int,
        model_name: str,
        use_trt: bool = True,
    ):
        import torch

        from ...upscale.tiling import OVERLAP, choose_tile_size

        spec = spec_for_restore(model_name)
        self.spec = spec
        self.torch = torch
        self.width, self.height = width, height

        # Padded full-frame dims (model size constraints).
        self.pw = max(math.ceil(width / spec.pad_mod) * spec.pad_mod, spec.min_dim)
        self.ph = max(math.ceil(height / spec.pad_mod) * spec.pad_mod, spec.min_dim)

        # Full frame vs tiling: estimate activation VRAM for one full frame.
        from ....inference.trt import free_vram_bytes

        tile_cover = choose_tile_size(
            height, width, free_vram_bytes(), spec.bytes_per_pixel, pad_mod=spec.pad_mod
        )
        self.tile_cover = tile_cover  # None -> full-frame mode

        if tile_cover is None:
            self.engine_h, self.engine_w = self.ph, self.pw
        else:
            side = tile_cover + 2 * OVERLAP
            self.engine_h = max(math.ceil(side / spec.pad_mod) * spec.pad_mod, spec.min_dim)
            self.engine_w = max(math.ceil(side / spec.pad_mod) * spec.pad_mod, spec.min_dim)

        log.info(
            "Restore %s @ %dx%d: %s (engine %dx%d)",
            model_name, width, height,
            f"tiled C={tile_cover} ov={OVERLAP}" if tile_cover else "full-frame",
            self.engine_w, self.engine_h,
        )

        self._trt_runner = None
        self._torch_model = None
        if use_trt:
            try:
                self._build_trt(weights_path)
            except Exception as exc:  # pragma: no cover - TRT build edge cases
                log.warning(
                    "TRT engine unavailable for %s (%s); falling back to PyTorch-CUDA.",
                    model_name, str(exc)[:300],
                )
        if self._trt_runner is None:
            self._torch_model = load_restore(model_name, weights_path, device="cuda", dtype=torch.float16).eval()

    # -- backend setup -------------------------------------------------------

    def _build_trt(self, weights_path: str | Path) -> None:
        # NOTE: torch is imported HERE (not at module level — this package keeps
        # heavy imports lazy). A missing local import raised NameError on every
        # build attempt and silently disabled TRT for both restore models.
        import torch

        from ....inference.onnx_export import _ensure_self_contained
        from ....inference.trt import TrtRunner, build_engine, engine_cache_dir, free_vram_bytes

        weights_path = Path(weights_path)
        sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()[:16]
        profile = f"{self.spec.name}|{self.width}x{self.height}|eng{self.engine_w}x{self.engine_h}"
        key = hashlib.sha256(f"{sha}|fp16|{profile}".encode()).hexdigest()[:32]

        cache_dir = engine_cache_dir() / "restore"
        onnx_path = cache_dir / f"{self.spec.name}_{key}.onnx"
        engine_path = cache_dir / f"{self.spec.name}_{key}_fp16.engine"

        if not engine_path.exists():
            # Engine builds need VRAM headroom beyond inference (an fp32 trace
            # forward + the largest intermediate tensor + workspace pool). When
            # another process holds most of the GPU, skip the doomed build and
            # let the PyTorch-CUDA fallback cover it.
            min_free = (6 if self.tile_cover is None else 2) * 2**30
            free = free_vram_bytes()
            if free < min_free:
                log.info(
                    "Skipping TRT build for %s: %.1f GiB VRAM free (< %.0f GiB headroom); using PyTorch-CUDA.",
                    self.spec.name, free / 2**30, min_free / 2**30,
                )
                return
            model = load_restore(self.spec.name, weights_path, device="cuda", dtype=None)  # fp32 export; TRT picks layer precision
            x = torch.zeros(1, 3, self.engine_h, self.engine_w, device="cuda")
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                torch.onnx.export(
                    model,
                    (x,),
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=18,
                    dynamo=False,  # legacy TorchScript exporter: TAS-proven for RIFE/SR
                )
            _ensure_self_contained(onnx_path)
            del model
            torch.cuda.empty_cache()
            build_engine(onnx_path, engine_path, precision="fp16")
        self._trt_runner = TrtRunner(engine_path)
        log.info("Restore TRT runner ready (%s fp16, engine %dx%d)", self.spec.name, self.engine_w, self.engine_h)

    # -- inference -----------------------------------------------------------

    def __call__(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        """Run the model on one HWC uint8 frame; returns HWC float32 in [0,1] at input size."""
        if self.tile_cover is None:
            out = self._run_full_frame(frame_hwc_u8)
        else:
            out = self._run_tiled(frame_hwc_u8)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _to_nchw(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        f = frame_hwc_u8.astype(np.float32) / 255.0
        return np.ascontiguousarray(f.transpose(2, 0, 1)[None])

    def _pad_to_engine(self, region_nchw: np.ndarray) -> np.ndarray:
        """Edge-pad (or pass through) a NCHW region to the fixed engine size.

        Content is kept at [0,0]; the deficit is padded on bottom/right only
        with edge replication. Both consumers crop from the top-left corner
        (full-frame ``[:h, :w]`` and tile cover offsets), so centered padding
        would silently shift every padded region — see test_sr_pad.py.
        The padded area is never used in the output; replicated edges just keep
        it plausible context for models with global ops. (Same helper semantics
        as SRRunner._pad_to_engine.)
        """
        _, _, rh, rw = region_nchw.shape
        if rh == self.engine_h and rw == self.engine_w:
            return region_nchw
        take_h = min(rh, self.engine_h)
        take_w = min(rw, self.engine_w)
        core = region_nchw[:, :, :take_h, :take_w]
        dh = self.engine_h - take_h
        if dh > 0:
            core = np.concatenate(
                [core, np.repeat(core[:, :, -1:], dh, axis=2)],
                axis=2,
            )
        dw = self.engine_w - take_w
        if dw > 0:
            core = np.concatenate(
                [core, np.repeat(core[:, :, :, -1:], dw, axis=3)],
                axis=3,
            )
        return core

    def _infer(self, nchw_f32: np.ndarray) -> np.ndarray:
        """Run the model on a NCHW float32 [0,1] array at engine size; returns NCHW float32."""
        if self._trt_runner is not None:
            out = self._trt_runner.infer_multi({"input": nchw_f32})["output"]
            return np.ascontiguousarray(out)
        model = self._torch_model
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        with self.torch.no_grad():
            x = self.torch.from_numpy(nchw_f32).to(dev, non_blocking=True).to(dtype)
            out = model(x)
        return out.float().cpu().numpy()

    def _run_full_frame(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        nchw = self._pad_to_engine(self._to_nchw(frame_hwc_u8))
        out = self._infer(nchw)
        return out[0, :, : self.height, : self.width].transpose(1, 2, 0)

    def _run_tiled(self, frame_hwc_u8: np.ndarray) -> np.ndarray:
        from ...upscale.tiling import (
            OVERLAP,
            apply_edge_floor,
            blend_into,
            feather_mask,
            plan_tiles,
        )

        h, w = self.height, self.width
        ov = OVERLAP
        acc = np.zeros((1, 3, h, w), dtype=np.float32)  # NCHW accumulator (scale=1)
        wsum = np.zeros((h, w), dtype=np.float32)

        for tile in plan_tiles(h, w, self.tile_cover, ov):
            # Model input region: cover expanded by the overlap margin.
            in_t = max(0, tile.top - ov)
            in_b = min(h, tile.top + tile.cover_h + ov)
            in_l = max(0, tile.left - ov)
            in_r = min(w, tile.left + tile.cover_w + ov)
            region = self._to_nchw(frame_hwc_u8[in_t:in_b, in_l:in_r])
            nchw = self._pad_to_engine(region)
            out = self._infer(nchw)

            # Crop the cover part of the model output (offsets account for clamping).
            ct = tile.top - in_t
            cb = ct + tile.cover_h
            cl = tile.left - in_l
            cr = cl + tile.cover_w
            out_crop = np.ascontiguousarray(out[0, :, ct:cb, cl:cr][None])

            mask = feather_mask(
                tile.cover_h, tile.cover_w,
                nominal_h=self.tile_cover, nominal_w=self.tile_cover, overlap=ov
            )
            apply_edge_floor(mask, tile.top, tile.left, tile.cover_h, tile.cover_w, h, w)
            blend_into(acc, wsum, tile.top, tile.left, out_crop, mask)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = acc[0] / wsum[None]
        return np.nan_to_num(out).transpose(1, 2, 0)


# -- process-level runner cache --------------------------------------------------

_runner_cache: dict[tuple[str, int, int], RestoreRunner | None] = {}


def get_restore_runner(model_name: str, width: int, height: int) -> RestoreRunner | None:
    """Get (building on first use) the restore runner for one frame size.

    Returns ``None`` when AI repair is unavailable — no CUDA device, download
    failure, or build/load error. Failures are cached so repairs fall back to
    their CV paths without retrying per frame. Weights auto-download from the
    manifest (``task: restore``) on first use; see module docstring.
    """
    key = (model_name, width, height)
    if key in _runner_cache:
        return _runner_cache[key]

    import torch

    if not torch.cuda.is_available():
        log.warning("AI repair '%s' unavailable: no CUDA device; using CV fallback.", model_name)
        _runner_cache[key] = None
        return None

    try:
        from ....models.downloader import download_model
        from ....models.registry import get_model

        info = get_model(model_name, task="restore")
        weights = download_model(info)
        runner = RestoreRunner(weights, width, height, model_name=model_name)
    except Exception as exc:
        log.warning("AI repair '%s' unavailable (%s); using CV fallback.", model_name, str(exc)[:300])
        _runner_cache[key] = None
        return None

    _runner_cache[key] = runner
    return runner


def clear_restore_runner_cache() -> None:
    """Drop cached runners (tests / VRAM reclamation)."""
    import torch

    _runner_cache.clear()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - non-CUDA environments
        pass


__all__ = [
    "RESTORE_SPECS",
    "RestoreModelSpec",
    "RestoreRunner",
    "apply_restore_state",
    "build_restore",
    "clear_restore_runner_cache",
    "get_restore_runner",
    "load_restore",
    "load_restore_state_dict",
    "spec_for_restore",
]
