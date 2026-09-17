"""Super-resolution model registry, loading and ONNX export (M3).

Two model families ship in M3:

* ``shufflecugan_x2`` — ShuffleCUGAN x2 (UpCunet2x_fast), the anime default.
  Tiny (1.4M params); input dims must be multiples of 4 and >= 40.
* ``realesrgan_x4plus`` — Real-ESRGAN x4plus (RRDBNet, 16.7M params), the
  live-action / general default; no size constraints.

Both are loaded through a non-strict state-dict load that is *fatal on missing
inference keys* (same pattern as :func:`aive.stages.interpolate.rife_model.apply_rife_state`):
training-only extras may be ignored, but the model must never run with random
weights. Checkpoint wrappers are unwrapped here: Real-ESRGAN training releases
ship ``{'params_ema': ...}``, some community checkpoints ship plain state dicts.

ONNX export uses the legacy TorchScript exporter (``dynamo=False``) — the same
path proven for RIFE in M2 on this machine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# Registry names for SR models (manifest `task: sr`).
SR_MODELS = ("shufflecugan_x2", "realesrgan_x4plus")


@dataclass(frozen=True)
class SRModelSpec:
    """Static properties of an SR model family."""

    name: str
    scale: int  # native output scale factor (x2 / x4)
    pad_mod: int  # input dims must be multiples of this (1 = no constraint)
    min_dim: int  # minimum input dim in px (0 = none); reflect-pad limit for cugan
    bytes_per_pixel: float  # conservative fp16 peak-activation estimate, per INPUT pixel


SPECS: dict[str, SRModelSpec] = {
    "shufflecugan_x2": SRModelSpec(
        name="shufflecugan_x2", scale=2, pad_mod=4, min_dim=40, bytes_per_pixel=600.0
    ),
    "realesrgan_x4plus": SRModelSpec(
        name="realesrgan_x4plus", scale=4, pad_mod=1, min_dim=0, bytes_per_pixel=2500.0
    ),
}


def spec_for(name: str) -> SRModelSpec:
    if name not in SPECS:
        raise ValueError(f"Unknown SR model '{name}'. Known: {list(SPECS)}")
    return SPECS[name]


def build_sr(name: str, device: str = "cuda", dtype=None):
    """Construct a fresh SR model for `name` (weights NOT loaded)."""
    if name == "shufflecugan_x2":
        from ._cugan_arch import UpCunet2x_fast

        model = UpCunet2x_fast(in_channels=3, out_channels=3)
    elif name == "realesrgan_x4plus":
        from ._rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)
    else:
        raise ValueError(f"Unknown SR model '{name}'. Known: {list(SPECS)}")
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def unwrap_sr_state(state: dict) -> dict:
    """Unwrap training-checkpoint containers to a plain state dict.

    Real-ESRGAN releases ship ``{'params_ema': OrderedDict}``; some community
    checkpoints wrap in ``{'params': ...}`` or are already plain dicts.
    """
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint container type: {type(state)!r}")
    for key in ("params_ema", "params"):
        inner = state.get(key)
        if isinstance(inner, dict) and any(
            hasattr(v, "shape") for v in list(inner.values())[:1]
        ):
            return dict(inner)
    # Plain state dict (or unrecognized wrapper — let the loader report it).
    return state


def load_sr_state_dict(weights_path: str | Path) -> dict:
    """Load + unwrap SR weights from disk."""
    import torch

    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        # Some checkpoints store tensors under a top-level key only.
        return unwrap_sr_state(state)
    raise TypeError(f"Unexpected checkpoint type: {type(state)!r}")


def apply_sr_state(model, state: dict) -> None:
    """Load an unwrapped state dict into the arch.

    Non-strict on purpose (training-only extras are ignored), but any key the
    inference arch needs must be present and shape-compatible — otherwise the
    model would run with random weights.
    """
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        sample = ", ".join(result.missing_keys[:6])
        raise RuntimeError(
            f"SR weights do not match the arch: {len(result.missing_keys)} "
            f"missing key(s), e.g. {sample}"
        )


def load_sr(name: str, weights_path: str | Path, device: str = "cuda", dtype=None):
    """Load SR model `name` from a checkpoint file."""
    model = build_sr(name, device=device, dtype=dtype)
    state = load_sr_state_dict(weights_path)
    apply_sr_state(model, state)
    del state
    log.info("Loaded %s weights from %s", name, weights_path)
    return model


class SRExportWrapper:
    """Exports an SR arch to ONNX with named inputs/outputs.

    Input: ``input`` [1,3,H,W] (padded dims). Output: ``output``
    [1,3,H*scale,W*scale]. Single-tensor I/O keeps the TRT runner simple.
    """

    @staticmethod
    def export(model, spec: SRModelSpec, onnx_path: str | Path, height: int, width: int, opset: int = 18) -> Path:
        import torch

        from ...inference.onnx_export import _ensure_self_contained

        model.eval()
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        x = torch.zeros(1, 3, height, width, device=dev, dtype=dtype)
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                torch.onnx.export(
                    model,
                    (x,),
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=opset,
                    dynamo=False,  # legacy TorchScript exporter: TAS-proven for RIFE
                )
        _ensure_self_contained(onnx_path)
        log.info("Exported %s ONNX -> %s (%dx%d)", spec.name, onnx_path, width, height)
        return onnx_path
