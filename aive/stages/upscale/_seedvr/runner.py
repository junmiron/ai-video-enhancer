"""SeedVR2-3B clip-based video super-resolution runner (experimental).

Wraps the vendored official :class:`projects.video_diffusion_sr.infer.VideoDiffusionInfer`
(Apache-2.0) for single-GPU, one-step SR inference over temporal clips:

    input frames ─► bicubic resize to target res + normalize [-1,1]
                 ─► (windowed into clips of <= max_clip_frames, overlap-blended)
                 ─► Video-VAE encode ─► NaDiT one-step denoise (bf16 autocast)
                 ─► VAE decode ─► wavelet color-fix vs the resized source
                 ─► uint8 frames at target resolution

Unlike the per-frame CNN SR models in :mod:`aive.stages.upscale.stage`, SeedVR2
is a *temporal* diffusion model: it sees whole clips, so this runner buffers a
segment's frames and processes them in windows with linear crossfade blending at
the overlaps (same pattern as the depth streaming models). Windows are padded to
``T ≡ 1 (mod 4)`` by repeating the last frame (VAE temporal factor is 4), then
trimmed back.

One-step inference follows the official ``inference_seedvr2_3b.py`` defaults:
CFG scale 1.0, sampling steps 1, cond noise scale 0.0 (the condition latent
passes through unchanged — the lerp schedule's forward at t=0 is the identity).

The official repo's ``color_fix.py`` (wavelet reconstruction) is not present in
the upstream checkout, so :func:`wavelet_color_fix` re-implements it locally: a
single-level Haar split that keeps the *source* low frequencies (true colors /
exposure) and takes the high-frequency detail from the restored frames.

VRAM: DiT (~3 B params, bf16 ≈ 6 GiB) + Video-VAE-v3 + activations. The runner
follows the official memory dance (DiT off-GPU while the VAE encodes/decodes).
If a window still OOMs, the whole segment is retried with half the window size
(down to ``min_window_frames``) after clearing the CUDA cache.

License: Apache-2.0 (c) 2025 Bytedance Ltd. — see ``vendor/NOTICE.md``.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


class SeedVrError(RuntimeError):
    """Raised when a clip cannot be processed (size guard / unrecoverable OOM)."""


def _floor_to(n: int, mod: int) -> int:
    return max(mod, (int(n) // mod) * mod)


def _pad_t(t: int) -> int:
    """Smallest frame count >= t with T ≡ 1 (mod 4) (VAE temporal factor)."""
    r = (t - 1) % 4
    return t if r == 0 else t + (4 - r)


def wavelet_color_fix(gen_tchw: "object", ref_tchw: "object") -> "object":
    """Single-level Haar color fix: source low-freq + restored high-freq.

    ``gen_tchw`` / ``ref_tchw`` are CUDA/CPU float tensors of shape (B, C, H, W)
    in [-1, 1] with even spatial dims (ours are multiples of 16). The output's
    low-frequency band (and therefore overall color/exposure) comes from the
    reference — the exact condition input the model saw — while high-frequency
    detail comes from the generated frames. This is a local re-implementation of
    SeedVR's ``wavelet_reconstruction`` (absent from the upstream checkout); it
    uses an *exact* single-level Haar split/merge so constants reconstruct
    perfectly and no band energy leaks into the DC term.
    """
    import torch

    def _split(img):
        a = img[:, :, 0::2, 0::2]
        b = img[:, :, 0::2, 1::2]
        c = img[:, :, 1::2, 0::2]
        d = img[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 4
        hl = (a - b + c - d) / 4
        lh = (a + b - c - d) / 4
        hh = (a - b - c + d) / 4
        return ll, hl, lh, hh

    def _merge(ll, hl, lh, hh):
        out = torch.empty(
            (ll.shape[0], ll.shape[1], 2 * ll.shape[-2], 2 * ll.shape[-1]),
            dtype=ll.dtype, device=ll.device,
        )
        out[:, :, 0::2, 0::2] = ll + hl + lh + hh
        out[:, :, 0::2, 1::2] = ll - hl + lh - hh
        out[:, :, 1::2, 0::2] = ll + hl - lh - hh
        out[:, :, 1::2, 1::2] = ll - hl - lh + hh
        return out

    b, c_, h, w = gen_tchw.shape
    ph, pw = h % 2, w % 2
    g = gen_tchw if not (ph or pw) else torch.nn.functional.pad(gen_tchw, (0, pw, 0, ph))
    r = ref_tchw if not (ph or pw) else torch.nn.functional.pad(ref_tchw, (0, pw, 0, ph))

    ll_g, hl_g, lh_g, hh_g = _split(g.float())
    ll_r, _, _, _ = _split(r.float())
    out = _merge(ll_r, hl_g, lh_g, hh_g)
    return out[..., :h, :w].to(gen_tchw.dtype)


def _sdpa_varlen_forward(self, *args, **kwargs):
    """SDPA stand-in for ``flash_attn_varlen_func`` (no flash-attn on Windows).

    q/k/v arrive packed as ``(L_total, H, D)`` with per-segment offsets in
    ``cu_seqlens_*``. We unpack into a padded batch, run masked SDPA — the same
    math as varlen flash attention — and repack. Fully vectorized (no Python
    loop over segments).
    """
    import torch
    import torch.nn.functional as F

    if "q" in kwargs:
        q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
    else:
        q, k, v = args[0], args[1], args[2]
    cu_q = kwargs["cu_seqlens_q"].to(torch.int64)
    cu_k = kwargs["cu_seqlens_k"].to(torch.int64)

    lens_q = cu_q[1:] - cu_q[:-1]
    lens_k = cu_k[1:] - cu_k[:-1]
    b, h, d = int(lens_q.numel()), q.shape[-2], q.shape[-1]
    lmax_q = int(lens_q.max()) if b else 0
    lmax_k = int(lens_k.max()) if b else 0

    def _pad(x, cu, lmax):
        ar = torch.arange(lmax, device=x.device)
        idx = (cu[:-1].unsqueeze(1) + ar.unsqueeze(0)).clamp_(min=0, max=x.shape[0] - 1).reshape(-1)
        return x.index_select(0, idx).view(b, lmax, h, d).transpose(1, 2)

    qp = _pad(q, cu_q, lmax_q)
    kp = _pad(k, cu_k, lmax_k)
    vp = _pad(v, cu_k, lmax_k)

    ar_k = torch.arange(lmax_k, device=q.device)
    attn_mask = (ar_k[None, :] < lens_k[:, None]).view(b, 1, 1, lmax_k)

    out = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=attn_mask)  # (b,h,lq,d)

    total_q = int(cu_q[-1])
    seg_of = torch.repeat_interleave(torch.arange(b, device=q.device), lens_q)
    off_of = torch.arange(total_q, device=q.device) - cu_q[:-1][seg_of]
    flat_src = (seg_of * lmax_q + off_of).view(-1)
    return out.transpose(1, 2).reshape(-1, h, d).index_select(0, flat_src)


class SeedVrRunner:
    """Runs SeedVR2-3B over whole segments (windowed internally).

    Construct with the segment's *input* resolution and target scale; call
    :meth:`load` once (VRAM-heavy), then ``runner(frames)`` per segment.
    """

    def __init__(
        self,
        weights_dir: str | Path,
        width: int,
        height: int,
        scale: int = 4,
        cfg_scale: float = 1.0,
        seed: int = 666,
        max_clip_frames: int = 129,
        overlap: int = 16,
        min_window_frames: int = 9,
    ):
        self.weights_dir = Path(weights_dir)
        self.src_w, self.src_h = int(width), int(height)
        # Model output dims: target rounded DOWN to a multiple of 16 (the official
        # DivisibleCrop(16, 16) semantics). The stage resamples to the exact
        # requested size when they differ.
        self.out_w = _floor_to(self.src_w * scale, 16)
        self.out_h = _floor_to(self.src_h * scale, 16)
        self.cfg_scale = float(cfg_scale)
        self.seed = int(seed)
        self.max_clip_frames = int(max_clip_frames)
        self.overlap = max(4, min(int(overlap), self.max_clip_frames // 2))
        self.min_window_frames = int(min_window_frames)

        self.torch = None
        self._runner = None      # VideoDiffusionInfer
        self._pos_emb = None     # (58, 5120) bf16
        self._neg_emb = None     # (64, 5120) bf16

    # -- setup -----------------------------------------------------------------

    def _load_config(self):
        """Load configs_3b/main.yaml and resolve the VAE ``__inherit__`` manually.

        The upstream ``load_config`` resolves inheritance paths relative to the
        CWD; we anchor them at the vendor directory instead so the runner works
        from any working directory (and inside a frozen bundle).
        """
        from omegaconf import OmegaConf

        from . import VENDOR_DIR, ensure_vendor_on_path

        ensure_vendor_on_path()
        vendor = VENDOR_DIR
        main = OmegaConf.load(str(vendor / "configs_3b" / "main.yaml"))
        inherit = main.vae.model.pop("__inherit__", None)
        if inherit:
            parent = OmegaConf.load(str(vendor / Path(inherit)))
            main.vae.model = OmegaConf.merge(parent, main.vae.model)

        # apex (FusedRMSNorm/FusedLayerNorm) needs a CUDA-toolkit build and is not
        # installable on Windows; diffusers RMSNorm / nn.LayerNorm are
        # mathematically identical, so swap the fused variants for plain ones.
        from omegaconf import DictConfig, ListConfig

        _NORM_FALLBACK = {"fusedrms": "rms", "fusedln": "layer"}

        def _unfuse(node):
            if isinstance(node, DictConfig):
                for k in list(node.keys()):
                    v = node[k]
                    if isinstance(v, str) and v in _NORM_FALLBACK:
                        node[k] = _NORM_FALLBACK[v]
                    elif OmegaConf.is_config(v):
                        _unfuse(v)
            elif isinstance(node, ListConfig):
                for i in range(len(node)):
                    if OmegaConf.is_config(node[i]):
                        _unfuse(node[i])

        _unfuse(main)
        return main

    def load(self) -> None:
        """Instantiate DiT + VAE on CUDA and load checkpoints (VRAM-heavy)."""
        import torch

        from . import ensure_vendor_on_path

        ensure_vendor_on_path()
        # Vendor path must be set before these imports resolve.
        from common.distributed import get_device  # noqa: F401 - device sanity
        from common.seed import set_seed
        from projects.video_diffusion_sr.infer import VideoDiffusionInfer

        cfg = self._load_config()
        # Checkpoints live in the model cache dir (absolute paths; upstream uses
        # CWD-relative ./ckpts/...).
        cfg.vae.checkpoint = str(self.weights_dir / "ema_vae.pth")
        dit_ckpt = str(self.weights_dir / "seedvr2_ema_3b.pth")

        # Single-GPU inference: the vendor code assumes a torch.distributed
        # process group exists (barriers and get_rank() in the VAE cache paths).
        # Init a 1-rank gloo group so every dist call is a well-defined no-op
        # instead of raising "Default process group has not been initialized".
        if not torch.distributed.is_initialized():
            import os as _os
            import socket as _socket

            _os.environ.setdefault("RANK", "0")
            _os.environ.setdefault("WORLD_SIZE", "1")
            _os.environ.setdefault("LOCAL_RANK", "0")
            with _socket.socket() as _s:
                _s.bind(("127.0.0.1", 0))
                _port = _s.getsockname()[1]
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{_port}",
                rank=0,
                world_size=1,
            )

        # flash-attn is not installable on Windows; swap FlashAttentionVarlen's
        # forward for a mathematically identical SDPA implementation. Patching
        # the class covers every import site in the vendor tree.
        from models.dit_v2 import attention as _attn_mod

        if (
            _attn_mod.flash_attn_varlen_func is None
            and not getattr(_attn_mod.FlashAttentionVarlen, "_aive_sdpa_patched", False)
        ):
            _attn_mod.FlashAttentionVarlen.forward = _sdpa_varlen_forward
            _attn_mod.FlashAttentionVarlen._aive_sdpa_patched = True

        runner = VideoDiffusionInfer(cfg)
        from omegaconf import OmegaConf

        OmegaConf.set_readonly(runner.config, False)  # official does this before mutating cfg

        log.info("SeedVR: loading DiT checkpoint %s", Path(dit_ckpt).name)
        runner.configure_dit_model(device="cuda", checkpoint=dit_ckpt)
        log.info("SeedVR: loading VAE checkpoint %s", Path(cfg.vae.checkpoint).name)
        runner.configure_vae_model()
        if hasattr(runner.vae, "set_memory_limit"):
            runner.vae.set_memory_limit(**runner.config.vae.memory_limit)

        # One-step inference (official inference_seedvr2_3b.py argparse defaults):
        cfg.diffusion.cfg.scale = self.cfg_scale
        cfg.diffusion.cfg.rescale = 0.0
        cfg.diffusion.timesteps.sampling.steps = 1
        runner.configure_diffusion()

        set_seed(self.seed)  # deterministic per job (noise state then advances per window)

        emb_dir = self.weights_dir
        self._pos_emb = torch.load(emb_dir / "pos_emb.pt", map_location="cpu", weights_only=False)
        self._neg_emb = torch.load(emb_dir / "neg_emb.pt", map_location="cpu", weights_only=False)
        log.info(
            "SeedVR runner ready (%dx%d -> %dx%d, cfg=%.1f steps=1, window=%d overlap=%d)",
            self.src_w, self.src_h, self.out_w, self.out_h,
            self.cfg_scale, self.max_clip_frames, self.overlap,
        )
        self._runner = runner

    # -- inference ---------------------------------------------------------------

    def __call__(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Upscale a segment's HWC uint8 frames; returns HWC uint8 at (out_h, out_w)."""
        import torch

        if self._runner is None:
            raise SeedVrError("SeedVrRunner.load() must be called before inference.")
        T = len(frames)
        if T == 0:
            return []
        h0, w0 = frames[0].shape[:2]
        if (h0, w0) != (self.src_h, self.src_w):
            raise SeedVrError(
                f"SeedVR input size {w0}x{h0} does not match the runner's "
                f"{self.src_w}x{self.src_h}; build one runner per resolution."
            )

        # RAM guard: float32 accumulator for the whole segment at target res.
        acc_bytes = T * self.out_h * self.out_w * 3 * 4
        if acc_bytes > 16 * 2**30:
            raise SeedVrError(
                f"SeedVR would need ~{acc_bytes / 2**30:.0f} GiB of RAM to buffer this "
                f"{T}-frame segment at {self.out_w}x{self.out_h}; use a lower target "
                "resolution or fps."
            )

        x = self._prepare_input(frames)  # (C, T, out_h, out_w) float32 [-1,1] on GPU

        window = self.max_clip_frames
        while True:
            try:
                return self._run_windows(x, window)
            except torch.cuda.OutOfMemoryError:
                if window <= self.min_window_frames:
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise SeedVrError(
                        f"SeedVR ran out of VRAM even at a {window}-frame window "
                        f"({self.out_w}x{self.out_h}). Close other GPU apps (e.g. an LLM "
                        "server) and retry, or process at a lower resolution."
                    ) from None
                log.warning(
                    "SeedVR CUDA OOM at %d-frame window; retrying segment with %d frames.",
                    window, max(self.min_window_frames, window // 2),
                )
                gc.collect()
                torch.cuda.empty_cache()
                window = max(self.min_window_frames, window // 2)

    def _prepare_input(self, frames: Sequence[np.ndarray]):
        """HWC uint8 frames -> (C,T,out_h,out_w) float32 in [-1,1] on GPU.

        Mirrors the official transform pipeline: bicubic resize to target dims
        (AreaResize is a plain bicubic), clamp [0,1], DivisibleCrop(16,16)
        (no-op — our targets are multiples of 16 by construction), Normalize
        (0.5, 0.5).
        """
        import torch

        from common.distributed import get_device

        arr = np.stack(frames)  # (T,H,W,3) uint8
        x = torch.from_numpy(arr).to(get_device(), dtype=torch.float32) / 255.0
        x = x.permute(0, 3, 1, 2)  # TCHW
        if (x.shape[2], x.shape[3]) != (self.out_h, self.out_w):
            x = torch.nn.functional.interpolate(
                x, size=(self.out_h, self.out_w), mode="bicubic", align_corners=False
            )
        x.clamp_(0.0, 1.0)
        # DivisibleCrop(16,16) — center crop; defensive no-op for our dims.
        ch = (x.shape[2] // 16) * 16
        cw = (x.shape[3] // 16) * 16
        if ch != x.shape[2]:
            top = (x.shape[2] - ch) // 2
            x = x[:, :, top:top + ch, :]
        if cw != x.shape[3]:
            left = (x.shape[3] - cw) // 2
            x = x[:, :, :, left:left + cw]
        x.sub_(0.5).div_(0.5)  # Normalize(0.5, 0.5) -> [-1, 1]
        return x.permute(1, 0, 2, 3).contiguous()  # CTHW

    def _run_windows(self, x_cthw, window: int) -> list[np.ndarray]:
        """Process the segment in windows of `window` frames with overlap blending."""
        import torch

        T = x_cthw.shape[1]
        # Full-length windows at stride intervals; only the LAST window is
        # shortened (to end exactly at T). Every consecutive overlap is then
        # exactly `self.overlap` frames, so the fixed-ramp weights below sum to 1.
        if T <= window:
            starts = [0]
        else:
            stride = max(1, window - self.overlap)
            starts = [0]
            s = 0
            while s + window < T:
                s += stride
                starts.append(s)

        acc = np.zeros((T, self.out_h, self.out_w, 3), dtype=np.float32)
        wsum = np.zeros(T, dtype=np.float32)

        for idx, s in enumerate(starts):
            e = min(s + window, T)
            out_np = self._process_clip(x_cthw[:, s:e])  # (n,out_h,out_w,3) float32 [-1,1]

            n = out_np.shape[0]
            wts = np.ones(n, dtype=np.float32)
            o = min(self.overlap, n)
            if s > 0:  # ramp up across the overlap with the previous window
                wts[:o] *= (np.arange(o) + 1) / (o + 1)
            if e < T:  # ramp down across the overlap with the next window
                wts[-o:] *= np.arange(o, 0, -1) / (o + 1)

            acc[s:e] += out_np * wts[:, None, None, None]
            wsum[s:e] += wts
            log.debug("SeedVR window %d/%d frames [%d:%d]", idx + 1, len(starts), s, e)

        with np.errstate(divide="ignore", invalid="ignore"):
            out = acc / wsum[:, None, None, None]
        del acc
        # [-1,1] -> uint8 (official: clip(-1,1).mul(0.5).add(0.5).mul(255).round())
        out = np.clip(out * 0.5 + 0.5, 0.0, 1.0)
        return [np.round(out[i] * 255.0).astype(np.uint8) for i in range(T)]

    def _process_clip(self, clip_cthw):
        """One clip (C,T,out_h,out_w) [-1,1] on GPU -> (T,out_h,out_w,3) float32 CPU."""
        import torch

        from common.distributed import get_device

        runner = self._runner
        dev = get_device()

        # VAE temporal factor is 4: pad T to ≡ 1 (mod 4) by repeating the last frame.
        t = clip_cthw.shape[1]
        tp = _pad_t(t)
        if tp != t:
            clip_cthw = torch.cat([clip_cthw, clip_cthw[:, -1:].repeat(1, tp - t, 1, 1)], dim=1)

        # Official memory dance: DiT off-GPU while the VAE encodes.
        runner.dit.to("cpu")
        runner.vae.to(dev)
        cond_latents = runner.vae_encode([clip_cthw])  # [(t', h, w, 16)] on GPU
        runner.vae.to("cpu")
        runner.dit.to(dev)

        noises = [torch.randn_like(latent) for latent in cond_latents]
        aug_noises = [torch.randn_like(latent) for latent in cond_latents]

        # Official generation_step with cond_noise_scale=0.0: the lerp schedule's
        # forward at t=0 is the identity, so the condition latent passes through
        # unchanged — replicate the call faithfully (timestep_transform included).
        cond_noise_scale = 0.0

        def _add_noise(x_latent, aug_noise):
            tt = torch.tensor([1000.0], device=dev) * cond_noise_scale
            shape = torch.tensor(x_latent.shape[1:], device=dev)[None]
            tt = runner.timestep_transform(tt, shape)
            return runner.schedule.forward(x_latent, aug_noise, tt)

        conditions = [
            runner.get_condition(noise, task="sr", latent_blur=_add_noise(lb, an))
            for noise, an, lb in zip(noises, aug_noises, cond_latents)
        ]

        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=noises,
                conditions=conditions,
                dit_offload=True,  # DiT -> CPU after sampling, back on GPU after decode
                texts_pos=[self._pos_emb.to(dev)],
                texts_neg=[self._neg_emb.to(dev)],
            )

        sample = video_tensors[0]  # (C, T', out_h, out_w) in [-1,1]
        del video_tensors, noises, aug_noises, conditions, cond_latents

        # Wavelet color fix against the exact condition input this clip saw.
        gen_tchw = sample.float().permute(1, 0, 2, 3)   # (T', C, H, W)
        ref_tchw = clip_cthw.permute(1, 0, 2, 3).float()  # (T', C, H, W)
        fixed = wavelet_color_fix(gen_tchw, ref_tchw)
        del sample, gen_tchw, ref_tchw

        out = fixed[:, :t].permute(0, 2, 3, 1).contiguous().cpu().numpy()  # trim padding
        return out
