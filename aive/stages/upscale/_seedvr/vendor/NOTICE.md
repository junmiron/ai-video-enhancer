# Vendored: ByteDance SeedVR (Apache-2.0)

This directory vendors a subset of the official **SeedVR** repository for use in
the AI Video Enhancer app's experimental SeedVR2 video upscaler.

- Source: <https://github.com/ByteDance-Seed/SeedVR>
- License: Apache License 2.0 (c) 2025 Bytedance Ltd. and/or its affiliates
- Paper: "One-Step Video Restoration via Diffusion Adversarial Post-Training"
  (ICLR 2026), arXiv [2506.05301](https://arxiv.org/abs/2506.05301)

## Vendored modules

| Path | Upstream source |
|---|---|
| `common/` | `common/config.py`, `common/diffusion/*`, `common/distributed/*`, helpers (config factory, diffusion schedule/sampler/timesteps, device utils) |
| `models/dit_v2/` | NaDiT architecture (`nadit.py` + attention/embedding/mlp/mm/modulation/normalization/rope/window/nablocks/patch) |
| `models/video_vae_v3/` | Video-VAE-v3 (s8 c16 t4 inflation SD3) modules |
| `projects/video_diffusion_sr/infer.py` | `VideoDiffusionInfer` — the official inference wrapper (DiT/VAE setup, encode/decode, one-step sampling dispatch). Copied verbatim; its upstream absolute imports resolve via the vendor dir on `sys.path`. |
| `configs_3b/main.yaml` | SeedVR2-3B architecture config (omegaconf `__object__` factory) |

## Local modifications (ours, on top of Apache-2.0 code)

- `models/dit_v2/attention.py`: the top-level `from flash_attn import
  flash_attn_varlen_func` was made **optional** (try/except). The default
  inference path uses `TorchAttention` (PyTorch SDPA), so `flash_attn` is not a
  hard dependency; `FlashAttentionVarlen` raises a clear error if selected while
  the package is absent.
- The upstream repo's `projects/video_diffusion_sr/color_fix.py` (wavelet
  reconstruction) is **not present in the official checkout**, so it was not
  vendored. It is re-implemented locally as
  `aive/stages/upscale/_seedvr/runner.py::wavelet_color_fix` (single-level Haar,
  exact split/merge; source low-freq + restored high-freq).
- The official inference entry point (`projects/inference_seedvr2_3b.py`) is not
  vendored — the app's `SeedVrRunner` replaces it: single-GPU only (no
  distributed init / `sync_data`), CWD-independent config loading, and clip
  windowing with overlap blending for long segments.

All other vendored files are byte-for-byte copies of upstream (Apache-2.0). The
full Apache-2.0 license text is retained in each file's header and in the repo
root `LICENSE`.
