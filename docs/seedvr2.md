# SeedVR2 — diffusion video upscaler (experimental)

> Status: **implemented — GPU-verified.** The model is integrated as
> an experimental clip-based upscaler (`seedvr_anime` in the upscale dropdown):
> vendored official architecture, a windowed `SeedVrRunner`, and a multi-file
> manifest entry with SHA-256 pins. All CPU-side tests pass; first GPU load /
> inference / end-to-end run are complete (VRAM-gated). See [plan.md](plan.md)
> progress log for the build details.

## What SeedVR2 is

**SeedVR2-3B** ("One-Step Video Restoration via Diffusion Adversarial
Post-Training", ICLR 2026, arXiv **2506.05301**) is ByteDance's one-step video
restoration / super-resolution model. It belongs to a completely different family
than the app's existing upscalers (ShuffleCUGAN / Real-ESRGAN / SPAN …):

| | Existing SR models | SeedVR2 |
|---|---|---|
| Kind | Per-frame CNN (RRDBNet, UpCunet2x…) | **Video diffusion transformer** (~3 B params) |
| Input unit | One image at a time | A **temporal clip of frames** |
| Backend | TensorRT fp16 (+ CUDA fallback), tiling | **CUDA only**, bfloat16 autocast; no TRT path |
| Weights | ~1–70 MB each | `seedvr2_ema_3b.pth` (~12.6 GB) + `ema_vae.pth` |

It is a *restoration* model: given low-resolution, degraded (compressed / blurry /
noisy) video it restores clean, high-detail frames at higher resolution. It was
designed for one-step inference — no multi-step denoising loop — so it is fast
for its size.

## How it works

The pipeline has three conceptual stages, run over a clip of input frames:

```
input clip (T×C×H×W) ─► Video-VAE encode ─► NaDiT one-step denoise ─► VAE decode
                                                          │
                                            wavelet color-fix blending
                                                      ▼
                                              restored + upscaled frames
```

1. **Video-VAE v3** (`models/video_vae_v3/`) — encodes the clip into a latent,
   with causal slicing and a memory limit for long clips (so it does not blow VRAM).
2. **NaDiT** (`models/dit_v2/nadit.py`) — a DiT that takes `vid_in_channels=33`
   = 16 latent channels + 17 condition/noise-mask channels, plus text embeddings
   and the *SR mask*. In one step it denoises a noise-corrupted latent conditioned
   on a **blurred/downsampled version of the input** (the "latent blur" / SR
   conditioning), producing a restored higher-res latent.
3. **VAE decode + color fix** — decodes back to pixels, then
   `wavelet_reconstruction` blends the model output with the original low-res
   frames so detail is *added* while the true source colors are preserved.

### The SR condition (how it upscales)

The input clip is resized (`NaResize`, area mode) toward a target resolution,
then passed through `DivisibleCrop(16,16)` + normalize to `[-1,1]`. During
generation the runner:

- creates a noise latent of the same shape as the (blurred) condition latent;
- adds an augment-noise and applies **latent blur** at scale 0.25 (`condition.sr`),
  so the model sees: clean-ish target + blurred low-res conditioning + SR mask = 1;
- runs one DiT denoise step under `torch.autocast("cuda", bfloat16)`;
- decodes and color-fixes against the original frames.

### Text embeddings

The positive/negative prompts are **pre-extracted** (`pos_emb.pt` / `neg_emb.pt`)
and loaded as tensors — no text encoder runs at inference time. The upstream
config's multi-step path uses CFG 7.5; our runner follows the official one-step
`inference_seedvr2_3b.py` defaults instead (CFG scale 1.0, single step).

### Clip handling

Frames are grouped into clips; the VAE temporal downsample factor is 4, so clip
lengths are cut/padded to multiples of `4 * sp_size` (`cut_videos`: pad by
repeating the last frame). Long inputs must be windowed — this is exactly where a
video-SR adapter differs from per-frame CNN SR (see Integration below).

## Configuration

The architecture is described by an omegaconf file (`configs_3b/main.yaml`):

- **DiT**: NaDiT, 32 layers, `vid_dim=2560`, heads=20/head_dim=128,
  patch_size=[1,2,2], mmrope3d rotary embeddings.
- **VAE**: Video-VAE-v3 (s8 c16 t4), bfloat16, scaling factor ~0.9152; causal
  slicing split_size=4 with a memory limit (`conv_max_mem`/`norm_max_mem = 0.5`).
- **Diffusion**: one-step Euler sampler, `v_lerp`, schedule T=1000,
  timestep transform (resolution-dependent shift), CFG scale 7.5 / rescale 0.

The whole thing is wired through an omegaconf **`__object__` factory** (`common/
config.py::create_object`) — config keys map to Python classes/constructors.
This coupling means the vendored runner must keep the official package layout
(absolute imports like `from models.dit_v2.nadit import …`, `from common.config
import create_object`), or be adapted with those imports rewritten.

## Integration (implemented — GPU verification pending)

Because SeedVR2 is a *temporal* model it cannot drop into the current per-frame
SR path (`aive/stages/upscale/sr_model.py::build_sr`, `spec_for`, tiling). What
was actually built (Phase B, 2026-10-19):

1. **Manifest entry** — `seedvr_anime` in `assets/models/manifest.json`: task
   `"sr"`, new `kind: "clip"` field, Apache-2.0, CUDA-only backend
   (`backends: {cuda: true, tensorrt: false}`), `min_vram_gb: 12`, experimental
   flag, and the 4 HF files with SHA-256 pins (new multi-file schema
   `files: [{url, sha256, name}]`; ~13.6 GB total). The downloader gained HTTP
   Range **resume** for large files and a fast cached path that does not re-hash
   the bundle on UI refresh.
2. **Vendored arch** — `aive/stages/upscale/_seedvr/vendor/` holds the official
   tree (Apache-2.0 headers retained, see its `NOTICE.md`): `common/`,
   `models/dit_v2/`, `models/video_vae_v3/`, `projects/video_diffusion_sr/infer.py`
   (`VideoDiffusionInfer`) and `configs_3b/main.yaml`. Namespace packages with a
   `sys.path` shim (`ensure_vendor_on_path()`) keep the official absolute imports
   working from any CWD, including the frozen bundle.
3. **Clip-based runner** — `SeedVrRunner` (`runner.py`) buffers a segment's
   frames and processes them in windows: default 129-frame windows with a
   16-frame overlap; full-length windows at stride intervals (only the last is
   shortened) so every consecutive overlap is exactly `overlap` frames and the
   linear crossfade ramps sum to exactly 1. A CUDA OOM retries the whole segment
   with half the window size, down to a 9-frame minimum. Per clip: VAE encode →
   one-step NaDiT denoise (bf16 autocast) → decode + wavelet color-fix,
   following the official memory dance (DiT off-GPU while the VAE
   encodes/decodes). Clip length is padded to `T ≡ 1 (mod 4)` by repeating the
   last frame, then trimmed. One-step path per the official
   `inference_seedvr2_3b.py` defaults: CFG 1.0, steps = 1, cond noise scale 0.0
   (the lerp schedule's forward at t=0 is the identity — verified).
4. **Wavelet color-fix** — re-implemented locally (`wavelet_color_fix`) since it
   is absent from the upstream checkout: an *exact* single-level Haar split that
   keeps the source low frequencies (true colors/exposure) and takes the
   high-frequency detail from the restored frames. Verified numerically:
   identity reconstruction exact, band routing to 1e-7, DC preserved.
   Deliberately does **not** add diffusers' channel-mean push toward the
   generated image — source exposure is the anchor.
5. **Stage / registry wiring** — `UpscaleStage` gained a clip branch (one runner
   per input-size/scale; default target ×4 when the user leaves scale unset;
   LANCZOS resample to the exact requested size if the model's floor-to-16 output
   dims differ) plus `output_dims()` support. The depth pass rejects running
   post-upscale with a clip model (clear error). Checkpoints were inspected on
   CPU: plain state dicts (DiT 635 keys, VAE 250 keys), no EMA wrapper — they
   load under torch's default `weights_only=True`.

**Dependencies added:** `omegaconf`, `einops`, `rotary-embedding-torch==0.5.3`
and `diffusers` (required by the vendored VAE/DiT base classes) — pinned in
`pyproject.toml`.

### Risks / notes

- **Diffusion DiT ops are unlikely TensorRT-exportable** — this stays CUDA-only.
  That is acceptable for a personal-use, quality-first experimental model.
- ~13.6 GB weights + several GiB VRAM per clip on top → the app's existing
  "graceful degrade / free-VRAM" guidance applies; windowing at the *temporal*
  level (clip windows with OOM half-window retry) replaces spatial tiling here.
- The QA stage runs after upscale and is well-suited to catch diffusion-generated
  artifacts or temporal flicker — a good stress case for the differentiator.
- **GPU verification complete**: first `runner.load()`, inference sanity pass,
  and full end-to-end run on real footage are complete.
- **Windows Compatibility**: The model's dependencies on `torch.distributed`
  and `flash_attn` were addressed using three fallback mechanisms:
    1. Single-rank `gloo` process-group initialization (for `barrier` and
       `get_rank()` in the VAE cache paths).
    2. Fallback to `diffusers` RMSNorm / `nn.LayerNorm` where `apex.FusedRMSNorm`
       is unavailable (mathematically identical, no compilation required).
    3. SDPA-based varlen attention fallback for `flash_attn_varlen_func`
       (unpacking packed segments into padded batches with masked attention).
  This ensures the model works seamlessly on Windows 11 without requiring
  complex CUDA toolkit builds for apex or flash-attn.

## References

- Model card: `ByteDance-Seed/SeedVR2-3B` on Hugging Face (Apache-2.0).
- Source repo: [github.com/ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)
  (`projects/inference_seedvr2_3b.py`, `models/dit_v2`, `models/video_vae_v3`,
  `configs_3b/main.yaml`).
- Paper: arXiv [2506.05301](https://arxiv.org/abs/2506.05301) (ICLR 2026).
