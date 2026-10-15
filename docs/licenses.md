# License Audit — AI Video Enhancer

**Audit date: 2026-09-22 (M8c).** Scope: everything the app ships or downloads —
app code, vendored model architectures, runtime dependencies, and all 20 model
weights in [`assets/models/manifest.json`](../assets/models/manifest.json).

Context: this is a **personal-use** application (project decision D3), so
non-commercial (NC) and research-only weights are acceptable here. They are
nevertheless marked with ⚠️ badges in the UI model list and documented below,
so nothing is silently under-licensed. If the app were ever redistributed, the
flagged items in §4 would need to be removed or cleared first (§5).

**Method.** Every weight file is sha256-pinned in the manifest (a tampered or
swapped download fails verification and is rejected). Licenses below were checked
against upstream license files / release pages at integration time (per-milestone
dates in §4) and re-verified on 2026-09-22 where noted.

---

## 1. App code & vendored architectures

| Component | License | Source | Notes |
|---|---|---|---|
| Application code (`aive/`, `tools/`, `packaging/`) | **MIT** (this repo's [LICENSE](../LICENSE)) | ours | Design §8 proposed MIT; adopted. |
| `_cugan_arch.py` (ShuffleCUGAN / Fallin) | **MIT** | vendored from spandrel's `RealCUGAN/__arch/upcunet_v3.py`, which derives from bilibili/Real-CUGAN-models (MIT, © 2021 bilibili) | See finding F1 below — both lineages are MIT. Attribution header in file. |
| `_rrdbnet_arch.py` (Real-ESRGAN x4plus) | **BSD-3** | vendored from BasicSR (xinntao), BSD-3 | RRDBNet arch only; weights separately licensed (§2). |
| `_span_arch.py` (SPAN V2) | **Apache-2.0** (+ Conv3XC: MIT, TArdelean/DynamicConvolution) | vendored from umzi2/traiNNer-redux | Attribution header in file. |
| `_compact_arch.py` (Cyte V1 / OpenProteus) | **Apache-2.0** | umzi2's modified BasicSR `SRVGGNetCompact` (BasicSR family, Apache-2.0) | Attribution header in file. |
| `_gmfss/` (GMFSS) | code **MIT** (© 2023 98mxr); our fp16 dtype fixes are ours | vendored from GMFSS_Fortuna | Weights separately flagged (§4). |
| `_m2m/` (M2M) | **Adobe Research License** (non-commercial research only, no redistribution) | vendored from M2M_VFI (feinanshan); cupy kernels rewritten in pure torch by us with identical math | Code *and* weights under the Adobe license — personal use only (§4). |
| `depth/_da3/`, `_limbo_export.py` etc. (DA3/Limbo) | **Apache-2.0** (Depth-Anything-3 runtime) | vendored from Depth-Anything-3 | Limbo checkpoints are anime finetunes of DA3-small (§4). |
| SCUNet arch (`qa/repairs/_scunet_arch.py`) | **Apache-2.0** | cszn/SCUNet | Weights: personal-use TAS mirror of KAIR v1.0. |
| deepDeband-f arch (`qa/repairs/_deepdeband_arch.py`) | **BSD-3 + MIT** (pytorch-CycleGAN-and-pix2pix / RaymondLZhou/deepDeband) | vendored from both | Weights: personal-use TAS mirror. |

## 2. Model weights (manifest, all 20)

| Model | Task | Weights license | Host | ⚠️ NC/research | Verified |
|---|---|---|---|---|---|
| shufflecugan_x2 | sr | personal use (TAS mirror of bilibili Real-CUGAN family) | TAS-Models-Host | — | M3 09-17; sha-pinned |
| fallin_soft / fallin_strong | sr | personal use (TAS mirror) | TAS-Models-Host | — | M3.x 09-17; sha-pinned |
| span_v2 | sr | personal use (TAS mirror) | TAS-Models-Host | — | M3.x 09-17; sha-pinned |
| cyte_v1 | sr | **CC BY-NC 4.0** | NevermindNilas/Ai-models | ⚠️ NC | M3.x 09-17 (repo README); sha-pinned |
| openproteus | sr | personal use (TAS mirror) | TAS-Models-Host | — | M3.x 09-17; sha-pinned |
| realesrgan_x4plus | sr | **non-commercial** (release-page terms; code is BSD-3) | xinntao/Real-ESRGAN releases | ⚠️ NC | M3 09-17; re-checked LICENSE (BSD-3) + README 2026-09-22; sha-pinned |
| rife_425 / _lite | fi | **MIT** (Practical-RIFE) | Google Drive (official) / TAS mirror | — | M2 09-17; sha-pinned |
| rife_425_heavy | fi | MIT (community mirror of the same arch family) | TAS-Models-Host | — | M2 09-17; sha-pinned |
| gmfss | fi | **no license file upstream** (routineLife1/GMFSS, archived); code MIT via GMFSS_Fortuna | TAS-Models-Host union bundle | ⚠️ unlicensed weights | M2.x 09-21 (repo inspection); Fortuna LICENSE re-read 2026-09-22; sha-pinned |
| m2m | fi | **Adobe Research License** — non-commercial research only, no redistribution | Google Drive (official) | ⚠️ research-only | M2.x 09-21; upstream LICENSE re-read 2026-09-22; sha-pinned |
| amt_s / amt_l | fi | **CC BY-NC 4.0** code + **CC BY-NC-SA 4.0** weights (MCG-NKU/AMT) | Hugging Face (lalala125/AMT, official) | ⚠️ non-commercial | AMT integrated 2026-10-15; sha-pinned (amt-s `07e7e034…`, amt-l `985a3569…`); strict CPU load verified |
| scunet | restore | personal use (TAS mirror of KAIR v1.0 color-real-psnr; arch Apache-2.0) | TAS-Models-Host | — | M4.x 09-19 (strict CPU load vs real weights); sha-pinned |
| deepdeband_f | restore | personal use (TAS mirror; archs BSD-3 + MIT) | TAS-Models-Host | — | M4.x 09-19 (strict CPU load); sha-pinned |
| depth_anything_v2_small | depth | **Apache-2.0** (Depth-Anything-V2) | Hugging Face (official) | — | M5 09-20; sha-pinned |
| limbo_v1 / limbo_v2 | depth | **CC BY-NC 4.0 + TAS commercial grant** (NevermindNilas/Ai-models) | TAS-Models-Host / Ai-models | ⚠️ NC (grant covers TAS ecosystem use; personal use OK per D3 either way) | M5 09-20 (repo README); sha-pinned |
| depth_anything_3_small / _base | depth | **Apache-2.0** (Depth-Anything-3) | TAS-Models-Host (official checkpoints) | — | M5 09-20; sha-pinned |

## 3. Runtime dependencies (bundled in the frozen build)

| Dependency | License | Notes |
|---|---|---|
| Python 3.14 | PSF license | |
| PyTorch 2.12 (+cu132 CUDA runtime DLLs inside `torch/lib`) | BSD-3 (NVIDIA EULA for the CUDA toolkit components) | |
| TensorRT 11.1 | NVIDIA Software License Agreement | Standard for TRT; personal use fine. |
| onnxruntime, numpy/scipy | MIT / BSD | CPU-only ORT used for verification paths. |
| OpenCV (opencv-python) | Apache-2.0 | |
| PySide6 6.11 | **LGPL-3.0** | Linked as a plugin-based Qt app; LGPL obligations are satisfied by dynamic linking + no modification of Qt itself. |
| scenedetect, Pillow, requests, … | MIT / BSD / Apache-2.0 | Small pure-Python deps. |
| FFmpeg (Gyan *full* build: ffmpeg.exe + ffprobe.exe) | **LGPL with GPL components** (libx264 etc.) | Bundled as separate executables under `<install>\ffmpeg\` — mere aggregation, not linked into our code. Fine for personal use; a redistribution would need the LGPL/GPL terms honored or an FFmpeg build without GPL encoders. |

## 4. ⚠️ items from design §8 — status (all resolved)

| Item | Resolution |
|---|---|
| Real-ESRGAN weights non-commercial | Accepted per D3; ⚠️ badge in UI model list; documented here. Code is BSD-3 (re-verified 2026-09-22). Note: the current upstream README no longer repeats the NC statement — it lives on the release page, checked at M3. |
| Cyte V1 weights CC BY-NC 4.0 | Accepted per D3; ⚠️ badge in UI; documented here. |
| Limbo v1/v2 CC BY-NC 4.0 (+ TAS commercial grant) | Accepted per D3 (the grant is a bonus, not relied upon); ⚠️ badge in UI; documented here. |
| M2M Adobe Research License (code **and** weights) | Personal use only — no redistribution claims anywhere in the app or docs; ⚠️ badge in UI; documented here. The pure-torch softsplat/costvol rewrites are our own code implementing the published math, but the vendored arch files remain under the Adobe license. |
| GMFSS weights: no license file upstream | Personal use only (absence of a license = all rights reserved by default); ⚠️ badge in UI; documented here. The *code* we vendor is MIT (GMFSS_Fortuna, re-verified 2026-09-22). We track the best-maintained community variant per design §8 and would re-check terms if the source changes. |
| AMT (S/L) CC BY-NC(-SA) | Personal use only — non-commercial weights; ⚠️ badge in UI; documented here. Code vendored from MCG-NKU/AMT (CC BY-NC 4.0); checkpoints from the official Hugging Face repo `lalala125/AMT` (CC BY-NC-SA 4.0). Integrated 2026-10-15 as CUDA-only runners under `_amt/`. |
| **Spandrel GPL-3.0 concern** | **Resolved — the premise was wrong.** Design §8 listed Spandrel as GPL-3.0; on 2026-09-22 we verified spandrel's LICENSE is **MIT** (© 2024 The ChaiNNer Organization). Our `_cugan_arch.py` vendors a subset of its `upcunet_v3.py`, which itself derives from bilibili/Real-CUGAN-models (also MIT — the file retains the original Chinese comments; the bilibili repo is now deleted, 404, weights live on via TAS mirrors). Both lineages are permissive → **no GPL contamination, no subprocess isolation needed in v1.** The app has *no runtime dependency* on spandrel at all. |
| TAS / Video2X (AGPL-3.0) reference-only | Confirmed: zero code copied from either; we use TAS's model-host URLs and study its patterns (depth chunk streaming, aspect handling). Mere reference creates no license obligation. |

## 5. Redistribution note

This app is built for **personal use only**. The bundle as shipped must not be
redistributed because it can fetch NC/research-only weights (§2) and bundles a
GPL-component FFmpeg build (§3). If distribution ever becomes a goal: drop the
⚠️-flagged models from the manifest (the registry makes that a one-line change
each), swap FFmpeg for an LGPL-clean build, and re-run this audit.

## 6. Verification log

| Date | What was verified | How |
|---|---|---|
| 2026-09-17 (M2/M3) | RIFE MIT; Real-ESRGAN code BSD-3 + weights NC; CUGAN-family archs MIT; SPAN Apache-2.0; Cyte CC BY-NC; OpenProteus personal-use | Upstream repos/release pages at integration time; sha256 pins added to manifest |
| 2026-09-19 (M4.x) | SCUNet arch Apache-2.0 + checkpoint identity (strict CPU load, 540 keys); deepDeband-f BSD-3+MIT archs + checkpoint identity (strict CPU load, 54.4M params) | Strict loads against the real weight files; manifest sha pins |
| 2026-09-20 (M5) | DAv2/DA3 Apache-2.0; Limbo CC BY-NC + TAS grant (Ai-models README); checkpoint key verification for all five depth models | Upstream repos/HF cards; strict CPU loads; manifest sha pins |
| 2026-09-21 (M2.x) | GMFSS code MIT (Fortuna), weights unlicensed upstream; M2M Adobe Research License (code+weights); multi-file bundle integrity | Repo inspection + downloads; manifest sha pins |
| 2026-09-22 (M8c, this audit) | FILM Apache-2.0; M2M Adobe license text re-read; GMFSS_Fortuna MIT re-read; TAS AGPL noted; **spandrel LICENSE = MIT** + `upcunet_v3.py` provenance (bilibili-derived); Real-ESRGAN LICENSE (BSD-3) + current README re-checked; bilibili/Real-CUGAN-models confirmed deleted (404) | Local upstream clones in the session scratchpad + raw fetches from GitHub (spandrel, Real-ESRGAN); results recorded above |
| 2026-10-15 (FILM removal) | **FILM removed from product** — no longer bundled or shipped, so its Apache-2.0 license is no longer relevant to distribution; all other model licenses unchanged | Code/manifest/bundle inspection after removal; full test suite green |
