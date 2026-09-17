# Implementation Plan — AI Video Enhancer

> Status: **In execution** — M0+M1 ✅ (`9e8c151`) · M2 ✅ (`b164e51`, `3bb5b05`, `ce217c3`) · M3 ✅ (core) · next: M4 artifact QA stage
> v2 approved by user 2026-09-17 ("execute plan") · Companion: [design.md](design.md)
> Delivery model: incremental milestones M0–M7. Each milestone ends with something runnable you can try and react to. Estimates are agent-assisted development effort, not calendar time.

---

## Milestone overview

| # | Milestone | Contents | Est. | Demo point for you |
|---|---|---|---|---|
| M0 | Scaffolding & foundations | Repo, Blackwell toolchain (CUDA 12.8 / torch cu128 / TensorRT 10.x), config, logging, FFmpeg I/O round-trip | 1–2 d | Passthrough of a real video works; GPU/TRT detected correctly on the 5090 |
| M1 | Pipeline core + model manager | Worker process, job queue, progress/cancel, checkpoint store, model registry + downloader, **inference backend layer (TRT primary / CUDA fallback)** | 2–3 d | Headless enhance with live progress; models auto-download & cache; first TRT engine builds |
| M2 | Interpolation stage | RIFE 4.25/4.25-heavy/4.22-lite adapters + TRT engines (N×, UHD mode), AMT-SICDIA experimental; scene-cut protection, dedup option | 3–4 d | 24→60 fps clip with no cut-morphing; speed benchmark vs. PyTorch-only baseline |
| M3 | Upscale stage | Anime lineup (ShuffleCugan, Fallin Soft/Strong, SPAN, Cyte V1) + Real-ESRGAN live-action; full-frame inference at 32 GB; tiling safety valve; SwinIR/HAT + SAFA experimental | 3–4 d | 720p→1080p/4K upscale of the standard test clip, side-by-side vs. TAS output |
| M4 | Artifact QA stage ⭐ | Detectors (§4.2), hybrid CV+AI repairs (deepDeband-f, SCUNet, FastLineDarken…), verify pass, report data model + JSON/PNG export | 5–7 d | Injected-artifact test: app finds & fixes synthetic flicker/ghost/banding; real clip with before/after metrics |
| M5 | Depth map stage | DA v2 + Limbo v1/v2 adapters (TRT streaming export), DA3 video-streaming experimental; 10-bit grayscale MP4 / 16-bit PNG output | 2–3 d | Anime clip → depth video side-by-side with TAS's Limbo output |
| M6 | UI (PySide6) | Home/drop zone, settings+presets (incl. depth toggle), preview compare slider, processing view, report view, first-run model download UX | 4–6 d | Full GUI flow on a real video; UI stays responsive during a 1080p job |
| M7 | Polish & packaging | Preset tuning, error handling, **x265 10-bit anime-profile encode defaults**, PyInstaller + Inno Setup installer (bundles CUDA/TRT runtime) | 2–3 d | Clean-VM install → enhance with zero manual steps |
| M8 | Hardening & docs | Benchmark automation, user guide, license audit doc, beta feedback loop | ~2 d | Beta build for your review |

**Total: roughly 4–4.5 weeks of focused work**, delivered in the order above so you can steer after each milestone. (Note: interpolation lands before upscaling to match the TAS pipeline order — both are needed together for a full demo, which arrives at M3.)

---

## M0 — Scaffolding & foundations
- Repo init at `D:\Code\ai-video-enhancer`: Python 3.11 venv, `pyproject.toml`, ruff + mypy + pytest config, folder layout:
  ```
  aive/            # package
    core/          # pipeline orchestrator, job queue, events
    stages/        # ingest, scenes, dedup, interpolate/, upscale/, qa/
    inference/     # backend layer: TRT engine builder/cache, torch runner, ONNX export helpers
    models/        # registry manifests + loaders (incl. Spandrel-compatible custom import)
    ui/            # PySide6 (M5+)
  tests/           # unit + integration
  assets/models/   # registry JSON manifests
  docs/
  ```
- **Blackwell toolchain:** pin torch cu128, TensorRT 10.x (via `tensorrt` pip wheel or bundled), CUDA 12.8 runtime; GPU capability check (sm_120) with clear error if unsupported card.
- Config system (JSON in `%APPDATA%`), structured logging to file + console.
- FFmpeg integration: bundled static build discovery/download; `probe()` metadata API; decode→encode round-trip with passthrough.

**Acceptance:** passthrough of a real video produces valid output with correct duration/audio on the 5090 machine; startup log reports CUDA + TensorRT versions and GPU name; CI (lint+tests) green on Windows.

## M1 — Pipeline core + model manager
- Worker process architecture: GUI submits jobs → worker executes stages in sequence; event stream (stage start/progress/frame counts/ETA/warnings); pause/cancel semantics.
- Frame buffer with RAM budget + disk spill policy; checkpoint store writing per-segment FFV1 chunks; resume-from-last-completed-segment on restart.
- Model registry: JSON manifests (name, task, version, URLs, SHA-256, min VRAM, backend matrix, license); downloader with progress events + checksum verification; cache dir management.
- **Inference backend layer:** unified `run_model(frames)` interface → TensorRT path (ONNX load or export-from-torch on first use → FP16 `.engine` build → cache keyed by GPU/model/precision/shape) with PyTorch-CUDA fallback for non-exportable models; "Preparing model…" progress events during engine builds.

**Acceptance:** 60 s clip runs end-to-end in worker mode (passthrough stages) with live progress and clean cancel/resume; killing the process mid-run resumes without re-doing finished segments; a test model's TRT engine builds once, is cached, and second run skips the build.

## M2 — Interpolation stage
- RIFE 4.25 / 4.25-heavy / 4.22-lite adapters: arbitrary multiplier via recursive midpoint interpolation, target-FPS mode, UHD mode auto above 1080p source, fp16; TRT engines per variant (fixed shape profiles + dynamic timestep where clean).
- AMT-SICDIA adapter (experimental flag; also the ghost-repair alternate model).
- Additional FI adapters (experimental flags): **M2M** (efficient at 4×–8× multipliers — motion estimated once per pair), **GMFSS** (anime; track best-maintained community variant + license re-check at build time), **FILM** (Apache-2.0, benchmark quality).
- Scene-cut protection (never interpolate across cuts); optional frame de-duplication for anime (sensitivity setting, TAS/Flowframes-style).
- Audio timeline handling: output FPS change with audio passthrough — verify against Flowframes/TAS behavior.

**Acceptance:** 24→60 and 30→120 on test clips; zero morphing across cuts (automated check: cut frames in output match source cut frames); benchmark table recorded: TRT vs. PyTorch speed per variant at 720p/1080p.

## M3 — Upscale stage
- Loader layer for anime SR architectures (Spandrel-compatible state-dict auto-detection where possible; dedicated adapters otherwise): ShuffleCugan x2, Fallin Soft/Strong, SPAN, Cyte V1, OpenProteus, AniScale 2 (experimental), Real-ESRGAN (live action).
- Full-frame inference at 32 GB VRAM by default (fp16); tiled inference with overlap blending as safety valve for >4K targets; auto tile sizing from free VRAM.
- Experimental: SwinIR/HAT adapter; SAFA temporal-SR adapter (license check first — design §8).
- Benchmark harness: standard test clip (Video2X's 240p anime clip) + a live-action clip; log time/VRAM per model; visual comparison vs. TAS output for the same models.

**Acceptance:** 720p→1080p and →4K outputs visually on par with TAS for the same models; no tile seams at 4K (visual check); benchmark table recorded in `docs/benchmarks.md`. **First full-pipeline demo: dedup → interpolate → upscale end-to-end.**

## M4 — Artifact QA stage ⭐ (the differentiator)
- Proxy stream generation (720p) after upscaling.
- Detectors, each a small module with unit-testable core: flicker (temporal σ in low-flow regions), ghosting (dual-similarity + flow inconsistency masks), banding (histogram plateaus), halo/ringing (edge overshoot vs. source edge profile), line-art degradation (line-mask drift vs. source, anime), color drift (per-scene histogram tracking), regression frames (IQA z-score).
- Findings data model → repair policies per design §4.3: masked temporal smoothing + SCUNet stabilizing pass; neighbor-blend deghost + alternate-RIFE re-interp; **deepDeband-f** AI deband (+ classical fallback); edge softening; **FastLineDarken/Anime1080fixer** line repair (masked); scene color match; fallback-model frame reprocess.
- Verify pass: re-run detectors on repaired segments; record before/after deltas.
- Report generator: JSON export + PNG summary image (table + marked frames).

**Acceptance:** **artifact-injection harness** — take a clean processed clip, inject synthetic flicker / ghosting / banding into known regions → detectors catch ≥90% of injected regions with <10% false positives; on real clips, reported metrics improve after repair and the report renders correctly.

## M5 — Depth map stage
- Adapters: Depth-Anything V2 Small/Base (+faster variants) per-frame; **Limbo v1/v2** video streaming with chunk windows + overlap alignment, TensorRT engine export following TAS's validated path (FP16 with sensitive ops kept in FP32); DA3 `video_small_v3`/`base_v3` streaming as experimental.
- Output: 10-bit grayscale MP4 (default) or 16-bit PNG sequence; runs on source frames by default, post-upscale option behind Advanced (§3.4).
- Scene-cut handling for depth (no cross-cut blending inside streaming windows).

**Acceptance:** anime test clip → depth video visually on par with TAS's Limbo output; TRT streaming throughput within ~20% of TAS's published numbers at the same window size; PNG sequence round-trips losslessly.

## M6 — UI (PySide6)
- App shell: dark theme, GPU detection banner ("RTX 5090 — CUDA + TensorRT ready"), first-run model download flow with progress + sample test clip offer.
- Home screen: drop zone, queue list with per-file status, source info cards, estimated output size/time before start.
- Settings panel exactly as design §7.2 (presets drive all defaults; every field overridable).
- Preview: frame picker + split-slider compare + timeline artifact markers (post-run data from M4 report model).
- Processing view: per-stage progress, ETA, VRAM/RAM meters, collapsible log, pause/cancel; queue auto-advance.
- Report view as design §7.3 with JSON/PNG export.

**Acceptance:** full flow GUI → output file on a real video; UI remains responsive (measured) while processing 1080p content; all settings persist across restarts.

## M7 — Polish & packaging
- Preset tuning pass using your feedback from the M2–M6 demos.
- **Encode defaults:** libx265 main10, preset slow, CRF 19 + anime-oriented `x265-params` (anime profile; x265 v4 removed the old tunes); NVENC HEVC 10-bit fast option; H.264 fallback; container MP4/MKV; audio passthrough default.
- Error handling: no GPU / unsupported arch, insufficient VRAM, disk full mid-run, corrupt/unsupported input, model download failure (retry + mirror), TRT engine build failure (auto-fallback to PyTorch with notice).
- PyInstaller onedir → Inno Setup installer bundling Python + CUDA/TRT runtime + FFmpeg; auto-update check stub (manual "check for updates" in v1).

**Acceptance:** clean Windows VM: install → launch → drop video → done, with zero manual steps and no console windows; uninstall is clean.

## M8 — Hardening & docs
- Automated benchmark suite (runs on demand, appends to `docs/benchmarks.md`).
- User guide (`docs/user-guide.md`): presets explained, artifact report reading, model list + licenses screen content.
- License audit doc: final per-model license table with verification dates (design §8 items marked ⚠️ resolved).
- Beta handoff to you; feedback loop for v1.x fixes.

**Acceptance:** beta build installed on your machine; one full real-world video enhanced end-to-end without intervention.

---

## Testing strategy
| Layer | Approach |
|---|---|
| Unit | Every detector/repair core function with synthetic tensors (no GPU needed for logic tests); registry/downloader with mocked HTTP; TRT cache keying logic |
| Integration | Standard clips: Video2X 240p anime test clip + a live-action clip; pipeline round-trips per milestone acceptance criteria; depth output validated against TAS reference (visual + MAE spot checks) |
| QA-stage specific | Artifact-injection harness (M4) — the key quality gate for the differentiator |
| Performance | Benchmark table per model/clip/resolution/backend (TRT vs. CUDA); VRAM high-water marks logged; regression alerts if a stage slows >20% between milestones |
| Packaging | Clean-VM smoke test at M7 |

## Risks & mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| ONNX export gaps for some models (esp. experimental: SAFA, SPAN) | Can't use TRT path for them | PyTorch-CUDA fallback behind the same interface; TRT matrix shown honestly in UI model list |
| Blackwell + TensorRT version churn | Build breaks on driver/TRT updates | Pin exact versions in installer; smoke test at M0 and M6; engine cache invalidated cleanly on version change |
| Anime SR models' per-frame flicker on live-action content | Quality complaints | Live Action preset defaults to Real-ESRGAN/SAFA + flicker repair pass; presets tuned per content type |
| RIFE ghosting on extreme fast motion is inherent to the model | User-visible artifacts | That's exactly what M4's ghost detector + neighbor-blend repair targets; documented in report |
| GPL (Spandrel) vs. MIT app code | License contamination | Subprocess isolation for custom-model import (design §8); shipped models use our own loaders/adapters |
| Scope creep toward TAS feature parity (depth maps, AE plugin…) | Schedule slip | v1 scope frozen at design §1; those features explicitly deferred to v2+ |

## Progress log
- **M0+M1 ✅** — scaffold, FFmpeg I/O, scene detection, checkpointing, TRT backend (TRT 11 API), model registry + downloader.
- **M2 ✅ (2026-09-17)** — RIFE 4.25 / heavy / lite variants on TensorRT fp16 with PyTorch-CUDA fallback; scene-cut protection verified pixel-exact at both cut boundaries for 24→60 and 30→120 runs (474 frames @120fps, zero morphing); de-duplication option; benchmark table in [benchmarks.md](benchmarks.md) (TRT ≈ PyTorch — RIFE is memory-bandwidth bound on the 5090). Experimental FI models registered with metadata + license notes: M2M, GMFSS, FILM, AMT-SICDIA (adapters pending).
- **M3 ✅ core (2026-09-17)** — Upscale stage shipped: ShuffleCUGAN x2 (anime default, UpCunet2x_fast vendored arch) + Real-ESRGAN x4plus (live-action, RRDBNet vendored from BasicSR); both on TensorRT fp16 with PyTorch-CUDA fallback; full-frame inference by default at 32 GB with automatic tiled mode (overlap + feathered blending, partition-of-unity verified) as VRAM safety valve. E2E demo: 1080p→4K HEVC main10 pipeline run clean; tile-boundary visual check seam-free. Robustness fixes found during GPU validation: edge-floor wiring for tiled blend (zero-weight frame edges), pad-to-engine overshoot, mask upscaling to output res, TRT build headroom guard + nvidia-smi-aware free-VRAM estimate (cudaMemGetInfo is blind to VMM-mapped usage by other processes). Remaining M3 scope (deferred as M3.x): Fallin Soft/Strong, SPAN, Cyte V1, OpenProteus adapters — each is a vendored arch module + manifest entry on the same runner; benchmark table in `docs/benchmarks.md`.
- **Next: M4 artifact QA stage ⭐** — detectors (flicker, ghosting, banding, halo/ringing, line-art drift, color drift, regression frames) + hybrid CV/AI repairs + report export.
