# AI Video Enhancer (working title)

A desktop app that upscales video with state-of-the-art AI models, adds frame interpolation, and then **automatically detects artifacts introduced by the upscale/interpolation and repairs them** — with a friendly UI and a transparent artifact report.

## Status: DESIGN PHASE — awaiting review & approval

| Document | Purpose |
|---|---|
| [docs/design.md](<docs/design.md>) | Product design, architecture, model selection, artifact system, UI/UX |
| [docs/plan.md](<docs/plan.md>) | Phased implementation plan with milestones and acceptance criteria |

## One-paragraph pitch

Drop in a video → pick a preset (Anime / Live Action / Fast) → the app runs: **AI upscale → frame interpolation → artifact detection & repair** using current open models (RIFE 4.25/4.26, Real-CUGAN, Real-ESRGAN, SAFA, Anime4K v4), then hands back a clean file plus a report showing what artifacts were found and fixed.

## Reference tools studied

| Tool | What we take from it |
|---|---|
| Topaz Video AI (assumed "TAS" — please confirm) | Product UX: presets, denoise/upscale/interpolate stages, artifact-removal mindset |
| [Video2X](https://github.com/k4yt3x/video2x) v6.x | Model lineup (Anime4K v4, Real-CUGAN, Real-ESRGAN, RIFE), ncnn/Vulkan speed path, standard test clip for benchmarks |
| [Flowframes](https://github.com/n00mkrad/flowframes) | Interpolation UX: scene-cut protection, frame de-duplication (anime), UHD mode, audio/metadata preservation |
| [VapourSynth ecosystem](https://www.vapoursynth.com/) | Feature ideas: scriptable chains, Deband-style filters, per-scene processing; possible future "export as .py script" for power users |

## Key differentiator

No existing tool automatically measures what went wrong after AI upscale + interpolation (flicker, ghosting, banding, halos, color drift) and applies **targeted** fixes. That QA-and-repair stage is the core of this project.
