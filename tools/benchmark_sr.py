r"""SR model throughput benchmark: TensorRT fp16 vs PyTorch fp16 per model/resolution.

Usage (from the repo root):
    python tools/benchmark_sr.py [--models shufflecugan_x2 realesrgan_x4plus ...]
                                 [--res 640x360 1280x720] [--calls 30]

Prints a Markdown table and appends JSON results to tools/bench_results.json.
TensorRT engines are cached under %LOCALAPPDATA%\ai-video-enhancer\models\engines\sr\,
so re-runs skip the (slow) engine builds.

Notes:
- One call = one source frame upscaled at the model's native scale (x2 or x4).
  The "resolution" column is the INPUT size; an x4 model at 720p input produces
  a ~5K output and will run in tiled mode when VRAM requires it.
- SRRunner falls back to PyTorch-CUDA automatically when free VRAM is below the
  TRT build headroom (another process holding most of the GPU). The row records
  which backend actually ran: backend=pytorch + trt_fallback=true means the
  TensorRT column degraded, not that PyTorch was measured on purpose.
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import cycle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench_common import append_result, env_info, make_frames, new_entry, time_calls  # noqa: E402


def _sr_models() -> list[str]:
    from aive.models.registry import load_manifest

    return [m.name for m in load_manifest().values() if m.task == "sr"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, help="registry names (default: all SR models)")
    parser.add_argument("--res", nargs="+", default=["640x360", "1280x720"], help="WxH input size")
    parser.add_argument("--calls", type=int, default=30)
    args = parser.parse_args()

    import torch

    from aive.models.downloader import download_model
    from aive.models.registry import get_model
    from aive.stages.upscale.stage import SRRunner

    models = _sr_models() if args.models is None else list(args.models)
    for m in models:
        try:
            get_model(m)
        except KeyError as exc:
            raise SystemExit(f"Unknown SR model '{m}': {exc}") from exc

    print(
        f"# SR benchmark run {env_info()['torch']} on {env_info()['gpu']}",
        flush=True,
    )

    results = []
    for name in models:
        weights = download_model(get_model(name))
        for res in args.res:
            w, h = (int(x) for x in res.split("x"))
            frames = make_frames(w, h, n=4)

            # -- TensorRT fp16 (auto-fallback to PyTorch-CUDA on low VRAM) ----
            t0 = time.time()
            runner = SRRunner(weights, w, h, name, use_trt=True)
            build_s = round(time.time() - t0, 1)
            it = iter(cycle(frames))
            stats = time_calls(lambda: runner(next(it)), args.calls)
            results.append(
                {
                    "model": name,
                    "resolution": f"{w}x{h}",
                    "padded": f"{runner.pw}x{runner.ph}",
                    "backend": "tensorrt" if runner._trt_runner is not None else "pytorch",
                    "precision": "fp16",
                    "scale": runner.scale,
                    "tiled": runner.tile_cover,
                    "trt_fallback": runner._trt_runner is None,
                    **stats,
                    "frames_per_s": round(1000.0 / stats["mean_ms"], 2),
                    "engine_build_s_this_run": build_s,
                }
            )
            print(f"[{name} {res}] TRT path: mean={stats['mean_ms']} ms/frame", flush=True)
            del runner
            torch.cuda.empty_cache()

            # -- PyTorch fp16 --------------------------------------------------
            t0 = time.time()
            torch_runner = SRRunner(weights, w, h, name, use_trt=False)
            load_s = round(time.time() - t0, 1)
            it = iter(cycle(frames))
            stats = time_calls(lambda: torch_runner(next(it)), args.calls)
            results.append(
                {
                    "model": name,
                    "resolution": f"{w}x{h}",
                    "padded": f"{torch_runner.pw}x{torch_runner.ph}",
                    "backend": "pytorch",
                    "precision": "fp16",
                    "scale": torch_runner.scale,
                    "tiled": torch_runner.tile_cover,
                    **stats,
                    "frames_per_s": round(1000.0 / stats["mean_ms"], 2),
                    "model_load_s_this_run": load_s,
                }
            )
            print(f"[{name} {res}] PyTorch: mean={stats['mean_ms']} ms/frame", flush=True)
            del torch_runner
            torch.cuda.empty_cache()

    # -- Markdown table ---------------------------------------------------------
    lines = [
        "| Model | Input res (padded) | Backend | Precision | Scale | Tiled | Mean ms/frame | Median | p95 | Frames/s |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['resolution']} ({r['padded']}) | {r['backend']} "
            f"| {r['precision']} | x{r['scale']} | {'C=' + str(r['tiled']) if r.get('tiled') else 'no'} "
            f"| {r['mean_ms']} | {r['median_ms']} | {r['p95_ms']} | {r['frames_per_s']} |"
        )
    print("\n".join(lines))

    # -- JSON archive -----------------------------------------------------------
    append_result(new_entry(results, calls=args.calls))
    print(f"\nSaved -> {REPO_ROOT / 'tools' / 'bench_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
