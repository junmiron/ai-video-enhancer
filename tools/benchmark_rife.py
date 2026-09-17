"""RIFE 4.25 family benchmark: TensorRT fp16 vs PyTorch fp16 per variant/resolution.

Usage (from the repo root):
    python tools/benchmark_rife.py [--variants rife_425 rife_425_heavy rife_425_lite]
                                   [--res 1280x720 1920x1080] [--calls 30]

Prints a Markdown table and appends JSON results to tools/bench_results.json.
TensorRT engines are cached under %LOCALAPPDATA%\ai-video-enhancer\models\engines\rife\,
so re-runs skip the (slow) engine builds.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _frames(rng, w: int, h: int) -> list[tuple]:
    """A few random uint8 HWC frame pairs (content is irrelevant for timing)."""
    return [
        (
            rng.integers(0, 256, size=(h, w, 3), dtype="uint8"),
            rng.integers(0, 256, size=(h, w, 3), dtype="uint8"),
        )
        for _ in range(4)
    ]


def _bench(runner, pairs: list[tuple], calls: int, warmup: int = 3) -> dict:
    import torch

    for i in range(warmup):
        runner(*pairs[i % len(pairs)], 0.5)
    torch.cuda.synchronize()
    times_ms = []
    for i in range(calls):
        a, b = pairs[(i + warmup) % len(pairs)]
        t0 = time.perf_counter()
        runner(a, b, 0.5)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p95_ms": round(sorted(times_ms)[int(0.95 * (len(times_ms) - 1))], 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=["rife_425", "rife_425_heavy", "rife_425_lite"])
    parser.add_argument("--res", nargs="+", default=["1280x720", "1920x1080"], help="WxH")
    parser.add_argument("--calls", type=int, default=30)
    args = parser.parse_args()

    import torch

    from aive.models.downloader import download_model
    from aive.models.registry import get_model
    from aive.stages.interpolate.rife_model import RIFE_VARIANTS
    from aive.stages.interpolate.stage import RIFERunner

    for v in args.variants:
        if v not in RIFE_VARIANTS:
            raise SystemExit(f"Unknown variant '{v}'. Known: {list(RIFE_VARIANTS)}")

    print(f"# Benchmark run {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(
        f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"tensorrt={__import__('tensorrt').__version__}"
    )

    results = []
    for variant in args.variants:
        weights = download_model(get_model(variant))
        for res in args.res:
            w, h = (int(x) for x in res.split("x"))
            rng = __import__("numpy").random.default_rng(42)
            pairs = _frames(rng, w, h)

            # -- TensorRT fp16 -------------------------------------------------
            t0 = time.time()
            trt_runner = RIFERunner(weights, w, h, use_trt=True, variant=variant)
            build_s = round(time.time() - t0, 1)
            stats = _bench(trt_runner, pairs, args.calls)
            results.append(
                {
                    "variant": variant,
                    "resolution": f"{w}x{h}",
                    "padded": f"{trt_runner.pw}x{trt_runner.ph}",
                    "backend": "tensorrt",
                    "precision": trt_runner.precision,
                    **stats,
                    "calls_per_s": round(1000.0 / stats["mean_ms"], 2),
                    "engine_build_s_this_run": build_s,
                }
            )
            print(f"[{variant} {res}] TRT {trt_runner.precision}: mean={stats['mean_ms']} ms/call", flush=True)
            del trt_runner
            torch.cuda.empty_cache()

            # -- PyTorch fp16 --------------------------------------------------
            t0 = time.time()
            torch_runner = RIFERunner(weights, w, h, use_trt=False, variant=variant)
            load_s = round(time.time() - t0, 1)
            stats = _bench(torch_runner, pairs, args.calls)
            results.append(
                {
                    "variant": variant,
                    "resolution": f"{w}x{h}",
                    "padded": f"{torch_runner.pw}x{torch_runner.ph}",
                    "backend": "pytorch",
                    "precision": "fp16" if not torch_runner.uhd else "fp32",
                    **stats,
                    "calls_per_s": round(1000.0 / stats["mean_ms"], 2),
                    "model_load_s_this_run": load_s,
                }
            )
            print(f"[{variant} {res}] PyTorch: mean={stats['mean_ms']} ms/call", flush=True)
            del torch_runner
            torch.cuda.empty_cache()

    # -- Markdown table ---------------------------------------------------------
    lines = [
        "| Variant | Resolution (padded) | Backend | Precision | Mean ms/call | Median | p95 | Calls/s |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['resolution']} ({r['padded']}) | {r['backend']} "
            f"| {r['precision']} | {r['mean_ms']} | {r['median_ms']} | {r['p95_ms']} | {r['calls_per_s']} |"
        )
    print("\n".join(lines))

    # -- JSON archive -----------------------------------------------------------
    out = REPO_ROOT / "tools" / "bench_results.json"
    history: list = []
    if out.exists():
        try:
            history = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "tensorrt": __import__("tensorrt").__version__,
            "calls": args.calls,
            "results": results,
        }
    )
    out.write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
