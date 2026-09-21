"""GMFSS frame-interpolation benchmark: PyTorch fp16/fp32 per resolution.

Usage (from the repo root):
    python tools/benchmark_gmfss.py [--res 1280x720 1920x1080] [--calls 30]

GMFSS is CUDA-only (the softsplat splatting is an index_add_ scatter with
data-dependent boundary masks that ONNX cannot represent), so there is no
TensorRT column — this benchmarks the PyTorch backend only. Above-1080p sources
run flow at half resolution (scale=0.5) and, at/above UHD, in fp32; both are
handled by GMFSSRunner automatically.

Prints a Markdown table and appends JSON results to tools/bench_results.json.
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

    # GMFSS has a per-pair reuse cache; cycle through distinct pairs so each timed
    # call exercises the full reuse() + inference() path (the realistic cost).
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
    parser.add_argument("--res", nargs="+", default=["1280x720", "1920x1080"], help="WxH")
    parser.add_argument("--calls", type=int, default=30)
    args = parser.parse_args()

    import torch

    from aive.models.downloader import download_model
    from aive.models.registry import get_model
    from aive.stages.interpolate.gmfss_model import GMFSSRunner

    print(f"# Benchmark run {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    weights = download_model(get_model("gmfss"))  # directory of 5 checkpoints
    results = []
    for res in args.res:
        w, h = (int(x) for x in res.split("x"))
        rng = __import__("numpy").random.default_rng(42)
        pairs = _frames(rng, w, h)

        t0 = time.time()
        runner = GMFSSRunner(weights, w, h)
        load_s = round(time.time() - t0, 1)
        stats = _bench(runner, pairs, args.calls)
        results.append(
            {
                "model": "gmfss",
                "resolution": f"{w}x{h}",
                "padded": f"{runner.pw}x{runner.ph}",
                "backend": "pytorch",
                "precision": "fp16" if not runner.uhd else "fp32",
                "flow_scale": runner.scale,
                **stats,
                "calls_per_s": round(1000.0 / stats["mean_ms"], 2),
                "model_load_s_this_run": load_s,
            }
        )
        print(
            f"[gmfss {res}] PyTorch {'fp32' if runner.uhd else 'fp16'} "
            f"(scale={runner.scale}): mean={stats['mean_ms']} ms/call",
            flush=True,
        )
        del runner
        torch.cuda.empty_cache()

    # -- Markdown table ---------------------------------------------------------
    lines = [
        "| Model | Resolution (padded) | Backend | Precision | Flow scale | Mean ms/call | Median | p95 | Calls/s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['resolution']} ({r['padded']}) | {r['backend']} "
            f"| {r['precision']} | {r['flow_scale']} | {r['mean_ms']} | {r['median_ms']} "
            f"| {r['p95_ms']} | {r['calls_per_s']} |"
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
            "calls": args.calls,
            "results": results,
        }
    )
    out.write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
