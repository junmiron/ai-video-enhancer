"""Shared helpers for benchmark scripts (M8a).

Used by ``bench_all.py`` and available to future per-model scripts:

- :func:`make_frames` / :func:`time_calls` / :func:`stats_from_ms` — the standard
  timing methodology (random uint8 frames, warmup + N timed calls including
  H2D/D2H transfers and a GPU sync after each call).
- :func:`load_history` / :func:`append_result` — the JSON archive at
  ``tools/bench_results.json``.
- :func:`render_snapshot_md` / :func:`update_benchmarks_doc` — regenerate the
  "Latest snapshot" section of ``docs/benchmarks.md`` from that archive, so a
  single command (``python tools/bench_all.py``) keeps the doc's current-state
  table in sync with the newest measurements.

The snapshot is deterministic: for each (model, resolution, backend) it shows
the row from the *newest* run in the archive that measured it.
"""
from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "tools" / "bench_results.json"
BENCHMARKS_DOC = REPO_ROOT / "docs" / "benchmarks.md"

SNAPSHOT_BEGIN = "<!-- BENCH_SNAPSHOT_BEGIN -->"
SNAPSHOT_END = "<!-- BENCH_SNAPSHOT_END -->"


# -- frames & timing ----------------------------------------------------------


def make_frames(w: int, h: int, n: int = 4, seed: int = 42) -> list:
    """A few random uint8 HWC frames (content is irrelevant for timing)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(h, w, 3), dtype="uint8") for _ in range(n)]


def stats_from_ms(times_ms: list[float]) -> dict:
    s = sorted(times_ms)
    return {
        "mean_ms": round(statistics.mean(s), 2),
        "median_ms": round(statistics.median(s), 2),
        "p95_ms": round(s[int(0.95 * (len(s) - 1))], 2),
    }


def time_calls(fn, calls: int, warmup: int = 3) -> dict:
    """Time `calls` invocations of zero-arg ``fn`` after `warmup` un-timed ones.

    Each timed call is followed by a CUDA sync so H2D/D2H transfers and the GPU
    work are fully accounted for (same methodology as the original scripts).
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(calls):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return stats_from_ms(times_ms)


def env_info() -> dict:
    import tensorrt

    import torch

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "tensorrt": tensorrt.__version__,
    }


# -- JSON archive -------------------------------------------------------------


def load_history(path: str | Path = RESULTS_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def append_result(entry: dict, path: str | Path = RESULTS_PATH) -> None:
    history = load_history(path)
    history.append(entry)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, indent=1), encoding="utf-8")


def new_entry(results: list[dict], calls: int | None = None) -> dict:
    """Archive entry skeleton with timestamp + environment (see env_info)."""
    e = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **env_info()}
    if calls is not None:
        e["calls"] = calls
    e["results"] = results
    return e


# -- snapshot rendering -------------------------------------------------------


def _model_key(row: dict) -> str | None:
    """FI rows use 'variant' (RIFE family), the others use 'model'."""
    return row.get("model") or row.get("variant")


def latest_rows(history: list[dict]) -> dict[tuple, tuple[str, dict]]:
    """Latest measurement per (model, resolution, backend); newest run wins.

    The archive is chronological, so a simple overwrite walk suffices.
    """
    best: dict[tuple, tuple[str, dict]] = {}
    for entry in history:
        ts = entry.get("timestamp", "")
        for row in entry.get("results", []):
            key = (_model_key(row), row.get("resolution"), row.get("backend"))
            if any(k is None for k in key):
                continue
            best[key] = (ts, row)
    return best


def render_snapshot_md(history: list[dict]) -> str:
    """Markdown block (markers included) for the 'Latest snapshot' section."""
    rows = latest_rows(history)
    if not rows:
        body = "_No benchmark runs recorded yet — run `python tools/bench_all.py`._"
    else:
        lines = [
            "| Model | Resolution (padded) | Backend | Precision | Mean ms/frame | Median | p95 | Frames/s | Measured (UTC) |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
        for key in sorted(rows):
            ts, r = rows[key]
            mean = r.get("mean_ms")
            fps = round(1000.0 / mean, 2) if mean else None
            lines.append(
                f"| {key[0]} | {r['resolution']} ({r.get('padded', '?')}) | {r['backend']} "
                f"| {r.get('precision', '?')} | **{mean}** ms | {r.get('median_ms', '?')} ms "
                f"| {r.get('p95_ms', '?')} ms | {fps if fps is not None else '?'} | {ts[:10]} |"
            )
        body = "\n".join(lines)

    return (
        f"{SNAPSHOT_BEGIN}\n\n"
        "_Auto-generated by `tools/bench_all.py` from "
        "[`tools/bench_results.json`](../tools/bench_results.json) — do not edit between the markers. "
        "One row per (model, resolution, backend): the newest measurement in the archive. "
        "FI rows are interpolated frames; SR rows are source frames (output at native model scale)._\n\n"
        f"{body}\n\n{SNAPSHOT_END}"
    )


def update_benchmarks_doc(snapshot_md: str, doc_path: str | Path = BENCHMARKS_DOC) -> bool:
    """Insert/replace the snapshot section in docs/benchmarks.md.

    Returns True if the file changed. First insertion lands right before the
    first per-model section ('## RIFE …'); later runs replace in place.
    """
    p = Path(doc_path)
    text = p.read_text(encoding="utf-8")
    if SNAPSHOT_BEGIN in text and SNAPSHOT_END in text:
        pre, rest = text.split(SNAPSHOT_BEGIN, 1)
        _, post = rest.split(SNAPSHOT_END, 1)
        new = f"{pre}{snapshot_md}{post}"
    else:
        anchor = "\n## RIFE"
        if anchor in text:
            pre, post = text.split(anchor, 1)
            new = (
                f"{pre}\n\n## Latest snapshot (auto-generated)\n\n{snapshot_md}{anchor}{post}"
            )
        else:
            new = text.rstrip("\n") + f"\n\n## Latest snapshot (auto-generated)\n\n{snapshot_md}\n"
    if new != text:
        p.write_text(new, encoding="utf-8")
        return True
    return False
