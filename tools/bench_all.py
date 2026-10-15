"""One-command benchmark suite (M8a).

Runs the per-model benchmark scripts and regenerates the "Latest snapshot"
section of ``docs/benchmarks.md`` from the JSON archive, so a single command
keeps the doc's current-state table in sync with the newest measurements:

    python tools/bench_all.py                          # FI + SR, 720p+1080p, N=30
    python tools/bench_all.py --families fi            # frame interpolation only
    python tools/bench_all.py --res 640x360 1280x720   # custom resolutions
    python tools/bench_all.py --no-snapshot            # measure only; don't touch the doc

Each model script runs in its own subprocess, so one OOM or crash (e.g. an
engine build with little free VRAM) is reported and skipped without killing
the rest of the run. The snapshot is regenerated from whatever succeeded —
stale rows keep their last good numbers until re-measured.

Exit codes: 0 = all scripts passed, 2 = at least one script failed (snapshot
still updated).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# family -> scripts in run order. Each script accepts --res and --calls.
FI_SCRIPTS = [
    "tools/benchmark_rife.py",
    "tools/benchmark_gmfss.py",
    "tools/benchmark_m2m.py",
]
SR_SCRIPTS = ["tools/benchmark_sr.py"]


def run_script(script: str, res: list[str], calls: int) -> tuple[int, str]:
    """Run one benchmark script; return (exit_code, tail of its output)."""
    cmd = [sys.executable, str(REPO_ROOT / script), "--res", *res, "--calls", str(calls)]
    print(f"\n=== {script} {' '.join(cmd[2:])} ===", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=["fi", "sr"], default=["fi", "sr"])
    parser.add_argument("--res", nargs="+", default=["1280x720", "1920x1080"], help="WxH")
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--no-snapshot", action="store_true", help="do not update docs/benchmarks.md")
    args = parser.parse_args()

    scripts: list[str] = []
    if "fi" in args.families:
        scripts += FI_SCRIPTS
    if "sr" in args.families:
        scripts += SR_SCRIPTS

    failures: list[tuple[str, int]] = []
    for script in scripts:
        code, _ = run_script(script, args.res, args.calls)
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"--- {script}: {status}", flush=True)
        if code != 0:
            failures.append((script, code))

    if not args.no_snapshot:
        from bench_common import BENCHMARKS_DOC, load_history, render_snapshot_md, update_benchmarks_doc

        changed = update_benchmarks_doc(render_snapshot_md(load_history()))
        print(f"\nSnapshot in {BENCHMARKS_DOC.relative_to(REPO_ROOT)}: {'updated' if changed else 'unchanged'}")
    else:
        print("\n(--no-snapshot: docs/benchmarks.md left untouched)")

    if failures:
        print(f"\n{len(failures)}/{len(scripts)} scripts failed:")
        for script, code in failures:
            print(f"  - {script} (exit {code})")
        return 2
    print("\nAll benchmark scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
