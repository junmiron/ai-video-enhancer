"""Internal test harness (NOT a product CLI - the GUI ships at M6).

Usage:
  python -m aive probe <file>
  python -m aive passthrough <in> <out>
  python -m aive pipeline <in> <out> [--no-scene-split] [--codec X] [--crf N]
                            [--container mp4|mkv] [--keep-workspace]
                            [--fi] [--target-fps F] [--fi-multiplier N] [--dedup-threshold T]
  python -m aive download-model <name>
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    from .config import load_settings, save_settings
    from .logutil import setup_logging

    parser = argparse.ArgumentParser(prog="aive")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="print media info as JSON")
    p_probe.add_argument("file")

    p_pt = sub.add_parser("passthrough", help="fast remux sanity check (-c copy)")
    p_pt.add_argument("input")
    p_pt.add_argument("output")

    p_pipe = sub.add_parser("pipeline", help="full pipeline (M2: frame interpolation via --fi; checkpointed segments)")
    p_pipe.add_argument("input")
    p_pipe.add_argument("output")
    p_pipe.add_argument("--no-scene-split", action="store_true")
    p_pipe.add_argument("--codec", default=None)
    p_pipe.add_argument("--crf", type=int, default=None)
    p_pipe.add_argument("--container", choices=["mp4", "mkv"], default=None)
    p_pipe.add_argument("--keep-workspace", action="store_true")
    p_pipe.add_argument("--fi", action="store_true", help="enable frame interpolation (RIFE 4.25)")
    p_pipe.add_argument("--target-fps", type=float, default=None, help="output fps; wins over --fi-multiplier")
    p_pipe.add_argument("--fi-multiplier", type=int, default=None, help="fps multiplier when --target-fps is unset (default x2)")
    p_pipe.add_argument("--dedup-threshold", type=float, default=None, help="de-duplication mean-abs-diff threshold in [0,1]; 0 disables")

    p_dl = sub.add_parser("download-model", help="download a model from the registry")
    p_dl.add_argument("name")

    args = parser.parse_args(argv)
    settings = load_settings()
    setup_logging(settings.log_level)

    if args.cmd == "probe":
        from .stages.ingest import probe

        print(json.dumps(probe(args.file).__dict__, indent=2, default=str))
        return 0

    if args.cmd == "passthrough":
        from .stages.encode import stream_copy_passthrough

        out = stream_copy_passthrough(args.input, args.output)
        print(f"OK -> {out}")
        return 0

    if args.cmd == "pipeline":
        from .core.jobqueue import ControlToken
        from .core.orchestrator import PipelineCancelled, PipelineConfig, run_pipeline

        cfg = PipelineConfig.from_settings(settings)
        if args.no_scene_split:
            cfg.scene_split = False
        if args.codec:
            cfg.output_codec = args.codec
        if args.crf is not None:
            cfg.crf = args.crf
        if args.container:
            cfg.container = args.container
        if args.keep_workspace:
            cfg.keep_workspace = True
        if args.fi or settings.fi_enabled:
            cfg.fi_enabled = True
        if args.target_fps is not None:
            cfg.fi_target_fps = args.target_fps
        if args.fi_multiplier is not None:
            cfg.fi_multiplier = args.fi_multiplier
        if args.dedup_threshold is not None:
            cfg.fi_dedup_threshold = args.dedup_threshold

        def _emit(ev):
            d = ev.to_dict()
            keep = {k: v for k, v in d.items() if v not in (None, "", 0, 0.0)}
            print(f"[{ev.type}] " + json.dumps(keep), flush=True)

        try:
            result = run_pipeline("cli-test", args.input, args.output, cfg, emit=_emit, control=ControlToken())
        except PipelineCancelled:
            print("CANCELLED")
            return 130
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "download-model":
        from .models.downloader import download_model
        from .models.registry import get_model

        info = get_model(args.name)
        path = download_model(info, progress_cb=lambda done, total: print(
            f"  {done >> 20} MB / {(total >> 20) if total > 0 else '?'} MB", flush=True))
        print(f"OK -> {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
