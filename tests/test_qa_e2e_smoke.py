"""M4 E2E smoke (plan.md M4): run_pipeline on a real clip with QA enabled.

Lightweight by design — FI and SR are disabled so the stage chain is just the CPU
QAStage; ffmpeg does decode/encode. Validates the full wiring: PipelineConfig ->
QAStage, report accumulation into ``result["report"]``, the JobFinished event
payload, and JSON+PNG report export to the workspace dir.

Uses ``static_clip`` (flat gray): a single quantized level is maximal banding, so
this exercises detect -> repair (deband-dither) -> verify on real frames end-to-end.
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from aive.config import workspaces_dir
from aive.core.events import JobFinished
from aive.core.orchestrator import PipelineConfig, run_pipeline


def test_run_pipeline_qa_smoke(static_clip: str, tmp_path: Path):
    job_id = f"m4-smoke-{uuid.uuid4().hex[:8]}"
    out = tmp_path / "out.mp4"
    ws = workspaces_dir() / job_id

    events: list = []
    cfg = PipelineConfig(
        scene_split=False,  # deterministic single segment
        fi_enabled=False,   # no GPU needed for this smoke
        sr_enabled=False,
        qa_enabled=True,
        qa_mode="auto",
    )
    try:
        result = run_pipeline(job_id, static_clip, out, cfg, emit=events.append)

        # Output produced.
        assert out.exists() and out.stat().st_size > 0

        # Report wired into the result dict...
        report = result["report"]
        assert isinstance(report, dict), "QA stage should populate result['report']"
        assert report["num_frames"] == 30
        assert isinstance(report["findings"], list)

        # ...and into the JobFinished event.
        finished = [e for e in events if isinstance(e, JobFinished)]
        assert finished and finished[-1].ok is True
        assert finished[-1].report == report

        # Flat gray is maximal banding: detect -> repair -> verify must have run.
        banding = [f for f in report["findings"] if f["artifact"] == "banding"]
        assert banding, "flat-gray clip should produce banding findings"
        repaired = [f for f in banding if f["repaired"]]
        assert repaired, "auto mode should repair the flagged banding"
        deltas = [
            (f["before_metric"], f["after_metric"])
            for f in repaired
            if f["before_metric"] is not None and f["after_metric"] is not None
        ]
        assert deltas, "verify pass should record before/after metrics"
        assert all(after < before for before, after in deltas), (
            f"repair did not improve the metric: {deltas[:3]}"
        )

        # JSON + PNG report files exported to the workspace dir.
        jpath = ws / "artifact-report.json"
        ppath = ws / "artifact-report.png"
        assert jpath.exists(), f"missing {jpath}"
        assert ppath.exists(), f"missing {ppath}"
        data = json.loads(jpath.read_text(encoding="utf-8"))
        assert data["num_frames"] == 30
    finally:
        shutil.rmtree(ws, ignore_errors=True)
