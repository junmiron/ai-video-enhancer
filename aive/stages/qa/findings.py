"""Artifact QA data model (M4): findings, per-artifact summary and report.

Design principle (design.md §4.1): each detector emits *findings* — a location
(segment + frame), an artifact type, a severity score and the region that was
flagged. Repair policies consume findings and apply targeted fixes; a verify
pass re-measures the same metric and records before/after deltas for the report.

Findings are deliberately compact: they store a bounding box + area fraction of
the flagged region (not full-resolution masks), so a report stays small and can
cross process boundaries to the M6 UI. Representative thumbnails for the PNG
report are kept in-memory only (:attr:`QAReport.sample_frames`) and excluded from
JSON export.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class ArtifactType(str, Enum):
    """The v1 detector set (design.md §4.2)."""

    FLICKER = "flicker"            # temporal detail instability in static regions
    GHOSTING = "ghosting"          # double-exposure from FI blending / occlusion
    BANDING = "banding"            # posterization in smooth gradients
    HALO = "halo"                  # edge overshoot / ringing / over-sharpening
    LINEART = "lineart"            # line-art degradation (anime)
    COLOR_DRIFT = "color_drift"    # per-scene color/exposure drift
    REGRESSION = "regression"      # global quality-regression frame

    @classmethod
    def all(cls) -> list["ArtifactType"]:
        return list(cls)


# Human labels + report colors (design.md §7.3 marker palette).
ARTIFACT_META: dict[ArtifactType, tuple[str, str]] = {
    ArtifactType.FLICKER: ("Flicker", "#f5a623"),
    ArtifactType.GHOSTING: ("Ghosting", "#9b5de5"),
    ArtifactType.BANDING: ("Banding", "#4cc9f0"),
    ArtifactType.HALO: ("Halo / ringing", "#ffd60a"),
    ArtifactType.LINEART: ("Line-art", "#212121"),
    ArtifactType.COLOR_DRIFT: ("Color drift", "#e5e5e5"),
    ArtifactType.REGRESSION: ("Regression frame", "#ef476f"),
}


def label_of(artifact: ArtifactType) -> str:
    return ARTIFACT_META[artifact][0]


@dataclass
class Finding:
    """One detected (and optionally repaired) artifact instance."""

    artifact: ArtifactType
    frame_index: int  # index within the segment's processed stream
    severity: float  # 0..1 detector confidence/severity at detection time
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h at FULL resolution
    region_area_frac: float = 0.0  # fraction of the frame inside the flagged region
    repaired: bool = False
    method: str = ""  # repair applied, or "detect-only" / "none"
    before_metric: float | None = None  # detector metric pre-repair (verify baseline)
    after_metric: float | None = None  # detector metric post-repair
    rejected_methods: list[str] = field(default_factory=list)  # repairs attempted but rejected by the verify gate

    @property
    def improved(self) -> bool:
        if self.before_metric is None or self.after_metric is None:
            return False
        return self.after_metric < self.before_metric


@dataclass
class QAReport:
    """Aggregated artifact-QA report for one job (design.md §7.3)."""

    input_path: str = ""
    output_path: str = ""
    fps: float = 0.0
    num_frames: int = 0
    duration_s: float = 0.0
    mode: str = "auto"
    findings: list[Finding] = field(default_factory=list)
    # In-memory only (excluded from JSON): representative flagged frames for the
    # PNG report — each {"frame_index", "artifact", "thumb"(HWC uint8), "bbox"}.
    sample_frames: list[dict] = field(default_factory=list, repr=False)

    def by_artifact(self) -> dict[ArtifactType, list[Finding]]:
        out: dict[ArtifactType, list[Finding]] = {a: [] for a in ArtifactType.all()}
        for f in self.findings:
            out[f.artifact].append(f)
        return out

    def summary(self) -> dict[str, dict]:
        """Per-artifact rollup used by the report table and UI."""
        rows: dict[str, dict] = {}
        for art in ArtifactType.all():
            fs = self.by_artifact()[art]
            if not fs:
                continue
            sevs = [f.severity for f in fs]
            repaired = [f for f in fs if f.repaired]
            before = [f.before_metric for f in repaired if f.before_metric is not None]
            after = [f.after_metric for f in repaired if f.after_metric is not None]
            rows[art.value] = {
                "label": label_of(art),
                "count": len(fs),
                "max_severity": round(max(sevs), 3),
                "avg_severity": round(sum(sevs) / len(sevs), 3),
                "repaired_count": len(repaired),
                "before_metric": round(sum(before) / len(before), 4) if before else None,
                "after_metric": round(sum(after) / len(after), 4) if after else None,
            }
        return rows

    def frame_timestamp(self, frame_index: int) -> str:
        """Frame index (within a segment's processed stream) -> HH:MM:SS.mmm."""
        if self.fps <= 0:
            return f"frame {frame_index}"
        t = frame_index / self.fps
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def to_dict(self) -> dict:
        d = {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_s": round(self.duration_s, 3),
            "mode": self.mode,
            "summary": self.summary(),
            "findings": [
                {
                    "artifact": f.artifact.value,
                    "frame_index": f.frame_index,
                    "timestamp": self.frame_timestamp(f.frame_index),
                    "severity": round(f.severity, 4),
                    "bbox": list(f.bbox),
                    "region_area_frac": round(f.region_area_frac, 4),
                    "repaired": f.repaired,
                    "method": f.method,
                    "before_metric": None if f.before_metric is None else round(f.before_metric, 4),
                    "after_metric": None if f.after_metric is None else round(f.after_metric, 4),
                    "rejected_methods": list(f.rejected_methods),
                }
                for f in self.findings
            ],
        }
        return d

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def report_from_dict(d: dict) -> QAReport:
    """Inverse of :meth:`QAReport.to_dict` (for the M6 UI / cross-process)."""
    rep = QAReport(
        input_path=d.get("input_path", ""),
        output_path=d.get("output_path", ""),
        fps=float(d.get("fps", 0.0)),
        num_frames=int(d.get("num_frames", 0)),
        duration_s=float(d.get("duration_s", 0.0)),
        mode=d.get("mode", "auto"),
    )
    for f in d.get("findings", []):
        rep.findings.append(
            Finding(
                artifact=ArtifactType(f["artifact"]),
                frame_index=int(f["frame_index"]),
                severity=float(f["severity"]),
                bbox=tuple(f.get("bbox", (0, 0, 0, 0))),
                region_area_frac=float(f.get("region_area_frac", 0.0)),
                repaired=bool(f.get("repaired", False)),
                method=f.get("method", ""),
                before_metric=f.get("before_metric"),
                after_metric=f.get("after_metric"),
                rejected_methods=list(f.get("rejected_methods", [])),
            )
        )
    return rep
