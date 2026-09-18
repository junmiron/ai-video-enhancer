"""Artifact QA / restore stage (M4) — detect → repair → verify.

The core differentiator: after upscale + interpolation, automatically detect the
artifacts those AI stages introduced and apply targeted fixes, then produce a
human-readable report (design.md §4). Heavy imports (cv2/torch) are pulled lazily
by :mod:`aive.stages.qa.stage` so importing this package stays cheap.
"""
from .findings import ARTIFACT_META, ArtifactType, Finding, QAReport

__all__ = ["ARTIFACT_META", "ArtifactType", "Finding", "QAReport"]
