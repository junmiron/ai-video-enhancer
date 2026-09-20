"""M7: "upscale first" stage-order mode (design.md §3.1).

Covers chain ordering in both orders, per-stage frame-count accounting for
progress bars, checkpoint-fingerprint invalidation on order change, and the
preset defaults (Live Action ships with sr_first per design §3.1).
"""
from __future__ import annotations

import pytest

from aive.config import DEFAULTS, Settings
from aive.core.orchestrator import PipelineConfig, build_stage_chain, stage_frame_counts


class _FakeStage:
    """Lightweight stand-in for stage_frame_counts (no torch/cv2 imports)."""

    def __init__(self, name: str):
        self.name = name


def _names(cfg: PipelineConfig) -> list[str]:
    return [st.name for st in build_stage_chain(cfg)]


# -- chain ordering ---------------------------------------------------------------


def test_chain_order_fi_first_default():
    cfg = PipelineConfig(fi_enabled=True, sr_enabled=True, qa_enabled=True)
    assert _names(cfg) == ["interpolate", "upscale", "qa"]


def test_chain_order_sr_first():
    cfg = PipelineConfig(fi_enabled=True, sr_enabled=True, qa_enabled=True, stage_order="sr_first")
    assert _names(cfg) == ["upscale", "interpolate", "qa"]


@pytest.mark.parametrize("order", ("fi_first", "sr_first"))
def test_qa_always_last(order):
    cfg = PipelineConfig(fi_enabled=True, sr_enabled=True, qa_enabled=True, stage_order=order)
    assert _names(cfg)[-1] == "qa"


def test_chain_order_with_single_stage():
    # Only one of FI/SR enabled: order is irrelevant, QA still last.
    fi_only = PipelineConfig(fi_enabled=True, sr_enabled=False, qa_enabled=True, stage_order="sr_first")
    assert _names(fi_only) == ["interpolate", "qa"]
    sr_only = PipelineConfig(fi_enabled=False, sr_enabled=True, qa_enabled=True, stage_order="fi_first")
    assert _names(sr_only) == ["upscale", "qa"]


def test_chain_identity_fallback():
    cfg = PipelineConfig(fi_enabled=False, sr_enabled=False, qa_enabled=False)
    assert _names(cfg) == ["identity"]


def test_invalid_stage_order_rejected():
    with pytest.raises(ValueError):
        build_stage_chain(PipelineConfig(qa_enabled=True, stage_order="sideways"))


# -- per-stage frame counts (progress accounting) ---------------------------------


def test_frame_counts_fi_first():
    stages = [_FakeStage("interpolate"), _FakeStage("upscale"), _FakeStage("qa")]
    # Gap slotting: x2 inserts one frame per gap -> n + (n-1) = 179 for 90 frames;
    # every stage after FI sees the final count.
    assert stage_frame_counts(stages, 90, (2, 1)) == [179, 179, 179]


def test_frame_counts_sr_first():
    stages = [_FakeStage("upscale"), _FakeStage("interpolate"), _FakeStage("qa")]
    # Upscale runs BEFORE interpolation: it sees source counts only.
    assert stage_frame_counts(stages, 90, (2, 1)) == [90, 179, 179]


def test_frame_counts_fi_disabled():
    stages = [_FakeStage("upscale"), _FakeStage("qa")]
    assert stage_frame_counts(stages, 45, None) == [45, 45]


# -- checkpoint fingerprint ---------------------------------------------------------


def test_stage_order_changes_content_fingerprint():
    base = PipelineConfig(fi_enabled=True, sr_enabled=True, qa_enabled=True)
    flipped = PipelineConfig(
        fi_enabled=True, sr_enabled=True, qa_enabled=True, stage_order="sr_first"
    )
    assert base.content_fingerprint() != flipped.content_fingerprint()


# -- settings plumbing + presets ----------------------------------------------------


def test_settings_default_and_roundtrip():
    s = Settings()
    assert s.stage_order == "fi_first"
    cfg = PipelineConfig.from_settings(s)
    assert cfg.stage_order == "fi_first"


def test_preset_stage_orders():
    from aive.ui.presets import PRESETS, apply_preset

    for key in ("anime", "mixed", "fast_preview"):
        out = apply_preset(dict(DEFAULTS), key)
        assert out["stage_order"] == "fi_first", key
    # Design §3.1: the Live Action preset exposes upscale-first as its quality default.
    out = apply_preset(dict(DEFAULTS), "live_action")
    assert out["stage_order"] == "sr_first"
    # Every preset declares an explicit order (no silent defaults).
    for key, p in PRESETS.items():
        assert p["fields"].get("stage_order") in ("fi_first", "sr_first"), key
