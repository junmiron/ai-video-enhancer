"""Model registry: manifest loading, filtering, experimental flags."""
from __future__ import annotations

import pytest

from aive.models.registry import get_model, list_models, load_manifest


def test_manifest_loads_all_expected():
    models = load_manifest()
    for name in ("realesrgan_x4plus", "rife_425", "rife_425_heavy", "rife_425_lite"):
        assert name in models, f"missing {name}"


def test_rife_variants_have_pinned_hashes():
    models = load_manifest()
    for name in ("rife_425", "rife_425_heavy", "rife_425_lite"):
        assert len(models[name].sha256) == 64, f"{name} sha256 not pinned"


def test_experimental_models_flagged():
    models = load_manifest()
    for name in ("m2m", "gmfss", "amt_sicdia"):
        assert models[name].experimental is True, f"{name} should be experimental"
        assert models[name].task == "fi"
    # Stable pipeline models are not flagged.
    assert models["rife_425"].experimental is False


def test_list_models_filter_and_order():
    fi = list_models(task="fi")
    names = [m.name for m in fi]
    assert names == sorted(names)
    assert "rife_425" in names and "realesrgan_x4plus" not in names
    # Unfiltered listing covers every task, still deterministic.
    all_names = [m.name for m in list_models()]
    assert len(all_names) == len(set(all_names))


def test_get_model_task_mismatch():
    with pytest.raises(ValueError, match="has task"):
        get_model("rife_425", task="sr")
