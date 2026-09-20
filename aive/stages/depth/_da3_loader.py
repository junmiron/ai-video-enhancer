"""Checkpoint loading + strict weight verification for DA3-family depth models.

Covers both checkpoint families that share the vendored ``_da3`` arch:

* **Official Depth-Anything-3** checkpoints (``depth_anything_v3_vits/vitb.safetensors``,
  Apache-2.0) — keys need the general-format conversion from
  :mod:`aive.stages.depth._da3.utils.model_loading`.
* **Limbo v1/v2** anime finetunes of DA3-small (CC BY-NC 4.0; personal use per
  project decision D3) — runtime-format state dicts that carry extra
  ``model.cam_enc.*`` / ``model.cam_dec.*`` camera-module weights and may omit
  training-only ``*_aux*`` heads.

Both are verified with a strict load BEFORE any wiring: missing or unexpected
keys outside the documented allowances raise, so random/partially-loaded
weights can never silently corrupt depth maps (same contract as M4.x restore).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_checkpoint_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load a .safetensors or .pth checkpoint payload onto the CPU."""
    import torch

    path = Path(checkpoint_path)
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file as _load_st
        except ImportError as exc:  # pragma: no cover - env without safetensors
            raise ImportError("The safetensors package is required to load .safetensors checkpoints") from exc
        return _load_st(str(path), device="cpu")
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(payload)!r}")
    return payload


def unwrap_state_dict(payload: Any) -> dict[str, "object"]:
    """Extract the plain state dict from a checkpoint payload (handles nested bundles)."""
    import torch

    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        if payload and all(isinstance(k, str) for k in payload) and all(
            isinstance(v, torch.Tensor) for v in payload.values()
        ):
            return payload
    raise TypeError("Checkpoint does not contain a state dict")


def normalize_state_dict(state_dict: dict[str, Any], model_name: str) -> dict[str, Any]:
    """Map a checkpoint's key layout onto the vendored runtime arch.

    Runtime keys are prefixed ``model.`` (DepthAnything3Net holds backbone+head
    under ``self.model`` in TAS's wrapper; our runner builds the same nesting).
    """
    from ._da3.utils.model_loading import convert_general_state_dict, convert_metric_state_dict

    keys = tuple(state_dict.keys())
    if not keys:
        raise ValueError("Checkpoint state dict is empty")
    if any(k.startswith("model.") for k in keys):
        return state_dict
    if any(k.startswith(("backbone.", "head.")) for k in keys):
        return {f"model.{k}": v for k, v in state_dict.items()}
    name = model_name.strip().lower()
    if "large" in name or "vitl" in name:
        return convert_metric_state_dict(state_dict)
    return convert_general_state_dict(state_dict)


def build_da3_model(model_name: str):
    """Construct the vendored DA3 runtime for a preset (weights NOT loaded).

    ``model_name`` is one of ``da3-small`` / ``da3-base`` (the presets Limbo and
    the official monocular checkpoints use). Returns ``(wrapper, net)`` where
    ``wrapper.model`` is the DepthAnything3Net — mirroring TAS's
    MonocularDepthAnything3 nesting so checkpoint keys line up.
    """
    import torch.nn as nn

    from ._da3.model.da3 import DepthAnything3Net
    from ._da3.model.dinov2.dinov2 import DinoV2
    from ._da3.model.dualdpt import DualDPT

    presets = {
        "da3-small": {
            "backbone": {"name": "vits", "out_layers": [5, 7, 9, 11], "alt_start": 4,
                         "qknorm_start": 4, "rope_start": 4, "cat_token": True},
            "head": {"dim_in": 768, "output_dim": 2, "features": 64,
                     "out_channels": [48, 96, 192, 384]},
        },
        "da3-base": {
            "backbone": {"name": "vitb", "out_layers": [5, 7, 9, 11], "alt_start": 4,
                         "qknorm_start": 4, "rope_start": 4, "cat_token": True},
            "head": {"dim_in": 1536, "output_dim": 2, "features": 128,
                     "out_channels": [96, 192, 384, 768]},
        },
    }
    key = model_name.strip().lower()
    if key not in presets:
        raise ValueError(f"Unsupported DA3 preset '{model_name}'. Known: {list(presets)}")
    p = presets[key]

    class _Wrapper(nn.Module):
        """Minimal stand-in for TAS's MonocularDepthAnything3 (holds .model)."""

        def __init__(self) -> None:
            super().__init__()
            bb_cfg = p["backbone"]
            backbone = DinoV2(
                name=bb_cfg["name"], out_layers=bb_cfg["out_layers"], alt_start=bb_cfg["alt_start"],
                qknorm_start=bb_cfg["qknorm_start"], rope_start=bb_cfg["rope_start"],
                cat_token=bb_cfg["cat_token"],
            )
            head = DualDPT(
                dim_in=p["head"]["dim_in"], output_dim=p["head"]["output_dim"],
                features=p["head"]["features"], out_channels=p["head"]["out_channels"],
            )
            self.model = DepthAnything3Net(net=backbone, head=head)

    wrapper = _Wrapper()
    return wrapper


def verify_da3_weights(model_name: str, weights_path: str | Path) -> dict[str, list[str]]:
    """Strict-load a DA3-family checkpoint; raise on any disallowed mismatch.

    Allowances (both verified against the real checkpoints):
      * missing keys containing ``_aux`` — training-only auxiliary heads;
      * unexpected keys under ``model.cam_enc.`` / ``model.cam_dec.`` — camera
        encoder/decoder modules present in Limbo's runtime export but unused by
        monocular depth inference (TAS loads with the same allowances).

    Returns the (empty-or-not) missing/unexpected lists for logging.
    """
    wrapper = build_da3_model(model_name)
    state = normalize_state_dict(unwrap_state_dict(load_checkpoint_payload(weights_path)), model_name)
    result = wrapper.load_state_dict(state, strict=False)
    del state
    missing = [k for k in result.missing_keys if "_aux" not in k]
    unexpected = [
        k for k in result.unexpected_keys
        if not k.startswith(("model.cam_enc.", "model.cam_dec."))
    ]
    if missing or unexpected:
        sample = ", ".join((missing + unexpected)[:8])
        raise RuntimeError(
            f"Depth weights do not match the {model_name} arch ({weights_path}): "
            f"{len(missing)} disallowed-missing, {len(unexpected)} disallowed-unexpected; e.g. {sample}"
        )
    log.info(
        "Verified %s checkpoint %s (missing-aux=%d, cam-extra=%d)",
        model_name, Path(weights_path).name, len(result.missing_keys), len(result.unexpected_keys),
    )
    return {"missing": missing, "unexpected": unexpected}


def load_da3(model_name: str, weights_path: str | Path, device: str = "cuda", dtype=None):
    """Load a verified DA3-family checkpoint into the arch (eval mode)."""
    wrapper = build_da3_model(model_name)
    state = normalize_state_dict(unwrap_state_dict(load_checkpoint_payload(weights_path)), model_name)
    result = wrapper.load_state_dict(state, strict=False)
    del state
    missing = [k for k in result.missing_keys if "_aux" not in k]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(("model.cam_enc.", "model.cam_dec."))]
    if missing or unexpected:
        raise RuntimeError(f"{weights_path}: disallowed weight mismatch (missing={missing[:4]}, unexpected={unexpected[:4]})")
    wrapper = wrapper.to(device)
    if dtype is not None:
        wrapper = wrapper.to(dtype=dtype)
    return wrapper.eval()


__all__ = [
    "build_da3_model",
    "load_checkpoint_payload",
    "load_da3",
    "normalize_state_dict",
    "unwrap_state_dict",
    "verify_da3_weights",
]
