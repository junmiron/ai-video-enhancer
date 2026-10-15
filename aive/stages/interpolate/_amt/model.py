"""AMT model construction + checkpoint loading (thin wrapper over vendored arch).

Upstream checkpoints are ``{'epoch': ..., 'state_dict': {...}}``; we accept that
wrapper or a bare state dict. AMT-S and AMT-L differ only in architecture
defaults (channels / num_flows), which are the classes' own defaults, so building
is just importing the right module's ``Model``.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

AMT_VARIANTS = ("s", "l")


def build_amt(variant: str):
    """Construct a fresh AMT model for ``variant`` ('s' or 'l'); weights NOT loaded."""
    variant = (variant or "s").lower()
    if variant == "s":
        from .networks.amt_s import Model
    elif variant == "l":
        from .networks.amt_l import Model
    else:
        raise ValueError(f"Unknown AMT variant '{variant}' (expected one of {AMT_VARIANTS}).")
    return Model()


def load_amt_weights(model, weights_path) -> None:
    """Load an AMT checkpoint into ``model`` (strict).

    ``weights_only=False`` is required: the official releases carry non-tensor
    metadata (an ``'epoch'`` field) that torch's weights-only unpickler rejects
    (verified against amt-s.pth / amt-l.pth). These are trusted personal-use
    files, so full-unpickle loading is acceptable here. A strict load guarantees
    the model never runs with random weights.
    """
    import torch

    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state):  # DataParallel prefix, defensive
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    log.info("AMT weights loaded from %s", weights_path)
