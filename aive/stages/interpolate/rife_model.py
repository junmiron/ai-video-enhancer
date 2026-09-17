"""RIFE 4.25 loading, stateless inference helpers and ONNX export.

The vendored arch (``_rife425_arch.IFNet``) is *stateful* by design: its
forward caches ``f0 = encode(img0)`` on the first call and reuses it forever
after (the TAS driver keeps it fresh via a cache/cacheReset protocol). For our
pipeline we want **stateless** semantics — every call computes both encodings
from its own inputs, so no cross-frame state can leak across scene cuts. We get
that by resetting ``f0 = None`` before each torch-mode forward; the ONNX export
traces a fresh instance once and bakes both encodes into the graph, which is
inherently stateless.

Export uses the legacy TorchScript exporter (``dynamo=False``) — the same path
TheAnimeScripter uses for RIFE-on-TensorRT — because the arch's forward mutates
module attributes in ways the torch.export-based exporter does not accept.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build_rife_425(device: str = "cuda", dtype=None):
    """Construct a fresh RIFE 4.25 IFNet (weights NOT loaded)."""
    import torch

    from ._rife425_arch import IFNet

    model = IFNet(ensemble=False, dynamicScale=False, scale=1.0, interpolateFactor=2)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def _normalize_rife_state(state: dict) -> dict:
    """Normalize a RIFE checkpoint to the plain IFNet key layout.

    Official Practical-RIFE releases ship DataParallel *training* checkpoints:
    every key carries a ``module.`` prefix and training-only modules that are
    commented out of the inference arch (``teacher``, ``caltime``) are present.
    Strip the prefix; the extras are dropped by the non-strict load in
    :func:`apply_rife_state`.
    """
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def load_rife_state_dict(weights_path: str | Path) -> dict:
    """Load + normalize RIFE 4.25 weights from disk."""
    import torch

    state = torch.load(str(weights_path), map_location="cpu")
    return _normalize_rife_state(state)


def apply_rife_state(model, state: dict) -> None:
    """Load a normalized state dict into the arch.

    Non-strict on purpose (training-only extras like ``teacher``/``caltime``
    are ignored), but any key the inference arch needs must be present and
    shape-compatible — otherwise the model would run with random weights.
    """
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        sample = ", ".join(result.missing_keys[:6])
        raise RuntimeError(
            f"RIFE 4.25 weights do not match the arch: {len(result.missing_keys)} "
            f"missing key(s), e.g. {sample}"
        )


def load_rife_425(weights_path: str | Path, device: str = "cuda", dtype=None):
    """Load RIFE 4.25 weights (Practical-RIFE flownet.pkl / TAS rife425.pth)."""
    model = build_rife_425(device=device, dtype=dtype)
    state = load_rife_state_dict(weights_path)
    apply_rife_state(model, state)
    del state
    log.info("Loaded RIFE 4.25 weights from %s", weights_path)
    return model


def reset_for_call(model) -> None:
    """Restore stateless semantics before a torch-mode forward (see module doc)."""
    model.f0 = None
    model.counter = 1


class RIFEExportWrapper:
    """Helper that exports the arch to ONNX with named inputs/outputs.

    Inputs: img0 [1,3,H,W], img1 [1,3,H,W], timestep [1,1,H,W] (spatial map,
    constant per call — matches TAS's TRT binding layout). Output: [1,3,H,W].
    """

    @staticmethod
    def export(model, onnx_path: str | Path, height: int, width: int, opset: int = 18) -> Path:
        import torch

        from ...inference.onnx_export import _ensure_self_contained

        model.eval()
        reset_for_call(model)
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        img0 = torch.zeros(1, 3, height, width, device=dev, dtype=dtype)
        img1 = torch.zeros(1, 3, height, width, device=dev, dtype=dtype)
        timestep = torch.full((1, 1, height, width), 0.5, device=dev, dtype=dtype)

        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            import contextlib
            import io

            with contextlib.redirect_stdout(io.StringIO()):
                torch.onnx.export(
                    model,
                    (img0, img1, timestep),
                    str(onnx_path),
                    input_names=["img0", "img1", "timestep"],
                    output_names=["output"],
                    opset_version=opset,
                    dynamo=False,  # legacy TorchScript exporter: TAS-proven for RIFE
                )
        _ensure_self_contained(onnx_path)
        log.info("Exported RIFE ONNX -> %s (%dx%d)", onnx_path, width, height)
        return onnx_path
