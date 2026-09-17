"""PyTorch -> ONNX export helper (feeds the TensorRT engine builder)."""
from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def export_torch_to_onnx(model, example_input, out_path: str | Path, opset: int = 18) -> Path:
    """Export a torch module to ONNX. `example_input` is a cuda tensor used for tracing."""
    import torch

    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=["x"],
        output_names=["y"],
        opset_version=opset,
        dynamic_axes=None,  # fixed-shape engines first; profiles added per-model later
    )
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        # stdout is redirected: the dynamo exporter prints emoji log lines that
        # crash on cp1252/cp437 consoles and in packaged GUI apps without a console.
        try:
            # torch>=2.12's dynamo exporter defaults to external_data=True (a sibling
            # `<name>.onnx.data` file); TensorRT parsing from raw bytes cannot resolve
            # those relative paths, so always request a self-contained model.
            torch.onnx.export(model, (example_input,), str(out_path), external_data=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, (example_input,), str(out_path), **export_kwargs)
    _ensure_self_contained(onnx_path=out_path)
    log.info("Exported ONNX -> %s", out_path)
    return out_path


def _ensure_self_contained(onnx_path: Path) -> None:
    """Guarantee the ONNX file is self-contained (all weights inlined).

    Fallback for exporters/versions that still emit external-data references:
    load (which pulls any sibling `.data` into memory) and re-save inline.
    """
    import onnx

    model = onnx.load(onnx_path)
    if not any(t.external_data for t in model.graph.initializer):
        return
    onnx.save(model, onnx_path)
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    log.info("Inlined external weights into %s", onnx_path)
