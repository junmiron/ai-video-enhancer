"""Inference backend layer: torch -> ONNX -> TensorRT engine, validated against PyTorch."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trt = pytest.importorskip("tensorrt")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def test_torch_onnx_trt_roundtrip(tmp_path):
    from aive.inference.backend import TorchRunner
    from aive.inference.onnx_export import export_torch_to_onnx
    from aive.inference.trt import TrtRunner, build_engine

    torch.manual_seed(0)
    model = TinyNet()
    example = torch.randn(1, 3, 64, 64, device="cuda")

    onnx_path = export_torch_to_onnx(model, example, tmp_path / "tiny.onnx")
    assert onnx_path.exists() and onnx_path.stat().st_size > 0

    engine_path = build_engine(onnx_path, tmp_path / "engines" / "tiny.engine", precision="fp16")
    assert engine_path.exists() and engine_path.stat().st_size > 10_000

    x = np.random.randn(1, 3, 64, 64).astype(np.float32)
    ref = TorchRunner(model).infer(x)
    got = TrtRunner(engine_path).infer(x)

    assert got.shape == ref.shape
    # FP16 engine vs FP16 torch reference: tight tolerance.
    np.testing.assert_allclose(got, ref, atol=2e-2, rtol=5e-2)


def test_engine_cache_key_stable():
    from aive.inference.trt import _cache_key

    k1 = _cache_key("a" * 64, "fp16", "{}")
    k2 = _cache_key("a" * 64, "fp16", "{}")
    k3 = _cache_key("b" * 64, "fp16", "{}")
    assert k1 == k2 and k1 != k3
