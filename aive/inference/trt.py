"""TensorRT engine building, caching and execution (primary backend).

Targets TensorRT 10+/11 Python API (explicit-batch only, OnnxParser(network)).
Engines are cached per (GPU, TRT version, model hash, precision) so they are
built once at first use.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _trt():
    import tensorrt as trt  # noqa: PLC0415 - heavy import, lazy by design

    return trt


def gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    return torch.cuda.get_device_name(0)


def engine_cache_dir() -> Path:
    from ..config import model_cache_dir

    p = model_cache_dir() / "engines"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(model_sha256: str, precision: str, profile_json: str) -> str:
    raw = f"{gpu_name()}|{_trt().__version__}|{model_sha256[:16]}|{precision}|{profile_json}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _create_network(builder, trt):
    # TRT 10+: create_network() takes no flags; older versions need EXPLICIT_BATCH.
    try:
        return builder.create_network()
    except TypeError:  # pragma: no cover - legacy TRT <10
        return builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))


def _enable_fp16(config, network, trt) -> None:
    """Enable FP16 across TensorRT versions.

    TRT <= 10: BuilderFlag.FP16 (builder picks per-layer). TRT >= 11 removed the
    flag: set layer precision explicitly; layers that reject it keep default.
    """
    try:
        config.set_flag(trt.BuilderFlag.FP16)  # type: ignore[attr-defined]
        return
    except AttributeError:
        pass
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if hasattr(layer, "precision"):
            try:
                layer.precision = trt.float16
            except Exception:  # pragma: no cover - op without fp16 support
                continue


def build_engine(
    onnx_path: str | Path,
    engine_out: str | Path,
    precision: str = "fp16",
) -> Path:
    """Build a serialized TensorRT engine from an ONNX file.

    If FP16 is requested but the build fails (unsupported op, etc.), retries
    once with default precision so a usable engine always results.
    """
    trt = _trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    data = Path(onnx_path).read_bytes()

    def _build_once(prec: str) -> tuple[bytes, float]:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        network = _create_network(builder, trt)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(data):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed: {errors}")
        if prec == "fp16":
            _enable_fp16(config, network, trt)
        t0 = time.time()
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT engine build returned no data")
        return bytes(engine_bytes), time.time() - t0

    attempts = [precision] + (["default"] if precision == "fp16" else [])
    last_err: Exception | None = None
    for prec in attempts:
        try:
            blob, dt = _build_once(prec)
            break
        except Exception as exc:
            last_err = exc
            log.warning("TRT build with %s failed (%s); trying next precision.", prec, str(exc)[:300])
    else:
        raise RuntimeError(f"TensorRT engine build failed: {last_err}")

    Path(engine_out).parent.mkdir(parents=True, exist_ok=True)
    Path(engine_out).write_bytes(blob)
    log.info("Built TRT %s engine in %.1fs -> %s", prec.upper(), dt, engine_out)
    return Path(engine_out)


class TrtRunner:
    """Fixed-shape TensorRT inference runner with torch-managed CUDA buffers."""

    def __init__(self, engine_path: str | Path):
        import torch

        self.torch = torch
        trt = _trt()
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = self._runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine {engine_path}")
        self.context = self.engine.create_execution_context()

        # TRT >= 10 moved IO metadata from IExecutionContext to ICudaEngine.
        eng = self.engine
        ctx = self.context
        n_io = getattr(ctx, "num_io_tensors", None) or eng.num_io_tensors
        get_name = getattr(ctx, "get_tensor_name", None) or eng.get_tensor_name
        get_mode = getattr(ctx, "get_tensor_mode", None) or eng.get_tensor_mode
        get_dtype = getattr(ctx, "get_tensor_dtype", None) or eng.get_tensor_dtype

        self.io = {}  # name -> dict(mode, dtype_torch, shape)
        for i in range(n_io):
            name = get_name(i)
            mode = get_mode(name)
            try:
                shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            except Exception:  # pragma: no cover - legacy context
                shape = tuple(int(d) for d in eng.get_tensor_shape(name))
            trt_dtype = get_dtype(name)
            torch_dtype = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int8: torch.int8,
                trt.int32: torch.int32,
            }.get(trt_dtype, torch.float32)
            self.io[name] = {"mode": mode, "dtype": torch_dtype, "shape": shape}

        inputs = [n for n in self.io if self.io[n]["mode"] == trt.TensorIOMode.INPUT]
        outputs = [n for n in self.io if self.io[n]["mode"] == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected single input/output, got {inputs}/{outputs}")
        self.in_name, self.out_name = inputs[0], outputs[0]

    def infer(self, x_nchw_f32: np.ndarray) -> np.ndarray:
        torch = self.torch
        trt = _trt()
        spec_in = self.io[self.in_name]
        spec_out = self.io[self.out_name]

        in_t = torch.empty(spec_in["shape"], dtype=spec_in["dtype"], device="cuda")
        out_t = torch.empty(spec_out["shape"], dtype=spec_out["dtype"], device="cuda")
        in_t.copy_(torch.from_numpy(np.ascontiguousarray(x_nchw_f32)).to("cuda", non_blocking=True))

        stream = torch.cuda.current_stream().cuda_stream
        self.context.set_input_shape(self.in_name, spec_in["shape"])
        in_ptr, out_ptr = in_t.data_ptr(), out_t.data_ptr()
        if not self._execute(in_ptr, out_ptr, stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.synchronize()
        return out_t.float().cpu().numpy()

    def _execute(self, in_ptr: int, out_ptr: int, stream: int) -> bool:
        """Run the context across TRT binding generations.

        TRT >= 11 Python: bind tensor addresses first, then execute on a stream.
        Older bindings accept (inputs=..., outputs=...) dicts directly.
        """
        ctx = self.context
        if hasattr(ctx, "set_tensor_address"):
            try:
                ctx.set_tensor_address(self.in_name, in_ptr)
                ctx.set_tensor_address(self.out_name, out_ptr)
                return bool(ctx.execute_async_v3(stream_handle=stream))
            except TypeError:  # pragma: no cover - mixed binding versions
                pass
        return bool(
            ctx.execute_async_v3(
                inputs={self.in_name: in_ptr},
                outputs={self.out_name: out_ptr},
                stream_handle=stream,
            )
        )
