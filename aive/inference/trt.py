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


def _free_vram_nvidia_smi() -> int | None:
    """Free VRAM per nvidia-smi (MiB->bytes), or None when unavailable.

    cudaMemGetInfo is blind to memory other processes hold via CUDA VMM
    (cuMemMap — e.g. llama.cpp's allocator): it can report tens of GiB free
    while the physical device is nearly full, and engine builds then OOM on
    tiny commits. nvidia-smi sees the true physical usage.
    """
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        first = out.stdout.strip().splitlines()[0].strip()
        return int(first) * (1 << 20)
    except Exception:  # pragma: no cover - nvidia-smi quirks
        return None


def free_vram_bytes() -> int:
    """Conservative estimate of free GPU VRAM in bytes.

    Takes the MINIMUM of cudaMemGetInfo and nvidia-smi (when available) so VMM-
    mapped usage by other processes is accounted for. 0 when CUDA unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info()
        best = int(free)
    except Exception:  # pragma: no cover - non-CUDA environments
        return 0
    smi_free = _free_vram_nvidia_smi()
    if smi_free is not None:
        best = min(best, smi_free)
    return best


def build_engine(
    onnx_path: str | Path,
    engine_out: str | Path,
    precision: str = "fp16",
    workspace_bytes: int | None = None,
    profiles: dict[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None,
) -> Path:
    """Build a serialized TensorRT engine from an ONNX file.

    If FP16 is requested but the build fails (unsupported op, etc.), retries
    once with default precision so a usable engine always results.

    ``workspace_bytes=None`` adapts to current free VRAM: up to 4 GiB, but no
    more than half of what is actually free — a fixed 4 GiB pool makes the
    build OOM when other processes (e.g. an LLM server) hold most of the GPU.

    ``profiles`` optionally declares dynamic input shapes as
    ``{input_name: (min_shape, opt_shape, max_shape)}`` — used by the depth
    streaming engines whose view count varies per chunk (full window vs tail).
    """
    trt = _trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    data = Path(onnx_path).read_bytes()

    if workspace_bytes is None:
        free = free_vram_bytes()
        workspace_bytes = int(min(4 << 30, max(512 << 20, (free // 2) if free else 4 << 30)))

    def _build_once(prec: str) -> tuple[bytes, float]:
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        network = _create_network(builder, trt)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(data):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX parse failed: {errors}")
        if prec == "fp16":
            _enable_fp16(config, network, trt)
        if profiles:
            profile = builder.create_optimization_profile()
            for name, (min_s, opt_s, max_s) in profiles.items():
                profile.set_shape(name, min_s, opt_s, max_s)
            config.add_optimization_profile(profile)
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
    """Fixed-shape TensorRT inference runner with torch-managed CUDA buffers.

    Supports single-input/single-output engines (``infer``) and multi-I/O
    engines such as RIFE's img0/img1/timestep -> output (``infer_multi``).
    """

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

        self.in_names = [n for n in self.io if self.io[n]["mode"] == trt.TensorIOMode.INPUT]
        self.out_names = [n for n in self.io if self.io[n]["mode"] == trt.TensorIOMode.OUTPUT]
        if not self.in_names or not self.out_names:
            raise RuntimeError(f"Engine has no IO tensors: {list(self.io)}")

    def infer(self, x_nchw_f32: np.ndarray) -> np.ndarray:
        """Single-input/single-output convenience (float output as numpy)."""
        if len(self.in_names) != 1 or len(self.out_names) != 1:
            raise RuntimeError("infer() needs a single-I/O engine; use infer_multi()")
        return self.infer_multi({self.in_names[0]: x_nchw_f32})[self.out_names[0]]

    def infer_multi(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the engine with named CPU numpy inputs; returns float32 numpy outputs."""
        torch = self.torch
        in_tensors = {}
        for name in self.in_names:
            spec = self.io[name]
            arr = np.ascontiguousarray(inputs[name])
            if tuple(arr.shape) != spec["shape"]:
                raise ValueError(f"Input '{name}': expected shape {spec['shape']}, got {arr.shape}")
            t = torch.empty(spec["shape"], dtype=spec["dtype"], device="cuda")
            t.copy_(torch.from_numpy(arr).to("cuda", non_blocking=True))
            in_tensors[name] = t
        out_tensors = {
            name: torch.empty(self.io[name]["shape"], dtype=self.io[name]["dtype"], device="cuda")
            for name in self.out_names
        }

        stream = torch.cuda.current_stream().cuda_stream
        ctx = self.context
        for name in self.in_names:
            try:  # fixed-shape engines: informational, but required on some bindings
                ctx.set_input_shape(name, tuple(self.io[name]["shape"]))
            except Exception:  # pragma: no cover - legacy context
                pass

        ok = False
        if hasattr(ctx, "set_tensor_address"):
            try:
                for name, t in in_tensors.items():
                    ctx.set_tensor_address(name, t.data_ptr())
                for name, t in out_tensors.items():
                    ctx.set_tensor_address(name, t.data_ptr())
                ok = bool(ctx.execute_async_v3(stream_handle=stream))
            except TypeError:  # pragma: no cover - mixed binding versions
                ok = False
        if not ok:
            ok = bool(
                ctx.execute_async_v3(
                    inputs={n: t.data_ptr() for n, t in in_tensors.items()},
                    outputs={n: t.data_ptr() for n, t in out_tensors.items()},
                    stream_handle=stream,
                )
            )
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.synchronize()
        return {name: t.float().cpu().numpy() for name, t in out_tensors.items()}

    def infer_dynamic(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run a dynamic-shape engine (built with ``profiles``).

        Input shapes may vary per call within the declared min..max profile;
        output buffers are re-derived from the context after each shape set.
        Fixed-shape engines work too (shapes just never change).
        """
        torch = self.torch
        ctx = self.context
        in_tensors = {}
        for name in self.in_names:
            arr = np.ascontiguousarray(inputs[name])
            spec = self.io[name]
            if not all(
                a == b or (isinstance(b, int) and a >= 1)
                for a, b in zip(arr.shape, spec["shape"])
            ):
                raise ValueError(f"Input '{name}': shape {arr.shape} incompatible with engine dims")
            if not ctx.set_input_shape(name, tuple(arr.shape)):
                raise RuntimeError(f"TensorRT rejected input shape for '{name}': {tuple(arr.shape)}")
            t = torch.empty(tuple(arr.shape), dtype=spec["dtype"], device="cuda")
            t.copy_(torch.from_numpy(arr).to("cuda", non_blocking=True))
            in_tensors[name] = t
        out_tensors = {}
        for name in self.out_names:
            spec = self.io[name]
            try:
                shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            except Exception:  # pragma: no cover - legacy context
                shape = spec["shape"]
            out_tensors[name] = torch.empty(shape, dtype=spec["dtype"], device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        ok = False
        if hasattr(ctx, "set_tensor_address"):
            try:
                for name, t in in_tensors.items():
                    ctx.set_tensor_address(name, t.data_ptr())
                for name, t in out_tensors.items():
                    ctx.set_tensor_address(name, t.data_ptr())
                ok = bool(ctx.execute_async_v3(stream_handle=stream))
            except TypeError:  # pragma: no cover - mixed binding versions
                ok = False
        if not ok:
            ok = bool(
                ctx.execute_async_v3(
                    inputs={n: t.data_ptr() for n, t in in_tensors.items()},
                    outputs={n: t.data_ptr() for n, t in out_tensors.items()},
                    stream_handle=stream,
                )
            )
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 (dynamic) failed")
        torch.cuda.synchronize()
        return {name: t.float().cpu().numpy() for name, t in out_tensors.items()}
