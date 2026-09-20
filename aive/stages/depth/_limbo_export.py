"""Local multi-view ONNX export for DA3-family depth models (Limbo / official DA3).

Adapted from TheAnimeScripter's ``src/depth/streaming_export.py`` — the same
validated export path TAS uses for its Limbo TensorRT streaming backends:

* spatial size is FIXED (baked into the model; 504x280 widescreen or 504x378
  4:3/square), view count is DYNAMIC (dim 0, min 1 .. window);
* FP16 conversion keeps ``Resize`` / ``LayerNormalization`` / ``Softmax`` /
  ``Exp`` in FP32 and — the critical fix from TAS's validation report — all
  attention matrix products (nodes named ``/attn/MatMul``) in FP32, because an
  earlier all-FP16 candidate produced NaNs on black frames at long view counts.

The export is cached atomically by (weights sha256[:16], shape, precision,
format version), so first use pays the ONNX+engine cost once per model/shape.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import tempfile
import types
from pathlib import Path

import torch

log = logging.getLogger(__name__)

EXPORT_VERSION = 2


class _Positions:
    """Fixed-grid position getter (replaces the dynamic one for export)."""

    def __init__(self, height: int, width: int):
        import torch

        self.grid = torch.cartesian_prod(torch.arange(height), torch.arange(width))

    def __call__(self, batch_size: int, height: int, width: int, device) -> "object":
        return self.grid.to(device).unsqueeze(0).expand(batch_size, -1, -1).clone()


def _head_forward_depth_only(self, feats, H, W, patch_start_idx):
    """DualDPT head forward returning only the depth branch (no conf/sky)."""
    import torch

    from ._da3.model.utils.head_utils import custom_interpolate

    B, _, C = feats[0].shape
    ph, pw = (H // self.patch_size, W // self.patch_size)
    resized = []
    for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
        x = feats[take_idx][:, patch_start_idx:]
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, ph, pw)
        x = self.projects[stage_idx](x)
        if self.pos_embed:
            x = self._add_pos_embed(x, W, H)
        x = self.resize_layers[stage_idx](x)
        resized.append(x)
    l1, l2, l3, l4 = resized
    l1_rn = self.scratch.layer1_rn(l1)
    l2_rn = self.scratch.layer2_rn(l2)
    l3_rn = self.scratch.layer3_rn(l3)
    l4_rn = self.scratch.layer4_rn(l4)
    # Lateral wiring must mirror DualDPT._fuse exactly: each refinenet takes the
    # SAME-level feature map (already upsampled to that level by the previous
    # block) — passing a finer level here is a shape mismatch AND wrong features.
    out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
    out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
    out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
    out = self.scratch.refinenet1(out, l1_rn)
    out = self.scratch.output_conv1(out)
    h_out = int(ph * self.patch_size / self.down_ratio)
    w_out = int(pw * self.patch_size / self.down_ratio)
    out = custom_interpolate(out, (h_out, w_out), mode="bilinear", align_corners=True)
    if self.pos_embed:
        out = self._add_pos_embed(out, W, H)
    logits = self.scratch.output_conv2(out)
    fmap = logits.permute(0, 2, 3, 1)
    depth = self._apply_activation_single(fmap[..., :-1], self.activation).squeeze(-1)
    return {"depth": depth}


class MultiViewDepth:
    """One scene, N ordered views; preserves TAS's first-view camera reference.

    Wraps a loaded DA3 runtime (``wrapper.model`` = DepthAnything3Net) so the
    exported graph takes ``[N, 3, H, W]`` and returns ``[N, H, W]`` depth with
    N dynamic. Only this export instance is mutated (rope forward + head impl);
    global arch classes are never patched.
    """

    def __init__(self, net, height: int, width: int):
        self.net = net
        self.backbone, self.head = net.backbone, net.head
        self.height, self.width = height, width
        vit = self.backbone.pretrained
        vit.position_getter = _Positions(height // 14, width // 14)
        max_position = max(height // 14, width // 14) + 1

        def rope_forward(rope, tokens, positions):
            import torch

            cos, sin = rope._compute_frequency_components(
                tokens.size(-1) // 2, max_position, tokens.device, tokens.dtype
            )
            vertical, horizontal = tokens.chunk(2, dim=-1)
            return torch.cat(
                (
                    rope._apply_1d_rope(vertical, positions[..., 0], cos, sin),
                    rope._apply_1d_rope(horizontal, positions[..., 1], cos, sin),
                ),
                dim=-1,
            )

        # Only mutate this export model; never patch global DA3 classes.
        vit.rope.frequency_cache.clear()
        vit.rope.forward = types.MethodType(rope_forward, vit.rope)
        self.head._forward_impl = types.MethodType(_head_forward_depth_only, self.head)

    def forward(self, image):
        import torch

        camera = self.backbone.pretrained.camera_token
        tokens = torch.cat(
            (camera[:, :1], camera[:, 1:].expand(1, image.shape[0] - 1, -1)), dim=1
        )
        features, _ = self.backbone(
            image.unsqueeze(0),
            cam_token=tokens,
            export_feat_layers=[],
            ref_view_strategy="first",
        )
        # Disable Python chunking: it would freeze the traced view count.
        output = self.head(features, self.height, self.width, patch_start_idx=0, chunk_size=None)
        return output["depth"].reshape(-1, self.height, self.width)

    def __call__(self, image):
        # Plain class (not an nn.Module: the export facade registers `net` itself,
        # so registering here would double-register every parameter).
        return self.forward(image)


def export_streaming_onnx(
    weights_path: str | Path,
    model_name: str,
    cache_dir: str | Path,
    height: int,
    width: int = 504,
    half: bool = True,
) -> Path:
    """Atomically cache the multi-view ONNX export for one (weights, shape, precision).

    ``model_name`` is the DA3 preset (``da3-small`` / ``da3-base``); Limbo v1/v2
    are da3-small finetunes. Returns the .onnx path (existing or freshly built).
    """
    if (height, width) not in ((280, 504), (378, 504)):
        raise ValueError("Streaming depth requires 504x280 or 504x378")

    import torch

    from ._da3_loader import build_da3_model, load_checkpoint_payload, normalize_state_dict, unwrap_state_dict

    weights_path = Path(weights_path)
    with weights_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()[:16]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = (
        cache_dir
        / f"stream_v{EXPORT_VERSION}_{digest}_{width}x{height}_{'fp16' if half else 'fp32'}_{model_name}.onnx"
    )
    if destination.is_file():
        log.info("Streaming ONNX cached: %s", destination.name)
        return destination

    import onnx  # noqa: PLC0415 - heavy, lazy by design

    wrapper = build_da3_model(model_name)
    state = normalize_state_dict(unwrap_state_dict(load_checkpoint_payload(weights_path)), model_name)
    result = wrapper.load_state_dict(state, strict=False)
    del state
    missing = [k for k in result.missing_keys if "_aux" not in k]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(("model.cam_enc.", "model.cam_dec."))]
    if missing or unexpected:
        raise ValueError(
            f"Invalid depth checkpoint: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    net = wrapper.model.eval()
    example = torch.randn(4, 3, height, width, generator=torch.Generator().manual_seed(41))
    with tempfile.TemporaryDirectory(prefix="stream-export-", dir=cache_dir) as temporary:
        fp32_path = str(Path(temporary) / "fp32.onnx")
        with torch.inference_mode():
            reference = net(example.unsqueeze(0), ref_view_strategy="first")["depth"][0]
            wrapper_module = _ExportModule(net, height, width)
            torch.testing.assert_close(wrapper_module(example), reference, atol=1e-5, rtol=1e-5)
            net.backbone.pretrained.rope.frequency_cache.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                torch.onnx.export(
                    wrapper_module,
                    example,
                    fp32_path,
                    opset_version=17,
                    dynamo=False,  # legacy TorchScript exporter: TAS-proven for this graph
                    input_names=["image"],
                    output_names=["depth"],
                    dynamic_axes={"image": {0: "views"}, "depth": {0: "views"}},
                )
        onnx.checker.check_model(onnx.load(fp32_path), full_check=True)
        result_path = fp32_path
        if half:
            result_path = str(Path(temporary) / "fp16.onnx")
            to_fp16(
                fp32_path,
                result_path,
                extra_fp32_ops=["Resize", "LayerNormalization", "Softmax", "Exp"],
            )
        os.replace(result_path, destination)
    log.info("Exported streaming ONNX -> %s (%.1f MB)", destination.name, destination.stat().st_size / 1e6)
    return destination


class _ExportModule(torch.nn.Module):
    """nn.Module facade over MultiViewDepth for torch.onnx.export."""

    def __init__(self, net, height: int, width: int):
        super().__init__()
        self.net = net  # registered submodule -> all params visible to the tracer
        self._inner = MultiViewDepth(net, height, width)

    def forward(self, image):
        return self._inner.forward(image)


def to_fp16(
    src: str,
    dst: str,
    keep_io_types: bool = False,
    extra_fp32_ops: list[str] | None = None,
) -> dict:
    """FP16-convert an ONNX graph with TAS's validated precision blocklist."""
    import onnx
    from onnxruntime.transformers import float16

    m = onnx.load(src)
    n_stripped = _strip_noop_float_casts(m)
    block = sorted(set(float16.DEFAULT_OP_BLOCK_LIST) | set(extra_fp32_ops or []))
    mf = float16.convert_float_to_float16(
        m,
        keep_io_types=keep_io_types,
        op_block_list=block,
        disable_shape_infer=False,
        # Long repeated-view sequences can overflow FP16 attention products.
        node_block_list=[
            node.name for node in m.graph.node if node.op_type == "MatMul" and "/attn/MatMul" in node.name
        ],
    )
    _topo_sort(mf)
    del mf.graph.value_info[:]
    onnx.save(mf, dst)
    onnx.checker.check_model(onnx.load(dst), full_check=True)
    return {
        "path": dst,
        "mb": round(os.path.getsize(dst) / 1000000.0, 1),
        "keep_io_types": keep_io_types,
        "extra_fp32_ops": sorted(extra_fp32_ops or []),
        "noop_casts_stripped": n_stripped,
        "n_nodes": len(mf.graph.node),
    }


def _elem_types(model) -> dict:
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    g = inferred.graph
    types = {}
    for coll in (g.input, g.output, g.value_info):
        for vi in coll:
            types[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        types[init.name] = init.data_type
    return types


def _strip_noop_float_casts(model) -> int:
    from onnx import TensorProto

    types = _elem_types(model)
    g = model.graph
    graph_outputs = {o.name for o in g.output}
    rewire, drop = ({}, [])
    for node in g.node:
        if node.op_type != "Cast" or node.output[0] in graph_outputs:
            continue
        to = next((a.i for a in node.attribute if a.name == "to"), None)
        if to != TensorProto.FLOAT:
            continue
        if types.get(node.input[0]) != TensorProto.FLOAT:
            continue
        rewire[node.output[0]] = node.input[0]
        drop.append(node)

    def resolve(name):
        seen = set()
        while name in rewire and name not in seen:
            seen.add(name)
            name = rewire[name]
        return name

    for node in g.node:
        for i, inp in enumerate(node.input):
            if inp in rewire:
                node.input[i] = resolve(inp)
    for node in drop:
        g.node.remove(node)
    stale = {vi.name for vi in g.value_info} & set(rewire)
    for vi in [vi for vi in g.value_info if vi.name in stale]:
        g.value_info.remove(vi)
    return len(drop)


def _topo_sort(model) -> None:
    g = model.graph
    ready = {i.name for i in g.input} | {init.name for init in g.initializer} | {""}
    pending = list(g.node)
    ordered = []
    while pending:
        progressed = False
        still = []
        for node in pending:
            if all(inp in ready for inp in node.input):
                ordered.append(node)
                ready.update(node.output)
                progressed = True
            else:
                still.append(node)
        pending = still
        if not progressed:
            ordered.extend(pending)
            break
    del g.node[:]
    g.node.extend(ordered)


__all__ = ["EXPORT_VERSION", "export_streaming_onnx", "to_fp16"]
