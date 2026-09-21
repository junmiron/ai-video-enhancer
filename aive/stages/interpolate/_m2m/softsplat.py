# Vendored from feinanshan/M2M_VFI (Adobe Research License, non-commercial
# research only; https://github.com/feinanshan/M2M_VFI), file Test/model/
# softsplat.py — the splat math is sniklaus/softmax-splatting. Upstream
# compiles a cupy CUDA kernel at import time; this file rewrites the same
# forward scatter in pure torch, modeled on the vendored GMFSS splat
# (aive/stages/interpolate/_gmfss/softsplat.py).
#
# M2M calls ``softsplat_func.apply(tenIn, tenFlow)`` with a pre-concatenated
# [N, C+1, H, W] input whose last channel carries the per-pixel weight; this
# raw scatter accumulates value and weight alike (no normalization here —
# M2M's forwarp_mframe_mask divides by the accumulated weight itself). The
# class name/apply interface is kept so upstream call sites stay unchanged.
import torch


_grid_cache: dict = {}
_batch_cache: dict = {}


class softsplat_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tenIn: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
        N, C, H, W = tenIn.size()
        device = tenIn.device
        dtype = tenIn.dtype

        key = (H, W, str(device), dtype)
        if key not in _grid_cache:
            gridY, gridX = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            _grid_cache[key] = (
                gridY.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W),
                gridX.unsqueeze(0).unsqueeze(0).expand(N, 1, H, W),
            )
        if key not in _batch_cache:
            _batch_cache[key] = (
                torch.arange(N, device=device)
                .view(N, 1, 1)
                .expand(N, H, W)
                .reshape(-1)
            )
        gridY, gridX = _grid_cache[key]
        batch_idx = _batch_cache[key]

        fltX = (gridX + tenFlow[:, 0:1]).reshape(-1)
        fltY = (gridY + tenFlow[:, 1:2]).reshape(-1)
        tenIn_flat = tenIn.permute(0, 2, 3, 1).reshape(-1, C)

        finite = torch.isfinite(fltX) & torch.isfinite(fltY)
        if not finite.any():
            return torch.zeros_like(tenIn)

        fltX, fltY = fltX[finite], fltY[finite]
        tenIn_flat = tenIn_flat[finite]
        batch_idx = batch_idx[finite]

        intNW_X = torch.floor(fltX).to(dtype=torch.int32)
        intNW_Y = torch.floor(fltY).to(dtype=torch.int32)
        intNE_X = intNW_X + 1
        intSW_Y = intNW_Y + 1

        fltNW = (intNE_X - fltX) * (intSW_Y - fltY)
        fltNE = (fltX - intNW_X) * (intSW_Y - fltY)
        fltSW = (intNE_X - fltX) * (fltY - intNW_Y)
        fltSE = (fltX - intNW_X) * (fltY - intNW_Y)

        tenOut_flat = torch.zeros(N * H * W, C, device=device, dtype=dtype)

        positions = [
            (intNW_X, intNW_Y, fltNW),
            (intNE_X, intNW_Y, fltNE),
            (intNW_X, intSW_Y, fltSW),
            (intNE_X, intSW_Y, fltSE),
        ]

        for intX, intY, weight in positions:
            valid = (intX >= 0) & (intX < W) & (intY >= 0) & (intY < H)
            if not valid.any():
                continue
            idx_NHW = (
                batch_idx[valid] * H * W
                + intY[valid].to(torch.int64) * W
                + intX[valid].to(torch.int64)
            )
            tenOut_flat.index_add_(0, idx_NHW, tenIn_flat[valid] * weight[valid].unsqueeze(1))

        return tenOut_flat.view(N, H, W, C).permute(0, 3, 1, 2)
