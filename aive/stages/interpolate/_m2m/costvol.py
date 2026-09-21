# Vendored from feinanshan/M2M_VFI (Adobe Research License, non-commercial
# research only; https://github.com/feinanshan/M2M_VFI), file Test/model/
# costvol.py. Upstream compiles a cupy CUDA kernel at import time; this file
# rewrites the same math in pure torch so no compiled extension or cupy
# dependency is needed.
#
# Math (identical to the CUDA kernel): for each pixel and each of the 81
# offsets (dy, dx) with dy, dx in [-4, +4], the output channel holds the mean
# over input channels of |one[c,y,x] - two[c,y+dy,x+dx]| when the offset lands
# inside the frame, else mean_c |one[c,y,x]| (the kernel's out-of-bounds
# branch). Channel order: dy outer, dx inner. Accumulation is fp32 even for
# fp16 inputs so the channel mean keeps precision (the CUDA kernel accumulates
# in the input type; PWCNet only uses this at its coarsest decoder levels, so
# the extra upcast cost is negligible).
import torch


def costvol(tenOne: torch.Tensor, tenTwo: torch.Tensor) -> torch.Tensor:
    """Mean-abs-diff cost volume with radius 4.

    ``tenOne``/``tenTwo`` are [N,C,H,W]; returns [N,81,H,W] in the input dtype.
    """
    N, C, H, W = tenOne.shape
    R = 4
    out = torch.empty(N, (2 * R + 1) ** 2, H, W, device=tenOne.device, dtype=tenOne.dtype)

    one = tenOne if tenOne.dtype == torch.float32 else tenOne.float()
    two = tenTwo if tenTwo.dtype == torch.float32 else tenTwo.float()
    oob_value = one.abs().mean(dim=1, keepdim=True)  # [N,1,H,W] out-of-bounds value

    yy = torch.arange(H, device=tenOne.device)
    xx = torch.arange(W, device=tenOne.device)
    k = 0
    for dy in range(-R, R + 1):
        y_in = (yy + dy >= 0) & (yy + dy < H)
        for dx in range(-R, R + 1):
            x_in = (xx + dx >= 0) & (xx + dx < W)
            shifted = torch.roll(two, shifts=(-dy, -dx), dims=(2, 3))
            cost = (one - shifted).abs().mean(dim=1, keepdim=True)
            mask = y_in[:, None] & x_in[None, :]  # [H,W] bool
            out[:, k] = torch.where(mask[None], cost, oob_value).to(tenOne.dtype)
            k += 1
    return out
