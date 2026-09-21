"""M2M model assembly + the reuse/inference protocol split.

Vendored from feinanshan/M2M_VFI ``Test/model/m2m.py`` (Adobe Research
License, non-commercial research only; CVPR 2022 "Many-to-Many Splatting for
Frame Interpolation"). Differences from upstream:

- the cupy softsplat / costvol kernels are replaced by pure-torch rewrites
  with identical math (:mod:`._m2m.softsplat`, :mod:`._m2m.costvol`);
- deprecated positional ``Tensor.std([1,2,3], False, True)`` -> keyword args;
- the single forward() is split into :meth:`M2MModel.reuse` (once per frame
  pair: replicate-pad to a multiple of ratio*16, per-pair normalize, PWCNet
  bidir flow at half resolution, MotionRefineNet -> 4 branch flows + metrics)
  and :meth:`M2MModel.inference` (per slot: occlusion photo-metrics, many-to-
  many splat over the 4 branches, fallback blend for uncovered pixels,
  denormalize). The math is unchanged; the split exists so a per-slot runner
  contract doesn't re-run flow estimation for every intermediate frame.

Protocol (mirrors GMFSS): :meth:`reuse` returns a tuple of GPU tensors kept
alive for this pair only; :meth:`inference` synthesizes one frame at timestep
``t`` in (0,1) from that state, returning [0,1]-ish NCHW at the *original*
(unpadded) resolution.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backwarp, pwcnet, softsplat

log = logging.getLogger(__name__)


##########################################################


def forwarp_mframe_mask(tenIn1, tenFlow1, t1, tenIn2, tenFlow2, t2, tenMetric1=None, tenMetric2=None):
    """Many-to-many splat: each source pixel renders into multiple targets.

    Vendored verbatim from upstream m2m.py (see module docstring). ``tenIn*`` /
    ``tenFlow*`` / ``t*`` are [branch, N, C|2|1, H, W]; per branch one forward
    and one backward splat accumulate value+weight, which are fused by
    normalization. Returns (output, uncovered_mask) at [N, 3, H, W]. Note the
    upstream argument order: ``t1``/``t2`` are the *weight coefficients* of the
    two directions (the complementary timesteps), not the flow scalings — the
    flows arrive pre-scaled by their own timestep.
    """

    def one_fdir(tenIn, tenFlow, td, tenMetric):
        w = td * tenMetric.clip(-20.0, 20.0).exp()
        tenIn = torch.cat([tenIn * w, w], 1)

        tenOut = softsplat.softsplat_func.apply(tenIn, tenFlow)

        return tenOut[:, :-1, :, :], tenOut[:, -1:, :, :] + 0.0000001

    flow_num = tenFlow1.shape[0]
    tenOut = 0
    tenNormalize = 0
    for idx in range(flow_num):
        tenOutF, tenNormalizeF = one_fdir(tenIn1[idx], tenFlow1[idx], t1[idx], tenMetric1[idx])
        tenOutB, tenNormalizeB = one_fdir(tenIn2[idx], tenFlow2[idx], t2[idx], tenMetric2[idx])

        tenOut += tenOutF + tenOutB
        tenNormalize += tenNormalizeF + tenNormalizeB

    return tenOut / tenNormalize, tenNormalize < 0.00001


###################################################################

c = 16


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        torch.nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        torch.nn.PReLU(out_planes)
    )


class Conv2(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Conv2n(torch.nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2n, self).__init__()
        self.conv1 = conv(in_planes, in_planes, 3, stride, 1)
        self.conv2 = conv(in_planes, in_planes, 3, 1, 1)
        self.conv3 = conv(in_planes, in_planes, 1, 1, 0)
        self.conv4 = conv(in_planes, out_planes, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


#####################################################


class ImgPyramid(torch.nn.Module):
    def __init__(self):
        super(ImgPyramid, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return [x1, x2, x3, x4]


class EncDec(torch.nn.Module):
    def __init__(self, branch):
        super(EncDec, self).__init__()
        self.branch = branch

        self.down0 = Conv2(8, 2*c)
        self.down1 = Conv2(6*c, 4*c)
        self.down2 = Conv2(12*c, 8*c)
        self.down3 = Conv2(24*c, 16*c)

        self.up0 = deconv(48*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = torch.nn.Conv2d(c, 2*self.branch, 3, 1, 1)

        self.conv_m = torch.nn.Conv2d(c, 1, 3, 1, 1)

        # For Channel dimennsion
        self.conv_C = torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool2d(1),
                            torch.nn.Conv2d(16*c, 16*16*c, kernel_size=(1,1), stride=(1,1),padding=(0,0), bias=True),
                            torch.nn.Sigmoid()
                            )

        # For Height dimennsion
        self.conv_H = torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool2d((None,1)),
                            torch.nn.Conv2d(16*c, 16, kernel_size=(1,1), stride=(1,1),padding=(0,0), bias=True),
                            torch.nn.Sigmoid()
                            )

        # For Width dimennsion
        self.conv_W = torch.nn.Sequential(
                            torch.nn.AdaptiveAvgPool2d((1,None)),
                            torch.nn.Conv2d(16*c, 16, kernel_size=(1,1), stride=(1,1),padding=(0,0), bias=True),
                            torch.nn.Sigmoid()
                            )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, flow0, flow1, im0, im1, c0, c1):

        N_,C_,H_,W_ = im0.shape

        wim1 = backwarp.backwarp(im1, flow0)
        wim0 = backwarp.backwarp(im0, flow1)
        s0_0 = self.down0(torch.cat((flow0, im0, wim1), 1))
        s1_0 = self.down0(torch.cat((flow1, im1, wim0), 1))

        #########################################################################################
        flow0 = torch.nn.functional.interpolate(flow0, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1 = torch.nn.functional.interpolate(flow1, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp.backwarp(torch.cat((s0_0, c0[0]),1), flow1)
        wf1 = backwarp.backwarp(torch.cat((s1_0, c1[0]),1), flow0)

        s0_1 = self.down1(torch.cat((s0_0, c0[0], wf1), 1))
        s1_1 = self.down1(torch.cat((s1_0, c1[0], wf0), 1))

        #########################################################################################
        flow0 = torch.nn.functional.interpolate(flow0, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1 = torch.nn.functional.interpolate(flow1, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp.backwarp(torch.cat((s0_1, c0[1]),1), flow1)
        wf1 = backwarp.backwarp(torch.cat((s1_1, c1[1]),1), flow0)

        s0_2 = self.down2(torch.cat((s0_1, c0[1], wf1), 1))
        s1_2 = self.down2(torch.cat((s1_1, c1[1], wf0), 1))

        #########################################################################################
        flow0 = torch.nn.functional.interpolate(flow0, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1 = torch.nn.functional.interpolate(flow1, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp.backwarp(torch.cat((s0_2, c0[2]),1), flow1)
        wf1 = backwarp.backwarp(torch.cat((s1_2, c1[2]),1), flow0)

        s0_3 = self.down3(torch.cat((s0_2, c0[2], wf1), 1))
        s1_3 = self.down3(torch.cat((s1_2, c1[2], wf0), 1))

        #########################################################################################

        s0_3_c = self.conv_C(s0_3)
        s0_3_c = s0_3_c.view(N_,16,-1,1,1)

        s0_3_h = self.conv_H(s0_3)
        s0_3_h = s0_3_h.view(N_,16,1,-1,1)

        s0_3_w = self.conv_W(s0_3)
        s0_3_w = s0_3_w.view(N_,16,1,1,-1)

        cube0 = (s0_3_c*s0_3_h*s0_3_w).mean(1)

        s0_3 = s0_3*cube0

        s1_3_c = self.conv_C(s1_3)
        s1_3_c = s1_3_c.view(N_,16,-1,1,1)

        s1_3_h = self.conv_H(s1_3)
        s1_3_h = s1_3_h.view(N_,16,1,-1,1)

        s1_3_w = self.conv_W(s1_3)
        s1_3_w = s1_3_w.view(N_,16,1,1,-1)

        cube1 = (s1_3_c*s1_3_h*s1_3_w).mean(1)

        s1_3 = s1_3*cube1

        #########################################################################################
        flow0 = torch.nn.functional.interpolate(flow0, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        flow1 = torch.nn.functional.interpolate(flow1, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5

        wf0 = backwarp.backwarp(torch.cat((s0_3, c0[3]),1), flow1)
        wf1 = backwarp.backwarp(torch.cat((s1_3, c1[3]),1), flow0)

        x0 = self.up0(torch.cat((s0_3, c0[3], wf1), 1))
        x1 = self.up0(torch.cat((s1_3, c1[3], wf0), 1))

        x0 = self.up1(torch.cat((s0_2, x0), 1))
        x1 = self.up1(torch.cat((s1_2, x1), 1))

        x0 = self.up2(torch.cat((s0_1, x0), 1))
        x1 = self.up2(torch.cat((s1_1, x1), 1))

        x0 = self.up3(torch.cat((s0_0, x0), 1))
        x1 = self.up3(torch.cat((s1_0, x1), 1))

        m0 = self.sigmoid(self.conv_m(x0))*0.8+0.1
        m1 = self.sigmoid(self.conv_m(x1))*0.8+0.1

        x0 = self.conv(x0)
        x1 = self.conv(x1)

        return x0, x1, m0.repeat(1,self.branch,1,1), m1.repeat(1,self.branch,1,1)


class MotionRefineNet(torch.nn.Module):
    def __init__(self, branch):
        super(MotionRefineNet, self).__init__()
        self.branch = branch
        self.img_pyramid = ImgPyramid()
        self.motion_encdec = EncDec(branch)

    def forward(self, flow0, flow1, im0, im1, ratio):

        flow0 = ratio * torch.nn.functional.interpolate(input=flow0, scale_factor=ratio, mode='bilinear', align_corners=False)
        flow1 = ratio * torch.nn.functional.interpolate(input=flow1, scale_factor=ratio, mode='bilinear', align_corners=False)

        c0 = self.img_pyramid(im0)
        c1 = self.img_pyramid(im1)

        flow_res = self.motion_encdec(flow0, flow1, im0, im1, c0, c1)

        flow0 = flow0.repeat(1,self.branch,1,1) + flow_res[0]
        flow1 = flow1.repeat(1,self.branch,1,1) + flow_res[1]

        return flow0, flow1, flow_res[2], flow_res[3]


class M2MModel(nn.Module):
    """PWCNet bidir flow + MotionRefineNet (4 branches) + many-to-many splat."""

    def __init__(self, ratio=4):
        super().__init__()
        self.branch = 4
        self.ratio = ratio

        self.netFlow = pwcnet.Network()
        self.paramAlpha = nn.Parameter(10.0 * torch.ones(1, 1, 1, 1))
        self.MRN = MotionRefineNet(self.branch)

    # -- weights -------------------------------------------------------------

    def load_weights(self, weights_path: str | Path) -> None:
        """Load the single ``m2m_pretrained.pth`` state dict (strict)."""
        path = Path(weights_path)
        state = torch.load(str(path), map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"M2M checkpoint {path.name} is not a state dict.")
        if all(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        self.load_state_dict(state, strict=True)  # raises on any key mismatch
        log.info(
            "M2M weights loaded from %s (paramAlpha=%.3f)", path.name, float(self.paramAlpha.detach())
        )

    # -- protocol ------------------------------------------------------------

    def reuse(self, im0: torch.Tensor, im1: torch.Tensor):
        """Per-pair cache. ``im0``/``im1`` are NCHW [0,1] at exact resolution.

        Pads (replicate) to a multiple of ratio*16 internally — the runner
        passes unpadded frames and :meth:`inference` crops back. Returns a
        tuple of GPU tensors kept alive for this pair only.
        """
        ratio = self.ratio
        intHeight, intWidth = im0.shape[2], im0.shape[3]

        padr = ((ratio * 16) - (intWidth % (ratio * 16))) % (ratio * 16)
        padb = ((ratio * 16) - (intHeight % (ratio * 16))) % (ratio * 16)
        im0p = F.pad(im0, [0, padr, 0, padb], mode="replicate") if padr or padb else im0
        im1p = F.pad(im1, [0, padr, 0, padb], mode="replicate") if padr or padb else im1

        # Per-pair normalization (upstream's pooled mean/std over both frames).
        m0 = im0p.mean(dim=(1, 2, 3), keepdim=True)
        m1 = im1p.mean(dim=(1, 2, 3), keepdim=True)
        s0 = im0p.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
        s1 = im1p.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
        tenMean_ = (m0 + m1) / 2.0
        tenStd_ = (
            ((s0.square() + (tenMean_ - m0).square()) + (s1.square() + (tenMean_ - m1).square())) / 2.0
        ).sqrt()

        im0n = (im0p - tenMean_) / (tenStd_ + 0.0000001)
        im1n = (im1p - tenMean_) / (tenStd_ + 0.0000001)

        # PWCNet bidir flow at half resolution, refined into branch flows.
        im0h = F.interpolate(im0n, scale_factor=2.0 / ratio, mode="bilinear", align_corners=False)
        im1h = F.interpolate(im1n, scale_factor=2.0 / ratio, mode="bilinear", align_corners=False)

        tenFwd, tenBwd = self.netFlow.bidir(im0h, im1h)
        tenFwd, tenBwd, WeiMF, WeiMB = self.MRN(tenFwd, tenBwd, im0n, im1n, ratio)

        return (im0n, im1n, tenFwd, tenBwd, WeiMF, WeiMB, tenMean_, tenStd_, intHeight, intWidth)

    def inference(self, state: tuple, timestep: float) -> torch.Tensor:
        """Synthesize one frame at `timestep` in (0,1) from a :meth:`reuse` state.

        Returns [0,1]-ish NCHW at the original (unpadded) resolution; values can
        slightly exceed [0,1] near occlusions — the runner clamps to uint8.
        """
        im0n, im1n, tenFwd, tenBwd, WeiMF, WeiMB, tenMean_, tenStd_ = state[:8]
        intHeight, intWidth = state[8], state[9]

        N = im0n.shape[0]
        B = self.branch
        H_, W_ = im0n.shape[2], im0n.shape[3]
        dev = im0n.device
        dtype = im0n.dtype

        # Per-branch layouts (upstream reshapes [N, B*C, ...] -> [B, N, C, ...]).
        im0b = im0n.repeat(1, B, 1, 1).reshape(N, B, 3, H_, W_).permute(1, 0, 2, 3, 4)
        im1b = im1n.repeat(1, B, 1, 1).reshape(N, B, 3, H_, W_).permute(1, 0, 2, 3, 4)

        fwd = tenFwd.reshape(N, B, 2, H_, W_).view(N * B, 2, H_, W_)
        bwd = tenBwd.reshape(N, B, 2, H_, W_).view(N * B, 2, H_, W_)
        wmf = WeiMF.reshape(N, B, 1, H_, W_).view(N * B, 1, H_, W_)
        wmb = WeiMB.reshape(N, B, 1, H_, W_).view(N * B, 1, H_, W_)

        im0f = im0n.repeat(1, B, 1, 1).reshape(N * B, 3, H_, W_)
        im1f = im1n.repeat(1, B, 1, 1).reshape(N * B, 3, H_, W_)

        # Occlusion photo-metrics: how well each direction's flow explains the
        # other frame (channel-mean abs diff, weighted by the branch metric).
        tenPhotoone = (
            1.0 - (wmf * (im0f - backwarp.backwarp(im1f, fwd).detach()).abs().mean(dim=1, keepdim=True))
        ).clip(0.001, None).square()
        tenPhototwo = (
            1.0 - (wmb * (im1f - backwarp.backwarp(im0f, bwd).detach()).abs().mean(dim=1, keepdim=True))
        ).clip(0.001, None).square()

        t0 = torch.full((N * B, 1, 1, 1), float(timestep), device=dev, dtype=dtype)
        t1 = 1.0 - t0

        flow0 = (fwd * t0).reshape(N, B, 2, H_, W_).permute(1, 0, 2, 3, 4)
        flow1 = (bwd * t1).reshape(N, B, 2, H_, W_).permute(1, 0, 2, 3, 4)
        metric0 = (self.paramAlpha * tenPhotoone).reshape(N, B, 1, H_, W_).permute(1, 0, 2, 3, 4)
        metric1 = (self.paramAlpha * tenPhototwo).reshape(N, B, 1, H_, W_).permute(1, 0, 2, 3, 4)

        t0b = torch.full((B, N, 1, 1, 1), float(timestep), device=dev, dtype=dtype)
        t1b = 1.0 - t0b

        tenOutput, mask = forwarp_mframe_mask(im0b, flow0, t1b, im1b, flow1, t0b, metric0, metric1)

        # Uncovered pixels (no splat contribution) fall back to a plain blend.
        tenOutput = tenOutput + mask * (t1b.mean(0) * im0n + t0b.mean(0) * im1n)

        out = tenOutput * (tenStd_ + 0.0000001) + tenMean_
        return out[:, :, :intHeight, :intWidth]
