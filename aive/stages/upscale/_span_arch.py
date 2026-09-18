# Vendored from umzi2/traiNNer-redux (a fork of XPixelGroup/BasicSR, Apache 2.0),
# file traiNNer/archs/span_arch.py — the "SPAN" anime super-resolution arch that
# TAS's "Modern Spanimation V2" checkpoint (2x_ModernSpanimationV2.pth, hosted by
# NevermindNilas/TAS-Models-Host) was trained with. The Conv3XC block derives
# from TArdelean/DynamicConvolution (MIT).
# https://github.com/umzi2/traiNNer-redux  (Apache 2.0)
# https://github.com/TArdelean/DynamicConvolution  (MIT, Conv3XC origin)
#
# Inference-only adaptations vs. upstream:
# * forward() always uses the fused single-conv `eval_conv`. Upstream recomputes
#   the fusion on every eval-mode forward; here call fuse() once after loading
#   weights (the checkpoint's stored eval_conv may be stale — during training
#   only the unfused conv+sk path is used, so conv/sk are the source of truth).
# * The checkpoint ships a zero `no_norm` buffer: it was trained with norm=False,
#   i.e. NO input mean/range normalization at inference (raw [0,1] RGB in/out).
from __future__ import annotations

import torch
from torch import nn as nn
from torch.nn import functional as F


class Conv3XC(nn.Module):
    """1x1 skip + (1x1 -> 3x3 -> 1x1) expansion conv, reparameterizable to one 3x3."""

    def __init__(self, c_in: int, c_out: int, gain: int = 2, s: int = 1, bias: bool = True):
        super().__init__()
        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in * gain, 1, bias=bias),
            nn.Conv2d(c_in * gain, c_out * gain, 3, stride=s, bias=bias),
            nn.Conv2d(c_out * gain, c_out, 1, bias=bias),
        )
        self.eval_conv = nn.Conv2d(c_in, c_out, 3, padding=1, stride=s, bias=bias)

    def fuse(self) -> None:
        """Fold sk + conv(0/1/2) into eval_conv (upstream update_params math)."""
        w1, b1 = self.conv[0].weight.data.clone(), self.conv[0].bias.data.clone()
        w2, b2 = self.conv[1].weight.data.clone(), self.conv[1].bias.data.clone()
        w3, b3 = self.conv[2].weight.data.clone(), self.conv[2].bias.data.clone()

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        w_cat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b_cat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = F.pad(self.sk.weight.data.clone(), [1, 1, 1, 1])
        self.eval_conv.weight.data = w_cat + sk_w
        self.eval_conv.bias.data = b_cat + self.sk.bias.data.clone()

    def forward(self, x):
        if self.training:
            # Unfused training path (kept for parity; this app never trains).
            x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
            return self.conv(x_pad) + self.sk(x)
        return self.eval_conv(x)


class SPAB(nn.Module):
    """SPAN attention block: three Conv3XC convs with a sigmoid-gated residual."""

    def __init__(self, in_channels: int, end: bool = False):
        super().__init__()
        self.c1_r = Conv3XC(in_channels, in_channels)
        self.c2_r = Conv3XC(in_channels, in_channels)
        self.c3_r = Conv3XC(in_channels, in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.end = end

    def forward(self, x):
        out1 = self.c1_r(x)
        out2 = self.c2_r(self.act1(out1))
        out3 = self.c3_r(self.act1(out2))
        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        if self.end:
            return out, out1
        return out


class SPAN(nn.Module):
    """SPAN x2 as shipped in 2x_ModernSpanimationV2.pth (TAS "span" model).

    Layout matches the checkpoint exactly: conv_1 + block_1..6 (block_6 is the
    ``end`` variant) + conv_cat(4C->C, 1x1) + conv_2 + Conv/PixelShuffle
    upsampler. Trained with norm=False (checkpoint carries a zero ``no_norm``
    buffer), so no input normalization is applied. Call :meth:`fuse` after
    loading weights; forward then runs on the fused eval_convs only.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        feature_channels: int = 48,
        upscale: int = 2,
    ):
        super().__init__()
        # Present in the checkpoint (zero buffer marking norm=False); kept so
        # state-dict loads are exact. Inference never normalizes the input.
        self.register_buffer("no_norm", torch.zeros(1))

        self.conv_1 = Conv3XC(num_in_ch, feature_channels)
        self.block_1 = SPAB(feature_channels)
        self.block_2 = SPAB(feature_channels)
        self.block_3 = SPAB(feature_channels)
        self.block_4 = SPAB(feature_channels)
        self.block_5 = SPAB(feature_channels)
        self.block_6 = SPAB(feature_channels, end=True)

        self.conv_cat = nn.Conv2d(feature_channels * 4, feature_channels, 1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels)
        self.upsampler = nn.Sequential(
            nn.Conv2d(feature_channels, num_out_ch * upscale**2, 3, padding=1),
            nn.PixelShuffle(upscale),
        )

    def fuse(self) -> None:
        """Fold every Conv3XC into its single eval_conv (call after load)."""
        for m in self.modules():
            if isinstance(m, Conv3XC):
                m.fuse()

    def forward(self, x):
        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6, out_b5_2 = self.block_6(out_b5)
        out_b6 = self.conv_2(out_b6)
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        return self.upsampler(out)
