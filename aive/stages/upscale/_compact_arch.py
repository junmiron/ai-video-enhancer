# Vendored from umzi2/traiNNer-redux (a fork of XPixelGroup/BasicSR, Apache 2.0),
# file traiNNer/archs/srvgg_arch.py — the "Compact" SRVGGNetCompact variant used
# for TAS's Cyte V1 / OpenProteus anime checkpoints (options/train/Compact/*.yml:
# type=SRVGGNetCompact, act_type=prelu).
# https://github.com/XPixelGroup/BasicSR  (Apache 2.0)
# https://github.com/umzi2/traiNNer-redux  (Apache 2.0 fork)
#
# Differences from stock BasicSR SRVGGNetCompact (both matter for inference):
# * The final C -> C*upscale^2 conv lives INSIDE `body` (a ModuleList, so state
#   dict keys are body.N.*), followed by a bare PixelShuffle.
# * forward() adds a nearest-neighbor upsampled copy of the input as a residual:
#   out = pixel_shuffle(body(x)) + nn_upsample(x). The network learns a delta on
#   top of the plain upsample, so this term must be kept at inference time.
from __future__ import annotations

import torch
from torch import nn as nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style x2 SR: [Conv, PReLU] stack + PixelShuffle + NN residual."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 2,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.upscale = upscale

        def activation() -> nn.Module:
            if act_type == "relu":
                return nn.ReLU(inplace=True)
            if act_type == "leakyrelu":
                return nn.LeakyReLU(0.1, inplace=True)
            return nn.PReLU(num_parameters=num_feat)  # default

        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), activation()]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(activation())
        # Final conv inside the body (umzi2 variant) -> C*upscale^2 channels.
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale**2, 3, 1, 1))
        self.body = nn.ModuleList(body)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = F.pixel_shuffle(out, self.upscale)
        # Nearest-neighbor residual (umzi2 variant).
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
