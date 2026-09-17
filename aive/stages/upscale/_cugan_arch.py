# Vendored from spandrel's RealCUGAN architecture (spandrel/model/image/arch/realcugan/__arch/upcunet_v3.py),
# which derives from bilibili/Real-CUGAN-models. MIT license, Copyright (c) 2021 bilibili.
# https://github.com/bilibili/Real-CUGAN-models
#
# Only the pieces this project needs are kept: the shared blocks and
# UpCunet2x_fast — the "ShuffleCUGAN" variant used by sudo_shuffle_cugan_*.pth
# (3-channel input, internal PixelUnshuffle(2) to 12 channels at half res).
# The upstream cache-quantization helpers (q/dq) and TAS's split forward_a/b/c/d
# tiling-cache methods are intentionally omitted: this project tiles with
# overlap + feathered blending instead (see tiling.py).
from __future__ import annotations

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction=8, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // reduction, 1, 1, 0, bias=bias
        )
        self.conv2 = nn.Conv2d(
            in_channels // reduction, in_channels, 1, 1, 0, bias=bias
        )

    def forward(self, x):
        if "Half" in x.type():  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x


class UNetConv(nn.Module):
    def __init__(
        self, in_channels: int, mid_channels: int, out_channels: int, se: bool
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z


class UNet1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool):
        super().__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # type: ignore
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-4, -4, -4, -4))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z


class UNet2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool):
        super().__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:  # type: ignore
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha: float = 1):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x1 = F.pad(x1, (-16, -16, -16, -16))
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x3 = self.conv2_down(x2)
        x2 = F.pad(x2, (-4, -4, -4, -4))
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x4 = self.conv4(x2 + x3)
        x4 *= alpha
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)
        z = self.conv_bottom(x5)
        return z


class UpCunet2x_fast(nn.Module):
    """ShuffleCUGAN x2: 3-channel input -> 2x output.

    The "shuffle" trick: the input is PixelUnshuffled(2) to 12 channels at half
    resolution before entering the UNets, and a final conv + PixelShuffle(2)
    restores 3 channels at full (2x) resolution. A nearest-neighbor upsample of
    the original input is added on top (residual).

    Size constraints (verified against sudo_shuffle_cugan_9.584.969.pth):
      * every input dimension must be a multiple of 4, and
      * every input dimension must be > 38 (reflect padding limit) — in
        practice >= 40. Callers pad to multiples of 4 before invoking the model.
    """

    def __init__(self, *, in_channels=3, out_channels=3):
        super().__init__()
        self.unet1 = UNet1(4 * in_channels, 64, deconv=True)
        self.unet2 = UNet2(64, 64, deconv=False)
        self.ps = nn.PixelShuffle(2)
        self.conv_final = nn.Conv2d(64, 4 * out_channels, 3, 1, padding=0, bias=True)
        self.inv = nn.PixelUnshuffle(2)

    def forward(self, x: Tensor):
        _, _, h0, w0 = x.shape
        x00 = x
        ph = ((h0 - 1) // 2 + 1) * 2
        pw = ((w0 - 1) // 2 + 1) * 2
        x = F.pad(x, (38, 38 + pw - w0, 38, 38 + ph - h0), "reflect")  # 需要保证被2整除
        x = self.inv(x)  # +18
        x = self.unet1.forward(x)
        x0 = self.unet2.forward(x)
        x1 = F.pad(x, (-20, -20, -20, -20))
        x = torch.add(x0, x1)
        x = self.conv_final(x)
        x = F.pad(x, (-1, -1, -1, -1))
        x = self.ps(x)
        if w0 != pw or h0 != ph:
            x = x[:, :, : h0 * 2, : w0 * 2]
        x += F.interpolate(x00, scale_factor=2, mode="nearest")
        return x


__all__ = ["SEBlock", "UNetConv", "UNet1", "UNet2", "UpCunet2x_fast"]
