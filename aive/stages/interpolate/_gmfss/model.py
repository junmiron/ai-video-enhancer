"""GMFSS model assembly: five subnets + the reuse/inference protocol.

Adapted from GMFSS_Fortuna ``model/GMFSS_infer_u.py`` (MIT, Copyright (c) 2023
98mxr). Differences from upstream:

- pure-torch softsplat warp only — the cupy kernel path and its try/except are
  dropped (the torch splat is numerically equivalent; TAS ships a compiled CUDA
  version of the same math);
- :meth:`GMFSSModel.load_weights` takes an explicit weights *directory* and
  accepts both canonical names (``metric.pkl``, ``feat.pkl``, ``fusionnet.pkl``)
  and the TAS union-bundle names (``*_union.pkl``), so a raw TAS-style layout
  works too;
- no module-level device global — the runner owns placement.

Protocol (unchanged from upstream): :meth:`reuse` runs once per frame pair
(flow both directions + occlusion metrics + multi-scale features at half
resolution); :meth:`inference` then synthesizes one intermediate frame at
timestep ``t`` from that cached state, returning clamped [0,1] NCHW at full
(padded) resolution.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FeatureNet import FeatureNet
from .FusionNet_u import GridNet
from .IFNet_HDv3 import IFNet
from .MetricNet import MetricNet
from .gmflow.gmflow import GMFlow
from .softsplat import softsplat as warp

log = logging.getLogger(__name__)


# subnet attribute -> candidate checkpoint filenames (first match wins).
_WEIGHT_FILES: dict[str, tuple[str, ...]] = {
    "flownet": ("flownet.pkl",),
    "ifnet": ("rife.pkl",),
    "metricnet": ("metric.pkl", "metric_union.pkl"),
    "feat_ext": ("feat.pkl", "feat_union.pkl"),
    "fusionnet": ("fusionnet.pkl", "fusionnet_union.pkl"),
}


def _strip_module_prefix(state: dict) -> dict:
    """Drop DataParallel ``module.`` prefixes if every key carries them."""
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


class GMFSSModel(nn.Module):
    """GMFlow + IFNet_HDv3 + MetricNet + FeatureNet + FusionNet_u (union variant)."""

    def __init__(self):
        super().__init__()
        self.flownet = GMFlow()
        self.ifnet = IFNet()
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()

    # -- weights -------------------------------------------------------------

    @staticmethod
    def _pick(weights_dir: Path, candidates: tuple[str, ...]) -> Path:
        for name in candidates:
            p = weights_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"GMFSS weights: none of {list(candidates)} found in {weights_dir}"
        )

    def load_weights(self, weights_dir: str | Path) -> None:
        """Load the five checkpoints from `weights_dir` (canonical or union names)."""
        weights_dir = Path(weights_dir)
        for attr, candidates in _WEIGHT_FILES.items():
            path = self._pick(weights_dir, candidates)
            state = torch.load(str(path), map_location="cpu")
            if not isinstance(state, dict):
                raise RuntimeError(f"GMFSS checkpoint {path.name} is not a state dict.")
            state = _strip_module_prefix(state)
            subnet = getattr(self, attr)
            result = subnet.load_state_dict(state, strict=True)  # noqa: F841 - raises on mismatch
            del state
        log.info("GMFSS weights loaded from %s", weights_dir)

    # -- protocol ------------------------------------------------------------

    def reuse(self, img0: torch.Tensor, img1: torch.Tensor, scale: float = 1.0):
        """Per-pair cache: bidirectional flow + metrics + multi-scale features.

        ``img0``/``img1`` are NCHW [0,1] at full (padded) resolution; ``scale``
        < 1 computes the flow on a further-downscaled copy (TAS uses 0.5 above
        1080p to halve the transformer cost). Returns a tuple of GPU tensors —
        keep it alive for the :meth:`inference` calls of this pair only.
        """
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)
        feat_ext0 = [feat11, feat12, feat13]
        feat_ext1 = [feat21, feat22, feat23]

        img0h = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        img1h = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)

        if scale != 1.0:
            imgf0 = F.interpolate(img0h, scale_factor=scale, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(img1h, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            imgf0, imgf1 = img0h, img1h

        flow01 = self.flownet(imgf0, imgf1)
        flow10 = self.flownet(imgf1, imgf0)
        if scale != 1.0:
            flow01 = F.interpolate(flow01, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            flow10 = F.interpolate(flow10, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale

        metric0, metric1 = self.metricnet(img0h, img1h, flow01, flow10)

        return flow01, flow10, metric0, metric1, feat_ext0, feat_ext1

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        reuse_things: tuple,
        timestep: float | torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize one frame at `timestep` in (0,1) from a :meth:`reuse` cache.

        Returns clamped [0,1] NCHW at the input resolution.
        """
        flow01, metric0 = reuse_things[0], reuse_things[2]
        flow10, metric1 = reuse_things[1], reuse_things[3]
        feat_ext0, feat_ext1 = reuse_things[4], reuse_things[5]

        F1t = timestep * flow01
        F2t = (1 - timestep) * flow10

        Z1t = timestep * metric0
        Z2t = (1 - timestep) * metric1

        img0h = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        I1t = warp(img0h, F1t, Z1t, strMode="soft")
        img1h = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        I2t = warp(img1h, F2t, Z2t, strMode="soft")

        imgs = torch.cat((img0h, img1h), 1)
        rife = self.ifnet(imgs, timestep, scale_list=[8, 4, 2, 1])

        feat1t1 = warp(feat_ext0[0], F1t, Z1t, strMode="soft")
        feat2t1 = warp(feat_ext1[0], F2t, Z2t, strMode="soft")

        F1td = F.interpolate(F1t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat_ext0[1], F1td, Z1d, strMode="soft")
        F2td = F.interpolate(F2t, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor=0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat_ext1[1], F2td, Z2d, strMode="soft")

        F1tdd = F.interpolate(F1t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat1t3 = warp(feat_ext0[2], F1tdd, Z1dd, strMode="soft")
        F2tdd = F.interpolate(F2t, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor=0.25, mode="bilinear", align_corners=False)
        feat2t3 = warp(feat_ext1[2], F2tdd, Z2dd, strMode="soft")

        out = self.fusionnet(
            torch.cat([I1t, rife, I2t], dim=1),
            torch.cat([feat1t1, feat2t1], dim=1),
            torch.cat([feat1t2, feat2t2], dim=1),
            torch.cat([feat1t3, feat2t3], dim=1),
        )

        return torch.clamp(out, 0, 1)
