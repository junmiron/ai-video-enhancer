"""M2M frame-interpolation model (vendored + adapted).

Vendored from feinanshan/M2M_VFI — https://github.com/feinanshan/M2M_VFI,
Adobe Research License (non-commercial research only ⚠️), CVPR 2022 "Many-to-
Many Splatting for Frame Interpolation". The upstream reference targets old
torch + cupy CUDA kernels; this package rewrites the two cupy kernels
(softsplat splat, cost volume) in pure torch with identical math and splits
``M2M_PWC.forward`` into a per-pair :meth:`reuse` / per-slot :meth:`inference`
protocol used by :class:`aive.stages.interpolate.m2m_model.M2MRunner`.

Weights: single checkpoint ``m2m_pretrained.pth`` (Adobe Research License,
non-commercial research only — personal use per project decision D3). The
downloader caches the Drive file as-is; no extraction is configured.
"""
