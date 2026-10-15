"""Vendored AMT frame-interpolation architecture (MCG-NKU/AMT, CC BY-NC 4.0).

Files under ``networks/`` are copied verbatim from the upstream repo with only
their import statements rewritten to package-relative form; :mod:`flow_utils`
carries just the ``warp`` helper those blocks need. The runner that wraps this
architecture lives one level up in :mod:`aive.stages.interpolate.amt_model`.

Upstream: https://github.com/MCG-NKU/AMT — "AMT: All-Pairs Multi-Field Transforms
for Efficient Frame Interpolation" (CVPR 2023). Weights are non-commercial; this
app is personal-use only.
"""
