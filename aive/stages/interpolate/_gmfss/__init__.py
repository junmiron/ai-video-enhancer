"""GMFSS frame-interpolation model (vendored + adapted).

Vendored from GMFSS_Fortuna — https://github.com/98mxr/GMFSS_Fortuna, MIT
license, Copyright (c) 2023 98mxr. Files are unmodified except import paths
and attribution headers; :mod:`aive.stages.interpolate._gmfss.model` assembles
the five subnets and implements the reuse/inference protocol used by
:class:`aive.stages.interpolate.gmfss_model.GMFSSRunner`.

Weights: the TAS "gmfss-fortuna-union" bundle (no upstream license — personal
use only, see manifest). The downloader extracts it into a directory with the
canonical checkpoint names this package loads.
"""
