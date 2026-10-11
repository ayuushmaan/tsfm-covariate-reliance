"""Toto median-shape traceback (tiny input)."""
import traceback

import numpy as _np
import tsfm_audit.adapters_extra as _azt

_Y = _np.random.default_rng(0).normal(size=(2, 32))
_Xp = _np.random.default_rng(1).normal(size=(2, 32, 1))
_Xf = _np.random.default_rng(2).normal(size=(2, 8, 1))
try:
    _p = _azt.make_toto_predict(loaded["toto"])(_Y, _Xp, _Xf)
    print("toto OK", _p.shape)
except Exception:
    traceback.print_exc(limit=15)
