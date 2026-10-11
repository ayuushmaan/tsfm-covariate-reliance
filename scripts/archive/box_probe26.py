"""Tracebacks for t0, toto, timesfm25-compile (tiny inputs)."""
import traceback

import numpy as _np
import tsfm_audit.adapters_extra as _az

_Y = _np.random.default_rng(0).normal(size=(2, 32))
_Xp = _np.random.default_rng(1).normal(size=(2, 32, 1))
_Xf = _np.random.default_rng(2).normal(size=(2, 8, 1))

print("===== t0 =====")
try:
    _p = _az.make_t0_predict(loaded["t0"], n_known=1)(_Y, _Xp, _Xf)
    print("t0 OK", _p.shape)
except Exception:
    traceback.print_exc(limit=15)
print("===== toto =====")
try:
    _p = _az.make_toto_predict(loaded["toto"])(_Y, _Xp, _Xf)
    print("toto OK", _p.shape)
except Exception:
    traceback.print_exc(limit=12)
print("===== timesfm25 compile =====")
try:
    loaded["timesfm25"].compile()
    print("compile OK")
except Exception:
    traceback.print_exc(limit=12)
