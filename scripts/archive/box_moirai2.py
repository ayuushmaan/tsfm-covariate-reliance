"""Purge gluonts+uni2ts, fresh import, recreate predictor, mini predict (one shot)."""
import sys as _sy
for _m in [m for m in _sy.modules if m.split(".")[0] in ("gluonts", "uni2ts")]:
    del _sy.modules[_m]
import gluonts
print("gluonts:", gluonts.__version__)
import tsfm_audit.adapters_extra as _axh
import importlib as _ilh
_ilh.reload(_axh)
_pmh = _axh.load_moirai2(prediction_length=24, context_length=64)
from tsfm_audit.planted import PlantedConfig as _PCh, generate_planted_panel as _gpph
_ph = _gpph(1.0, _PCh(n_series=1, seed=0))
_prh = _axh.make_moirai2_predict(_pmh)(_ph["Y_past"], _ph["X_past"], _ph["X_future"])
import numpy as _nph
print("moirai predict:", _prh.shape, "nan:", _nph.isnan(_prh).sum(), "mean:", _nph.mean(_prh))
