"""Verify on-disk gluonts is really 0.14.3 (purge stale modules first)."""
import sys as _sy

for _m in [m for m in _sy.modules if m.split(".")[0] in ("gluonts", "uni2ts")]:
    del _sy.modules[_m]
import gluonts
import inspect
print("gluonts:", gluonts.__version__, gluonts.__file__)
from gluonts.model.forecast_generator import QuantileForecastGenerator
print("QFG sig:", inspect.signature(QuantileForecastGenerator.__init__))
