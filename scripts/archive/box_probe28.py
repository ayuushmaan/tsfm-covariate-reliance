"""t0 shape test (raw vs TimeSeries) + timesfm layout."""
import pathlib
import torch

tfm = pathlib.Path("/tmp/uv-venv/lib/python3.13/site-packages/timesfm")
print("timesfm files:", sorted(p.name for p in tfm.iterdir()))
print("timesfm3 files:", sorted((tfm / "timesfm3").glob("*.py")) if (tfm / "timesfm3").exists() else None)

from t0 import T0Forecaster
from t0.model.model import TimeSeries

_m = loaded["t0"]
_ctx = torch.randn(2, 1, 32)
_fcov = torch.randn(2, 2, 40)
_o1 = _m.predict(_ctx, horizon=8, quantile_levels=[0.5], future_covariates=_fcov)
print("raw ctx out:", type(_o1.median), tuple(_o1.median.shape))
_tsi = TimeSeries.from_array(_ctx, _fcov)
_o2 = _m.predict(_tsi, horizon=8, quantile_levels=[0.5])
print("TimeSeries out:", tuple(_o2.median.shape))
_o3 = _m.predict(_ctx, horizon=8, quantile_levels=[0.5])
print("no-cov out:", tuple(_o3.median.shape))
