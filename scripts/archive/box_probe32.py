"""t0 determinism check + timesfm25 recompile with backcast + probe."""
import tsfm_audit.adapters_extra as _aq2
import importlib as _ilq2
_ilq2.reload(_aq2)
from tsfm_audit.harness import run_probe as _rpq2
from tsfm_audit.planted import PlantedConfig as _PCq2, generate_planted_panel as _gpq2
from timesfm.configs import ForecastConfig as _FCq2

_pq2 = _gpq2(1.0, _PCq2(n_series=8, seed=0))
_kwq2 = dict(Y_past=_pq2["Y_past"], X_past=_pq2["X_past"], X_future=_pq2["X_future"], Y_future=_pq2["Y_future"])
_t0 = _aq2.make_t0_predict(loaded["t0"], n_known=1)
_a = _t0(_pq2["Y_past"], _pq2["X_past"], _pq2["X_future"])
_b = _t0(_pq2["Y_past"], _pq2["X_past"], _pq2["X_future"])
import numpy as _nq2
print("t0 deterministic:", bool(_nq2.array_equal(_a, _b, equal_nan=True)))
loaded["timesfm25"].compile(_FCq2(max_context=2048, max_horizon=256, per_core_batch_size=32, return_backcast=True))
print("2.5 recompiled", flush=True)
_rr = _rpq2("timesfm25", "planted_beta1", **_kwq2, predict=_aq2.make_timesfm25_predict(loaded["timesfm25"], n_known=1), seed=0).to_dict()
print(f"timesfm25: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
import json as _jq2
open("/marimo/smoke_all_planted3.jsonl", "a").write(_jq2.dumps(_rr) + "\n")
