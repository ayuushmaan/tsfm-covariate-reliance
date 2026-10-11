"""TimesFM25 batched probe (fixed) — print shapes + gaps."""
import tsfm_audit.adapters_extra as _aq3
import importlib as _ilq3
_ilq3.reload(_aq3)
from tsfm_audit.harness import run_probe as _rpq3
from tsfm_audit.planted import PlantedConfig as _PCq3, generate_planted_panel as _gpq3

_pq3 = _gpq3(1.0, _PCq3(n_series=8, seed=0))
_kwq3 = dict(Y_past=_pq3["Y_past"], X_past=_pq3["X_past"], X_future=_pq3["X_future"], Y_future=_pq3["Y_future"])
try:
    _rr = _rpq3("timesfm25", "planted_beta1", **_kwq3, predict=_aq3.make_timesfm25_predict(loaded["timesfm25"], n_known=1), seed=0).to_dict()
    print(f"timesfm25: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
    import json as _jq3
    open("/marimo/smoke_all_planted3.jsonl", "a").write(_jq3.dumps(_rr) + "\n")
except Exception as e:
    import traceback
    traceback.print_exc(limit=15)
