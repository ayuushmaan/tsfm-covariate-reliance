"""Smoke round 3: t0, toto, timesfm25 (compiled in-cell), moirai2 rebuilt."""
import marimo._code_mode as cm

SMOKE3_CODE = '''import tsfm_audit.adapters_extra as _az3
import importlib as _ilz3
_ilz3.reload(_az3)
from tsfm_audit.harness import run_probe as _rpz3
from tsfm_audit.planted import PlantedConfig as _PCz3, generate_planted_panel as _gpz3
from timesfm.configs import ForecastConfig as _FCz3
try:
    loaded["timesfm25"].compile(_FCz3(max_context=2048, max_horizon=256, per_core_batch_size=32))
    print("timesfm25 compiled", flush=True)
except Exception as e:
    print("compile FAIL:", str(e)[:200], flush=True)
_pz = _gpz3(1.0, _PCz3(n_series=8, seed=0))
_kwz = dict(Y_past=_pz["Y_past"], X_past=_pz["X_past"], X_future=_pz["X_future"], Y_future=_pz["Y_future"])
_mk3 = {
    "t0": lambda: _az3.make_t0_predict(loaded["t0"], n_known=1),
    "toto": lambda: _az3.make_toto_predict(loaded["toto"]),
    "timesfm25": lambda: _az3.make_timesfm25_predict(loaded["timesfm25"], n_known=1),
}
import json as _jz3
_smoke3 = []
for _name, _fac in _mk3.items():
    try:
        _rr = _rpz3(_name, "planted_beta1", **_kwz, predict=_fac(), seed=0).to_dict()
        _smoke3.append(_rr)
        print(f"{_name}: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
    except Exception as e:
        print(f"{_name}: PROBE FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)
with open("/marimo/smoke_all_planted3.jsonl", "w") as _f:
    for _r in _smoke3:
        _f.write(_jz3.dumps(_r) + "\\n")
print(f"banked {len(_smoke3)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(SMOKE3_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
