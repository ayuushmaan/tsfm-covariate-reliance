"""Planted smoke for all 8 new models (beta=1.0, N=8). Per-model try/except."""
import marimo._code_mode as cm

SMOKEALL_CODE = '''import tsfm_audit.adapters_extra as _axe
import importlib as _ilxe
_ilxe.reload(_axe)
from tsfm_audit.harness import run_probe as _rpx
from tsfm_audit.planted import PlantedConfig as _PCx, generate_planted_panel as _gppx

try:
    loaded["toto"] = _axe.load_toto()
    print("toto: LOADED", flush=True)
except Exception as e:
    loaded["toto"] = None
    print(f"toto: FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)

_px = _gppx(1.0, _PCx(n_series=8, seed=0))
_kwx = dict(Y_past=_px["Y_past"], X_past=_px["X_past"], X_future=_px["X_future"], Y_future=_px["Y_future"])
_mk = {
    "tirex": lambda: _axe.make_tirex_predict(loaded["tirex"]),
    "t0": lambda: _axe.make_t0_predict(loaded["t0"], n_known=1),
    "toto": lambda: _axe.make_toto_predict(loaded["toto"]),
    "timesfm3": lambda: _axe.make_timesfm3_predict(loaded["timesfm3"], n_known=1),
    "timesfm25": lambda: _axe.make_timesfm25_predict(loaded["timesfm25"], n_known=1),
    "c2synth": lambda: __import__("tsfm_audit.adapters", fromlist=["make_chronos2_predict"]).make_chronos2_predict(loaded["c2synth"]),
    "moirai2": lambda: _axe.make_moirai2_predict(loaded["moirai2"]),
    "tabpfn": lambda: _axe.make_tabpfn_predict(loaded["tabpfn"]),
}
import json as _jx
_smoke = []
for _name, _fac in _mk.items():
    if loaded.get(_name) is None:
        print(f"{_name}: SKIP (not loaded)")
        continue
    try:
        _rr = _rpx(_name, "planted_beta1", **_kwx, predict=_fac(), seed=0).to_dict()
        _smoke.append(_rr)
        print(f"{_name}: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
    except Exception as e:
        print(f"{_name}: PROBE FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)
with open("/marimo/smoke_all_planted.jsonl", "w") as _f:
    for _r in _smoke:
        _f.write(_jx.dumps(_r) + "\\n")
print(f"banked {len(_smoke)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(SMOKEALL_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
