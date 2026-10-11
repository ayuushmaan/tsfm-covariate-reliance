"""Full planted smoke: t0, toto, timesfm25, moirai2 (rebuilt), c2synth re-check."""
import marimo._code_mode as cm

SMOKE2_CODE = '''import tsfm_audit.adapters_extra as _ay
import importlib as _ily
_ily.reload(_ay)
from tsfm_audit.harness import run_probe as _rpy
from tsfm_audit.planted import PlantedConfig as _PCy, generate_planted_panel as _gppy
loaded["moirai2"] = _ay.load_moirai2(prediction_length=168, context_length=512)
print("moirai2 rebuilt", flush=True)
_py = _gppy(1.0, _PCy(n_series=8, seed=0))
_kwy = dict(Y_past=_py["Y_past"], X_past=_py["X_past"], X_future=_py["X_future"], Y_future=_py["Y_future"])
_mk2 = {
    "t0": lambda: _ay.make_t0_predict(loaded["t0"], n_known=1),
    "toto": lambda: _ay.make_toto_predict(loaded["toto"]),
    "timesfm25": lambda: _ay.make_timesfm25_predict(loaded["timesfm25"], n_known=1),
    "moirai2": lambda: _ay.make_moirai2_predict(loaded["moirai2"]),
}
import json as _jy
_smoke2 = []
for _name, _fac in _mk2.items():
    if loaded.get(_name) is None:
        print(f"{_name}: SKIP (not loaded)")
        continue
    try:
        _rr = _rpy(_name, "planted_beta1", **_kwy, predict=_fac(), seed=0).to_dict()
        _smoke2.append(_rr)
        print(f"{_name}: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
    except Exception as e:
        print(f"{_name}: PROBE FAIL {type(e).__name__}: {str(e)[:220]}", flush=True)
with open("/marimo/smoke_all_planted2.jsonl", "w") as _f:
    for _r in _smoke2:
        _f.write(_jy.dumps(_r) + "\\n")
print(f"banked {len(_smoke2)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(SMOKE2_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
