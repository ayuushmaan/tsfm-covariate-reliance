"""Ceiling MASE per task + capture fractions vs banked model intacts."""
import marimo._code_mode as cm

CEIL_CODE = '''import tsfm_audit.ceiling as _cl
import importlib as _ilc
_ilc.reload(_cl)
from tsfm_audit.ceiling import ceiling_ridge as _cr
from tsfm_audit.fev_convert import window_to_arrays as _wc
from tsfm_audit.metrics import mase as _mc
import numpy as _npc
import json as _jc

def _mmase(_Yf, _P, _Yp):
    return float(_npc.nanmean([_mc(_Yf[i], _P[i], _Yp[i]) for i in range(len(_Yp))]))

_ceil = {}
for _tn, _mcx, _sub, _key in [("rohlik_orders_1D", 512, None, "rohlik_orders_1D/w0"),
                         ("uk_covid_nation_1D/cumulative", 512, None, "uk_covid_nation_1D/cumulative/w0"),
                         ("rossmann_1W", 512, 32, "rossmann_1W/w0x32")]:
    _t = [t for t in mini_cov_tasks if t.task_name == _tn][0]
    _a = _wc(_t.get_window(0), max_context=_mcx)
    _ix = list(range(len(_a["ids"])))[:_sub] if _sub else list(range(len(_a["ids"])))
    _P = _cr(_a["Y_past"][_ix], _a["X_past"][_ix], _a["X_future"][_ix])
    _m = _mmase(_a["Y_future"][_ix], _P, _a["Y_past"][_ix])
    _ceil[_key] = _m
    print(f"{_key}: ceiling={_m:.4f}", flush=True)
for _tn in ["epf_np", "solar_with_weather_1H", "uci_air_quality_1H", "proenfo_gfc14"]:
    _t = [t for t in mini_cov_tasks if t.task_name == _tn][0]
    _ms = []
    for _wi in range(_t.num_windows):
        _a = _wc(_t.get_window(_wi), max_context=512)
        _P = _cr(_a["Y_past"], _a["X_past"], _a["X_future"])
        _ms.append(_mmase(_a["Y_future"], _P, _a["Y_past"]))
    _ceil[_tn + "/xwin"] = float(_npc.nanmean(_ms))
    print(f"{_tn}: ceiling={_ceil[_tn + '/xwin']:.4f}", flush=True)
with open("/marimo/ceiling.json", "w") as _f:
    _f.write(_jc.dumps(_ceil, indent=1))
print("banked ceiling.json")

import glob as _gb
_rows = []
for _fp in _gb.glob("/marimo/sweep_*.jsonl"):
    for _ln in open(_fp):
        try:
            _rows.append(_js.loads(_ln))
        except Exception:
            pass
print(f"loaded {len(_rows)} banked rows")
from collections import defaultdict as _dd
_grp = _dd(list)
for _r in _rows:
    if "mase_intact" in _r:
        _grp[(_r["task"], _r["model"])].append(_r)
print(f"{'task':38s} {'model':10s} {'intact':>7s} {'gap':>7s} {'capture':>7s}")
for (_task, _model), _rs in sorted(_grp.items()):
    _CEIL = [m for m in _rows if False]
    _intact = float(_npc.mean([r["mase_intact"] for r in _rs]))
    _gap = float(_npc.mean([r["swap_gap"] for r in _rs]))
    _cm = _ceil.get(_task)
    _bolt = _npc.mean([r["mase_intact"] for r in _grp.get((_task, "bolt"), [])])
    if _cm is None or not _bolt or (_bolt - _cm) <= 1e-9:
        _cap = "n/a"
    else:
        _cap = f"{(_bolt - _intact) / (_bolt - _cm):+.2f}"
    print(f"{_task:38s} {_model:10s} {_intact:7.3f} {_gap:+7.3f} {_cap:>7s}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(CEIL_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
