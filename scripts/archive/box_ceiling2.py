"""Ceiling v2 + bank-file census + capture table (ceiling = min(ridge, naive))."""
import marimo._code_mode as cm

CEIL2_CODE = '''import tsfm_audit.ceiling as _cl2
import importlib as _ilc2
_ilc2.reload(_cl2)
from tsfm_audit.ceiling import ceiling_ridge as _cr2
from tsfm_audit.fev_convert import window_to_arrays as _wc2
from tsfm_audit.metrics import mase as _mc2
import numpy as _npc2
import json as _jc2
import glob as _gb2
import os as _os2
print("jsonl on box:", sorted(_os2.path.basename(p) for p in _gb2.glob("/marimo/*.jsonl")))

def _mm2(_Yf, _P, _Yp):
    return float(_npc2.nanmean([_mc2(_Yf[i], _P[i], _Yp[i]) for i in range(len(_Yp))]))

def _naive(_Yp, _H):
    import numpy as _nq
    return _nq.repeat(_Yp[:, -1:], _H, axis=1)

_ceil2 = {}
_jobs = [("rohlik_orders_1D", 512, None, "rohlik_orders_1D/w0"),
         ("uk_covid_nation_1D/cumulative", 512, None, "uk_covid_nation_1D/cumulative/w0"),
         ("rossmann_1W", 512, 32, "rossmann_1W/w0x32")]
for _tn, _mcx, _sub, _key in _jobs:
    _t = [t for t in mini_cov_tasks if t.task_name == _tn][0]
    _a = _wc2(_t.get_window(0), max_context=_mcx)
    _ix = list(range(len(_a["ids"])))[:_sub] if _sub else list(range(len(_a["ids"])))
    _skip = tuple(_a["col_names"].index(c) for c in _a["cat_maps"])
    _Pr = _cr2(_a["Y_past"][_ix], _a["X_past"][_ix], _a["X_future"][_ix], skip_cols=_skip)
    _mr = _mm2(_a["Y_future"][_ix], _Pr, _a["Y_past"][_ix])
    _mn = _mm2(_a["Y_future"][_ix], _naive(_a["Y_past"][_ix], _a["Y_future"].shape[1]), _a["Y_past"][_ix])
    _ceil2[_key] = min(_mr, _mn)
    print(f"{_key}: ridge={_mr:.4f} naive={_mn:.4f} -> ceiling={_ceil2[_key]:.4f}", flush=True)
for _tn in ["epf_np", "solar_with_weather_1H", "uci_air_quality_1H", "proenfo_gfc14"]:
    _t = [t for t in mini_cov_tasks if t.task_name == _tn][0]
    _ms = []
    for _wi in range(_t.num_windows):
        _a = _wc2(_t.get_window(_wi), max_context=512)
        _skip = tuple(_a["col_names"].index(c) for c in _a["cat_maps"])
        _Pr = _cr2(_a["Y_past"], _a["X_past"], _a["X_future"], skip_cols=_skip)
        _mr = _mm2(_a["Y_future"], _Pr, _a["Y_past"])
        _mn = _mm2(_a["Y_future"], _naive(_a["Y_past"], _a["Y_future"].shape[1]), _a["Y_past"])
        _ms.append(min(_mr, _mn))
    _ceil2[_tn + "/xwin"] = float(_npc2.nanmean(_ms))
    print(f"{_tn}/xwin: ceiling={_ceil2[_tn + '/xwin']:.4f}", flush=True)
with open("/marimo/ceiling.json", "w") as _f:
    _f.write(_jc2.dumps(_ceil2, indent=1))
print("banked ceiling.json")

_rows = []
for _fp in _gb2.glob("/marimo/sweep_*.jsonl"):
    for _ln in open(_fp):
        try:
            _rows.append(_jc2.loads(_ln))
        except Exception:
            pass
print(f"loaded {len(_rows)} banked rows")
from collections import defaultdict as _dd2
_grp = _dd2(list)
for _r in _rows:
    if "mase_intact" in _r:
        _grp[(_r["task"], _r["model"])].append(_r)
print(f"{'task':38s} {'model':10s} {'intact':>7s} {'gap':>7s} {'capture':>7s}")
for (_task, _model), _rs in sorted(_grp.items()):
    _intact = float(_npc2.mean([r["mase_intact"] for r in _rs]))
    _gap = float(_npc2.mean([r["swap_gap"] for r in _rs]))
    _cm = _ceil2.get(_task)
    _bl = _grp.get((_task, "bolt"), [])
    _bolt = float(_npc2.mean([r["mase_intact"] for r in _bl])) if _bl else float("nan")
    if _cm is None or not (_bolt > _cm):
        _cap = "n/a"
    else:
        _cap = f"{(_bolt - _intact) / (_bolt - _cm):+.2f}"
    print(f"{_task:38s} {_model:10s} {_intact:7.3f} {_gap:+7.3f} {_cap:>7s}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(CEIL2_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
