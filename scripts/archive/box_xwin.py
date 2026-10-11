"""Cross-window probe on single-series tasks (synthetic per-window stamps)."""
import marimo._code_mode as cm

WIN_CODE = '''import tsfm_audit.adapters as _aw
import tsfm_audit.fev_convert as _fw
import tsfm_audit.harness as _hw
import importlib as _ilw
for _mm in (_aw, _fw, _hw):
    _ilw.reload(_mm)
from tsfm_audit.adapters import make_bolt_predict as _mk_bw, make_chronos2_predict as _mk_cw
from tsfm_audit.fev_convert import window_to_arrays as _ww
from tsfm_audit.harness import run_probe_windows as _rpw
import json as _jw

win_rows = []
for _tname in ["epf_np", "solar_with_weather_1H", "uci_air_quality_1H", "proenfo_gfc14"]:
    _tw = [t for t in mini_cov_tasks if t.task_name == _tname][0]
    print(f"{_tname}: num_windows={_tw.num_windows} H={_tw.horizon}", flush=True)
for _tname in ["epf_np", "solar_with_weather_1H"]:
    _tw = [t for t in mini_cov_tasks if t.task_name == _tname][0]
    if _tw.num_windows < 2:
        print(f"{_tname}: skip (<2 windows)")
        continue
    _wins = []
    for _wi in range(_tw.num_windows):
        _a = _ww(_tw.get_window(_wi), max_context=512)
        _wins.append({"Y_past": _a["Y_past"][0], "X_past": _a["X_past"][0],
                      "X_future": _a["X_future"][0], "Y_future": _a["Y_future"][0]})
    print(f"{_tname}: converted {len(_wins)} windows", flush=True)
    _wc = _mk_cw(chronos2_pipe)
    _wb = _mk_bw(bolt_pipe)
    for _s in [0, 1]:
        for _mn, _mp in [("chronos2", _wc), ("bolt", _wb)]:
            _rr = _rpw(_mn, f"{_tname}/xwin", _wins, predict=_mp, seed=_s).to_dict()
            win_rows.append(_rr)
            print(f"  seed={_s} {_mn}: intact={_rr['mase_intact']:.4f} swapped={_rr['mase_swapped']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
with open("/marimo/grid_xwin.jsonl", "w") as _f:
    for _r in win_rows:
        _f.write(_jw.dumps(_r) + "\\n")
print(f"banked {len(win_rows)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(WIN_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
