"""TabPFN full grid: 3 series tasks + 4 window tasks (token already in box env)."""
import marimo._code_mode as cm

TABPFN_CODE = '''import tsfm_audit.adapters_extra as _axt2
import importlib as _ilt2
_ilt2.reload(_axt2)
import tsfm_audit.fev_convert as _fct2
_ilt2.reload(_fct2)
from tsfm_audit.fev_convert import window_to_arrays as _wt2
from tsfm_audit.harness import run_probe as _rp2, run_probe_windows as _rq2
import json as _jt2

_tp = _axt2.make_tabpfn_predict(loaded["tabpfn"])
_tab_rows = []
# series tasks (known + past + static; lib keeps known only)
for _tn, _seeds, _mc, _sub in [("rohlik_orders_1D", [0, 1, 2], 512, None),
                               ("uk_covid_nation_1D/cumulative", [0, 1, 2], 512, None),
                               ("rossmann_1W", [0, 1], 512, 32)]:
    _tt = [t for t in mini_cov_tasks if t.task_name == _tn][0]
    _a = _wt2(_tt.get_window(0), max_context=_mc)
    _ix = list(range(len(_a["ids"])))[:_sub] if _sub else list(range(len(_a["ids"])))
    _kw = dict(Y_past=_a["Y_past"][_ix], X_past=_a["X_past"][_ix], X_future=_a["X_future"][_ix], Y_future=_a["Y_future"][_ix])
    _pfn = _axt2.make_tabpfn_predict(loaded["tabpfn"], ids=[_a["ids"][i] for i in _ix])
    _tag = f"{_tn}/w0" + (f"x{_sub}" if _sub else "")
    for _s in _seeds:
        try:
            _rr = _rp2("tabpfn_ts", _tag, **_kw, predict=_pfn, seed=_s).to_dict()
            _tab_rows.append(_rr)
            print(f"{_tag} seed={_s}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
        except Exception as e:
            print(f"{_tag} seed={_s}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
# window tasks
for _tn in ["epf_np", "solar_with_weather_1H", "uci_air_quality_1H", "proenfo_gfc14"]:
    _tt = [t for t in mini_cov_tasks if t.task_name == _tn][0]
    _wins = []
    for _wi in range(_tt.num_windows):
        _a = _wt2(_tt.get_window(_wi), max_context=512)
        _wins.append({"Y_past": _a["Y_past"][0], "X_past": _a["X_past"][0],
                      "X_future": _a["X_future"][0], "Y_future": _a["Y_future"][0]})
    for _s in [0, 1]:
        try:
            _rr = _rq2("tabpfn_ts", f"{_tn}/xwin", _wins, predict=_tp, seed=_s).to_dict()
            _tab_rows.append(_rr)
            print(f"{_tn} seed={_s}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
        except Exception as e:
            print(f"{_tn} seed={_s}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
with open("/marimo/sweep_TABPFN.jsonl", "w") as _f:
    for _r in _tab_rows:
        _f.write(_jt2.dumps(_r) + "\\n")
print(f"banked {len(_tab_rows)} rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(TABPFN_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
