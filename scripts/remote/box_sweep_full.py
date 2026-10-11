"""Night-1 FULL sweep driver. MODE picks family chunk; per-row banking + resume."""
import marimo._code_mode as cm

MODE = "F9"

# (kind, tasks, seeds, max_context, subsample, bank)
CHUNKS = {
    "F1": ("xwin", ["proenfo_gfc12", "proenfo_gfc17"], [0, 1], 512, None, "full_F1.jsonl"),
    "F2": ("series", ["rohlik_sales_1D", "rohlik_sales_1W", "rohlik_orders_1W",
                       "uk_covid_nation_1D/new", "uk_covid_nation_1W/new",
                       "uk_covid_nation_1W/cumulative"], [0, 1], 512, 64, "full_F2.jsonl"),
    "F3": ("xwin", ["entsoe_15T", "entsoe_30T", "entsoe_1H", "epf_be", "epf_de", "epf_fr",
                    "epf_pjm"], [0, 1], 512, None, "full_F3.jsonl"),
    "F4": ("series", ["restaurant", "hermes", "walmart"], [0, 1], 512, 64, "full_F4.jsonl"),
    "F5": ("series", ["m5_1D", "m5_1W", "m5_1M"], [0, 1], 512, 64, "full_F5.jsonl"),
    "F6": ("series", ["favorita_stores_1D", "favorita_stores_1W", "favorita_stores_1M",
                       "favorita_transactions_1D", "favorita_transactions_1W",
                       "favorita_transactions_1M"], [0, 1], 512, 64, "full_F6.jsonl"),
    "F7": ("xwin", ["solar_with_weather_15T", "uci_air_quality_1D"], [0, 1], 512, None, "full_F7.jsonl"),
    "F8": ("series", ["kdd_cup_2022_10T", "kdd_cup_2022_30T", "kdd_cup_2022_1D",
                       "redset_5T", "redset_15T", "redset_1H", "gvar"], [0, 1], 512, 64, "full_F8.jsonl"),
    "F9": ("xwin", ["fred_md_2025/cee", "fred_qd_2025/cee"], [0, 1], 512, None, "full_F9.jsonl"),
    "F5B": ("series", ["m5_1W", "m5_1M"], [0, 1], 512, 64, "full_F5B.jsonl"),
    "F6B": ("series", ["favorita_transactions_1D", "favorita_transactions_1W",
                       "favorita_transactions_1M"], [0, 1], 512, None, "full_F6B.jsonl"),
    "F0a": ("series", ["rohlik_orders_1D", "rossmann_1W", "rossmann_1D",
                       "uk_covid_nation_1D/cumulative"], [0, 1, 2], 512, 32, "full_F0a.jsonl"),
    "F0b": ("xwin", ["epf_np", "solar_with_weather_1H", "uci_air_quality_1H",
                     "proenfo_gfc14"], [0, 1], 512, None, "full_F0b.jsonl"),
}

GEN = '''import tsfm_audit.adapters as _ad
import tsfm_audit.adapters_extra as _axe
import tsfm_audit.fev_convert as _fc
import tsfm_audit.harness as _hh
import tsfm_audit.metrics as _mt
import tsfm_audit.ceiling as _cl
import importlib as _il
for _mm in (_ad, _axe, _fc, _hh, _mt, _cl):
    _il.reload(_mm)
from tsfm_audit.adapters import make_bolt_predict as _mk_bolt, make_chronos2_predict as _mk_c2
from tsfm_audit.fev_convert import window_to_arrays as _w
from tsfm_audit.harness import run_probe as _rp, run_probe_windows as _rpw
import json as _js
import os as _os

_kind, _tasks, _seeds, _mc, _sub, _bank = _CHUNK2
_bankp = "/marimo/" + _bank
_done = set()
from fev import Benchmark as _B9
_FT = {t.task_name: t for t in _B9.from_yaml("https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml").tasks}
if _os.path.exists(_bankp):
    for _ln in open(_bankp):
        try:
            _d = _js.loads(_ln)
            _done.add((_d["model"], _d["task"], _d["seed"]))
        except Exception:
            pass
print(f"resume skip {len(_done)} rows", flush=True)

def _bank_row(_r):
    with open(_bankp, "a") as _f:
        _f.write(_js.dumps(_r) + "\\n")

def _build(nk, meta):
    _W = {}
    if loaded.get("tirex") is not None: _W["tirex"] = _axe.make_tirex_predict(loaded["tirex"])
    if loaded.get("moirai2") is not None: _W["moirai2"] = _axe.make_moirai2_predict(loaded["moirai2"])
    if loaded.get("t0") is not None: _W["t0"] = _axe.make_t0_predict(loaded["t0"], nk)
    if loaded.get("toto") is not None: _W["toto"] = _axe.make_toto_predict(loaded["toto"], num_samples=32)
    if loaded.get("timesfm3") is not None: _W["timesfm3"] = _axe.make_timesfm3_predict(loaded["timesfm3"], nk)
    if loaded.get("timesfm25") is not None: _W["timesfm25"] = _axe.make_timesfm25_predict(loaded["timesfm25"], nk)
    if loaded.get("tabpfn") is not None: _W["tabpfn_ts"] = _axe.make_tabpfn_predict(loaded["tabpfn"], ids=meta.get("ids"))
    _W["chronos2"] = _mk_c2(chronos2_pipe, **meta)
    _W["bolt"] = _mk_bolt(bolt_pipe, **meta)
    if loaded.get("c2synth") is not None: _W["c2synth"] = _mk_c2(loaded["c2synth"], **meta)
    return _W

for _tname in _tasks:
    _task = _FT[_tname]
    _nt = len(getattr(_task, "target_columns", ["x"]))
    _multi = f"|multi{_nt}" if _nt > 1 else ""
    if _kind == "series":
        _conv = _w(_task.get_window(0), max_context=_mc)
        _nk = _conv["n_known"]
        _idx = list(range(len(_conv["ids"])))
        if _sub and len(_idx) > _sub:
            _idx = sorted(__import__("random").Random(0).sample(_idx, _sub))
        _kw = dict(Y_past=_conv["Y_past"][_idx], X_past=_conv["X_past"][_idx],
                   X_future=_conv["X_future"][_idx], Y_future=_conv["Y_future"][_idx])
        _meta = dict(ids=[_conv["ids"][i] for i in _idx], past_ts=[_conv["past_ts"][i] for i in _idx],
                     fut_ts=[_conv["fut_ts"][i] for i in _idx], col_names=_conv["col_names"],
                     known_names=_conv["known_names"], cat_maps=_conv["cat_maps"])
        _W = _build(_nk, _meta)
        _tag = f"{_tname}/w0{_multi}"
        print(f"{_tag}: N={len(_idx)} C={_conv['X_past'].shape[2]} known={_nk}", flush=True)
        for _s in _seeds:
            for _mn, _mp in _W.items():
                if (_mn, _tag, _s) in _done:
                    continue
                try:
                    _rr = _rp(_mn, _tag, **_kw, predict=_mp, seed=_s).to_dict()
                    _bank_row(_rr)
                    print(f"  s={_s} {_mn}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
                except Exception as e:
                    print(f"  s={_s} {_mn}: FAIL {type(e).__name__}: {str(e)[:150]}", flush=True)
    else:
        _c0 = _w(_task.get_window(0), max_context=_mc)
        _nk = _c0["n_known"]
        _wins = []
        for _wi in range(_task.num_windows):
            _a = _w(_task.get_window(_wi), max_context=_mc)
            _wins.append({"Y_past": _a["Y_past"][0], "X_past": _a["X_past"][0],
                          "X_future": _a["X_future"][0], "Y_future": _a["Y_future"][0]})
        _W = _build(_nk, {})
        _tag = f"{_tname}/xwin{_multi}"
        print(f"{_tag}: {len(_wins)} windows", flush=True)
        for _s in _seeds:
            for _mn, _mp in _W.items():
                if (_mn, _tag, _s) in _done:
                    continue
                try:
                    _rr = _rpw(_mn, _tag, _wins, predict=_mp, seed=_s).to_dict()
                    _bank_row(_rr)
                    print(f"  s={_s} {_mn}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
                except Exception as e:
                    print(f"  s={_s} {_mn}: FAIL {type(e).__name__}: {str(e)[:150]}", flush=True)
print("chunk done")
'''

async with cm.get_context() as ctx:
    kind, tasks, seeds, mc, sub, bank = CHUNKS[MODE]
    code = f"_CHUNK2 = ({kind!r}, {tasks!r}, {seeds!r}, {mc!r}, {sub!r}, {bank!r})\n" + GEN
    cid = ctx.create_cell(code, hide_code=False)
    print("created", cid, "for", MODE)
    ctx.run_cell(cid)
print("queued")
