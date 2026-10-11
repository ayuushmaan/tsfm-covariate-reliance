"""Night-1 sweep driver. MODE picks the chunk; per-row banking + resume by key."""
import marimo._code_mode as cm

MODE = "E"  # A1 rohlik s012 | A2 ukcovid s012 | B rossmann32 s01 | C1 epf | C2 solar | D1 uci | D2 proenfo (reruns resume: only missing keys run)

CHUNKS = {
    "A1": ("series", "rohlik_orders_1D", [0, 1, 2], 512, None, "sweep_A1.jsonl"),
    "A2": ("series", "uk_covid_nation_1D/cumulative", [0, 1, 2], 512, None, "sweep_A2.jsonl"),
    "B": ("series", "rossmann_1W", [0, 1], 512, 32, "sweep_B.jsonl"),
    "C1": ("xwin", "epf_np", [0, 1], 512, None, "sweep_C1.jsonl"),
    "C2": ("xwin", "solar_with_weather_1H", [0, 1], 512, None, "sweep_C2.jsonl"),
    "D1": ("xwin", "uci_air_quality_1H", [0, 1], 512, None, "sweep_D1.jsonl"),
    "D2": ("xwin", "proenfo_gfc14", [0, 1], 512, None, "sweep_D2.jsonl"),
    "E": ("series", "rossmann_1D", [0, 1], 512, 32, "sweep_E.jsonl"),
}

GEN = '''import tsfm_audit.adapters as _ad
import tsfm_audit.adapters_extra as _axe
import tsfm_audit.fev_convert as _fc
import tsfm_audit.harness as _hh
import importlib as _il
for _mm in (_ad, _axe, _fc, _hh):
    _il.reload(_mm)
from tsfm_audit.adapters import make_bolt_predict as _mk_bolt, make_chronos2_predict as _mk_c2
from tsfm_audit.fev_convert import window_to_arrays as _w
from tsfm_audit.harness import run_probe as _rp, run_probe_windows as _rpw
import json as _js
import os as _os

_kind, _tname, _seeds, _mc, _sub, _bank = _CHUNK
_bankp = "/marimo/" + _bank
_done = set()
if _os.path.exists(_bankp):
    for _ln in open(_bankp):
        try:
            _d = _js.loads(_ln)
            _done.add((_d["model"], _d["task"], _d["seed"]))
        except Exception:
            pass
print(f"chunk {_tname}: resume skip {len(_done)} rows", flush=True)
_task = [t for t in mini_cov_tasks if t.task_name == _tname][0]

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
    _W["chronos2"] = _mk_c2(chronos2_pipe, **meta)
    _W["bolt"] = _mk_bolt(bolt_pipe, **meta)
    if loaded.get("c2synth") is not None: _W["c2synth"] = _mk_c2(loaded["c2synth"], **meta)
    return _W

if _kind == "series":
    _conv = _w(_task.get_window(0), max_context=_mc)
    _nk = _conv["n_known"]
    _idx = list(range(len(_conv["ids"])))[:_sub] if _sub else list(range(len(_conv["ids"])))
    _kw = dict(Y_past=_conv["Y_past"][_idx], X_past=_conv["X_past"][_idx],
               X_future=_conv["X_future"][_idx], Y_future=_conv["Y_future"][_idx])
    _meta = dict(ids=[_conv["ids"][i] for i in _idx], past_ts=[_conv["past_ts"][i] for i in _idx],
                 fut_ts=[_conv["fut_ts"][i] for i in _idx], col_names=_conv["col_names"],
                 known_names=_conv["known_names"], cat_maps=_conv["cat_maps"])
    _W = _build(_nk, _meta)
    _tag = f"{_tname}/w0" + (f"x{_sub}" if _sub else "")
    for _s in _seeds:
        for _mn, _mp in _W.items():
            if (_mn, _tag, _s) in _done:
                print(f"  skip {_mn} seed={_s} (banked)")
                continue
            try:
                _rr = _rp(_mn, _tag, **_kw, predict=_mp, seed=_s).to_dict()
                _bank_row(_rr)
                print(f"  seed={_s} {_mn}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
            except Exception as e:
                print(f"  seed={_s} {_mn}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
else:
    _c0 = _w(_task.get_window(0), max_context=_mc)
    _nk = _c0["n_known"]
    _wins = []
    for _wi in range(_task.num_windows):
        _a = _w(_task.get_window(_wi), max_context=_mc)
        _wins.append({"Y_past": _a["Y_past"][0], "X_past": _a["X_past"][0],
                      "X_future": _a["X_future"][0], "Y_future": _a["Y_future"][0]})
    print(f"{_tname}: {len(_wins)} windows", flush=True)
    _W = _build(_nk, {})
    _tag = f"{_tname}/xwin"
    for _s in _seeds:
        for _mn, _mp in _W.items():
            if (_mn, _tag, _s) in _done:
                print(f"  skip {_mn} seed={_s} (banked)")
                continue
            try:
                _rr = _rpw(_mn, _tag, _wins, predict=_mp, seed=_s).to_dict()
                _bank_row(_rr)
                print(f"  seed={_s} {_mn}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
            except Exception as e:
                print(f"  seed={_s} {_mn}: FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
print("chunk done")
'''

async with cm.get_context() as ctx:
    kind, tname, seeds, mc, sub, bank = CHUNKS[MODE]
    code = f"_CHUNK = ({kind!r}, {tname!r}, {seeds!r}, {mc!r}, {sub!r}, {bank!r})\n" + GEN
    cid = ctx.create_cell(code, hide_code=False)
    print("created", cid, "for", MODE)
    ctx.run_cell(cid)
print("queued")
