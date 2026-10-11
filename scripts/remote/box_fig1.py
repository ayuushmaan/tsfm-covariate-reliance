"""Figure 1 (swap gap per model per task) + Table 1 CSV from all banks."""
import marimo._code_mode as cm

FIG_CODE = '''import json as _jf, glob as _gf
from collections import defaultdict as _ddf
import numpy as _npf
_rows = []
for _fp in _gf.glob("/marimo/sweep_*.jsonl"):
    for _ln in open(_fp):
        try:
            _r = _jf.loads(_ln)
            if "mase_intact" in _r:
                _rows.append(_r)
        except Exception:
            pass
_grp = _ddf(list)
for _r in _rows:
    _grp[(_r["task"], _r["model"])].append(_r)
_tasks = ["epf_np/xwin", "proenfo_gfc14/xwin", "uci_air_quality_1H/xwin", "solar_with_weather_1H/xwin",
          "rohlik_orders_1D/w0", "uk_covid_nation_1D/cumulative/w0", "rossmann_1W/w0x32", "rossmann_1D/w0x32"]
_models = ["chronos2", "c2synth", "timesfm3", "timesfm25", "t0", "toto", "tabpfn_ts", "bolt", "tirex", "moirai2"]
import matplotlib.pyplot as _plt
_fig, _axs = _plt.subplots(2, 4, figsize=(14, 7), sharey=False)
for _ax, _t in zip(_axs.flat, _tasks):
    _g = [_npf.mean([r["swap_gap"] for r in _grp.get((_t, _m), [])]) if _grp.get((_t, _m)) else float("nan") for _m in _models]
    _ax.bar(range(len(_models)), _g)
    _ax.set_title(_t.split("/")[0][:18], fontsize=9)
    _ax.set_xticks(range(len(_models)))
    _ax.set_xticklabels(["c2", "c2s", "tfm3", "tfm25", "t0", "toto", "tab", "bolt", "tir", "moi"], rotation=45, fontsize=7)
    _ax.set_ylabel("swap gap (MASE)")
    _ax.axhline(0, color="k", lw=0.5)
_fig.suptitle("Covariate reliance (swap gap) per model per task — higher = reads covariates more")
_fig.tight_layout()
_fig.savefig("/marimo/figure1.png", dpi=150)
print("saved figure1.png")
import csv as _csv
with open("/marimo/table1.csv", "w", newline="") as _f:
    _w = _csv.writer(_f)
    _w.writerow(["task", "model", "n", "intact", "swapped", "swap_gap"])
    for (_t, _m), _rs in sorted(_grp.items()):
        _w.writerow([_t, _m, len(_rs), round(float(_npf.mean([r["mase_intact"] for r in _rs])), 4),
                     round(float(_npf.mean([r["mase_swapped"] for r in _rs])), 4),
                     round(float(_npf.mean([r["swap_gap"] for r in _rs])), 4)])
print("saved table1.csv")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(FIG_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
