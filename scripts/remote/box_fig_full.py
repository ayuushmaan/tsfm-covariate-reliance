"""Full-46 figure (mean gap per model + heatmap) + table CSV on Box B."""
import marimo._code_mode as cm

FIGF_CODE = '''import json as _jf
import glob as _gf
from collections import defaultdict as _ddf
import numpy as _npf
_rows = []
for _fp in _gf.glob("/marimo/full_F*.jsonl"):
    for _ln in open(_fp):
        try:
            _r = _jf.loads(_ln)
            if "mase_intact" in _r:
                _rows.append(_r)
        except Exception:
            pass
print(f"rows: {len(_rows)}", flush=True)
_grp = _ddf(list)
for _r in _rows:
    _grp[(_r["task"], _r["model"])].append(_r)
_tasks = sorted({_t for _t, _m in _grp})
_models = ["chronos2", "c2synth", "timesfm3", "timesfm25", "t0", "toto", "tabpfn_ts", "bolt", "tirex", "moirai2"]
print(f"tasks: {len(_tasks)} models: {sorted({_m for _t, _m in _grp})}", flush=True)
import matplotlib.pyplot as _plt
_means = {m: _npf.nanmean([_r["swap_gap"] for (_t, _mm), _rs in _grp.items() if _mm == m for _r in _rs]) for m in _models}
_fig, _ax = _plt.subplots(figsize=(9, 4))
_ax.bar(_models, [_means[m] for m in _models])
_ax.set_ylabel("mean swap gap (MASE, 46 tasks)")
_ax.set_title("Covariate reliance by model (full fev-bench covariate set)")
_ax.tick_params(axis="x", rotation=30, labelsize=8)
_fig.tight_layout()
_fig.savefig("/marimo/figure_full.png", dpi=150)
_H = _npf.full((len(_models), len(_tasks)), _npf.nan)
for _i, _m in enumerate(_models):
    for _j, _t in enumerate(_tasks):
        _rs = _grp.get((_t, _m), [])
        if _rs:
            _H[_i, _j] = _npf.mean([r["swap_gap"] for r in _rs])
_fig2, _ax2 = _plt.subplots(figsize=(16, 5))
_im = _ax2.imshow(_H, aspect="auto", vmin=-0.2, vmax=0.6)
_ax2.set_yticks(range(len(_models)))
_ax2.set_yticklabels(_models, fontsize=8)
_ax2.set_xticks(range(len(_tasks)))
_ax2.set_xticklabels([t.split("/")[0][:14] for t in _tasks], rotation=90, fontsize=6)
_ax2.set_title("Swap gap heatmap (models x 46 tasks)")
_fig2.colorbar(_im, shrink=0.8)
_fig2.tight_layout()
_fig2.savefig("/marimo/heatmap_full.png", dpi=150)
import csv as _csv
with open("/marimo/table_full.csv", "w", newline="") as _f:
    _w = _csv.writer(_f)
    _w.writerow(["task", "model", "n", "intact", "swapped", "swap_gap"])
    for (_t, _m), _rs in sorted(_grp.items()):
        _w.writerow([_t, _m, len(_rs), round(float(_npf.mean([r["mase_intact"] for r in _rs])), 4),
                     round(float(_npf.mean([r["mase_swapped"] for r in _rs])), 4),
                     round(float(_npf.mean([r["swap_gap"] for r in _rs])), 4)])
print("saved figure_full.png heatmap_full.png table_full.csv")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(FIGF_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
