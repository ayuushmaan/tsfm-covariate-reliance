"""TabPFN backfill: rossmann_1D w0x32, same subsample as sweep E."""
import marimo._code_mode as cm

BACKFILL_CODE = '''import tsfm_audit.adapters_extra as _axb
import importlib as _ilb
_ilb.reload(_axb)
from tsfm_audit.fev_convert import window_to_arrays as _wb
from tsfm_audit.harness import run_probe as _rpb
import json as _jb
_t = [t for t in mini_cov_tasks if t.task_name == "rossmann_1D"][0]
_a = _wb(_t.get_window(0), max_context=512)
_ix = sorted(__import__("random").Random(0).sample(list(range(len(_a["ids"]))), 32))
_kw = dict(Y_past=_a["Y_past"][_ix], X_past=_a["X_past"][_ix], X_future=_a["X_future"][_ix], Y_future=_a["Y_future"][_ix])
_pfn = _axb.make_tabpfn_predict(loaded["tabpfn"], ids=[_a["ids"][i] for i in _ix])
with open("/marimo/sweep_E.jsonl", "a") as _f:
    for _s in [0, 1]:
        _rr = _rpb("tabpfn_ts", "rossmann_1D/w0x32", **_kw, predict=_pfn, seed=_s).to_dict()
        _f.write(_jb.dumps(_rr) + "\\n")
        print(f"seed={_s}: intact={_rr['mase_intact']:.4f} gap={_rr['swap_gap']:+.4f}", flush=True)
print("backfilled")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(BACKFILL_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
