"""Reship package, then probe rohlik window-0 with both real models (one cell)."""
import marimo._code_mode as cm

PROBE_CODE = '''import importlib as _il2
import tsfm_audit.adapters as _a2
importlib.reload(_a2)
from tsfm_audit.adapters import make_bolt_predict as _mk_bolt2, make_chronos2_predict as _mk_c2_2
from tsfm_audit.fev_convert import window_to_arrays as _w2a2
from tsfm_audit.harness import run_probe as _rp2

_rohlik = [t for t in mini_cov_tasks if t.task_name == "rohlik_orders_1D"][0]
fev_arrays = _w2a2(_rohlik.get_window(0), max_context=512)
print("shapes:", fev_arrays["Y_past"].shape, fev_arrays["X_past"].shape, fev_arrays["X_future"].shape, fev_arrays["Y_future"].shape)
print("covs:", fev_arrays["covs"], "| n_known:", fev_arrays["n_known"])
_fev_c2 = _mk_c2_2(chronos2_pipe, ids=fev_arrays["ids"], past_ts=fev_arrays["past_ts"], fut_ts=fev_arrays["fut_ts"])
_fev_bolt = _mk_bolt2(bolt_pipe, ids=fev_arrays["ids"], past_ts=fev_arrays["past_ts"], fut_ts=fev_arrays["fut_ts"])
_fev_kw = dict(Y_past=fev_arrays["Y_past"], X_past=fev_arrays["X_past"], X_future=fev_arrays["X_future"], Y_future=fev_arrays["Y_future"])
fev_rows = []
fev_rows.append(_rp2("chronos2", "rohlik_orders_1D/w0", **_fev_kw, predict=_fev_c2, seed=0).to_dict())
fev_rows.append(_rp2("bolt", "rohlik_orders_1D/w0", **_fev_kw, predict=_fev_bolt, seed=0).to_dict())
for _r in fev_rows:
    print(f"{_r['model']}: intact={_r['mase_intact']:.4f} swapped={_r['mase_swapped']:.4f} swap_gap={_r['swap_gap']:+.4f}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(PROBE_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
