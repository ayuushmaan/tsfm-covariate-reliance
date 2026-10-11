"""Per-series MASE breakdown to find the NaN source."""
import marimo._code_mode as cm

BREAK_CODE = '''import numpy as _np6
import tsfm_audit.harness as _h6
importlib.reload(_h6)
from tsfm_audit.harness import run_probe as _rp6
from tsfm_audit.metrics import mase as _m6
import tsfm_audit.adapters as _a6
importlib.reload(_a6)
from tsfm_audit.adapters import make_bolt_predict as _mk_b6

_b6 = _mk_b6(bolt_pipe, ids=fev_arrays["ids"], past_ts=fev_arrays["past_ts"], fut_ts=fev_arrays["fut_ts"])
_pb6 = _b6(fev_arrays["Y_past"], fev_arrays["X_past"], fev_arrays["X_future"])
print("pred nan:", _np6.isnan(_pb6).sum(), "pred shape:", _pb6.shape)
for _i in range(7):
    _v = (~_np6.isnan(fev_arrays["Y_future"][_i])).sum()
    print(f"series {_i} ({fev_arrays['ids'][_i]}): valid_truth={_v}/61 mase={_m6(fev_arrays['Y_future'][_i], _pb6[_i], fev_arrays['Y_past'][_i])}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(BREAK_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
