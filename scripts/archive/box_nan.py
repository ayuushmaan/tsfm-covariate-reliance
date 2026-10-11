"""Diagnose NaN source: arrays + one raw Chronos-2 call."""
import marimo._code_mode as cm

NAN_CODE = '''import numpy as _np3
print("nan Yp:", _np3.isnan(fev_arrays["Y_past"]).sum(), "Xp:", _np3.isnan(fev_arrays["X_past"]).sum(),
      "Xf-known:", _np3.isnan(fev_arrays["X_future"][:, :, :4]).sum(), "Yf:", _np3.isnan(fev_arrays["Y_future"]).sum())
print("inf Xp:", _np3.isinf(fev_arrays["X_past"]).sum())
for _ci, _cc in enumerate(fev_arrays["covs"]):
    print(f"  {_cc}: nan_past={_np3.isnan(fev_arrays['X_past'][:, :, _ci]).sum()}")
print("Yf head:", fev_arrays["Y_future"][0][:5])
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(NAN_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
