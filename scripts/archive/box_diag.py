"""Diagnose model NaNs: channel stds + raw single-series predict."""
import marimo._code_mode as cm

DIAG_CODE = '''import numpy as _np4
for _ci, _cc in enumerate(fev_arrays["covs"]):
    _v = fev_arrays["X_past"][:, :, _ci]
    print(f"{_cc}: std={_np4.std(_v):.4g} min={_v.min():.4g} max={_v.max():.4g}")
print("Y std per series:", [_np4.std(fev_arrays["Y_past"][i]) for i in range(7)])
from tsfm_audit.adapters import arrays_to_long_df as _a2df
_cdf, _fdf, _kc = _a2df(fev_arrays["Y_past"][:1], fev_arrays["X_past"][:1], fev_arrays["X_future"][:1],
                        [fev_arrays["ids"][0]], [fev_arrays["past_ts"][0]], [fev_arrays["fut_ts"][0]])
print("ctx rows:", len(_cdf), "fut rows:", len(_fdf), "known:", _kc)
_raw = chronos2_pipe.predict_df(_cdf, future_df=_fdf, prediction_length=61, id_column="id", timestamp_column="timestamp", target="target", batch_size=64)
print(_raw[["predictions"]].describe())
print("raw nan:", _raw["predictions"].isna().sum())
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(DIAG_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
