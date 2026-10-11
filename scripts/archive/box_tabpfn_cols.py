"""Print TabPFN pred_df schema (env already on box; token persists in kernel)."""
import pandas as _pd

_ctx = _pd.DataFrame({"item_id": ["a"] * 40, "timestamp": _pd.date_range("2020-01-01", periods=40, freq="D"),
                      "target": range(40), "cov_0": range(40)})
_fut = _pd.DataFrame({"item_id": ["a"] * 6, "timestamp": _pd.date_range("2020-02-10", periods=6, freq="D"),
                      "cov_0": range(6)})
_out = loaded["tabpfn"].predict_df(context_df=_ctx, future_df=_fut)
print("columns:", list(_out.columns))
print(_out.head(3).to_string())
