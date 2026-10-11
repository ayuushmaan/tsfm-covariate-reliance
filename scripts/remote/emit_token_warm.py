"""Inject TABPFN_TOKEN (from local env) into box kernel + warm cache with tiny predict."""
import os
import sys

tok = os.environ.get("TABPFN_TOKEN", "")
if not tok:
    sys.exit("set TABPFN_TOKEN locally first")
code = (
    "import os as _os\n"
    f"_os.environ['TABPFN_TOKEN'] = {tok!r}\n"
    "_os.environ['TABPFN_NO_BROWSER'] = '1'\n"
    "import pandas as _pd\n"
    "_ctx = _pd.DataFrame({'item_id': ['a'] * 40, 'timestamp': _pd.date_range('2020-01-01', periods=40, freq='D'), 'target': range(40), 'cov_0': range(40)})\n"
    "_fut = _pd.DataFrame({'item_id': ['a'] * 6, 'timestamp': _pd.date_range('2020-02-10', periods=6, freq='D'), 'cov_0': range(6)})\n"
    "_o = loaded['tabpfn'].predict_df(context_df=_ctx, future_df=_fut)\n"
    "print('tabpfn warm:', list(_o.columns)[:3], flush=True)\n"
)
print(code)
