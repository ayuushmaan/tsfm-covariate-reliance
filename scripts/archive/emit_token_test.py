"""Read TABPFN_TOKEN from the *local* env and inject it into box os.environ via stdin code."""
import os
import sys

tok = os.environ.get("TABPFN_TOKEN", "")
if not tok:
    sys.exit("set TABPFN_TOKEN locally first")
code = (
    "import os as _os\n"
    f"_os.environ['TABPFN_TOKEN'] = {tok!r}\n"
    "_os.environ['TABPFN_NO_BROWSER'] = '1'\n"
    "import tsfm_audit.adapters_extra as _axt\n"
    "import importlib as _ilt\n"
    "_ilt.reload(_axt)\n"
    "import numpy as _npt\n"
    "_pt = _axt.make_tabpfn_predict(loaded['tabpfn'])\n"
    "_Yt = _npt.random.default_rng(0).normal(size=(2, 64))\n"
    "_Xt = _npt.random.default_rng(1).normal(size=(2, 64, 1))\n"
    "_Xf = _npt.random.default_rng(2).normal(size=(2, 12, 1))\n"
    "_pt_out = _pt(_Yt, _Xt, _Xf)\n"
    "print('tabpfn tiny predict:', _pt_out.shape, flush=True)\n"
)
print(code)
