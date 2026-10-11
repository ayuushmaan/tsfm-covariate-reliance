"""Full zoo install, hard lessons baked in: gluonts 0.14.3 FIRST (--no-deps),
then everything else; toto/uni2ts --no-deps (their numpy pins can't build)."""
import subprocess
import sys


def pip(args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args,
                       capture_output=True, text=True, timeout=2400)
    return r.returncode


steps = [
    ("gluonts014", ["--no-deps", "gluonts==0.14.3"]),
    ("base", ["chronos-forecasting", "fev", "tirex-ts", "tfc-t0", "lightning"]),
    ("toto", ["--no-deps", "toto-ts"]),
    ("timesfm", ["timesfm[torch]"]),
    ("uni2ts", ["--no-deps", "uni2ts"]),
    ("tabpfn", ["tabpfn-time-series", "tabpfn"]),
    ("repin-gluonts", ["--no-deps", "--force-reinstall", "gluonts==0.14.3"]),
]
for name, args in steps:
    print(f"--- {name} ---", flush=True)
    print("rc:", pip(args), flush=True)

import importlib

for m in ["chronos", "fev", "tirex", "t0", "toto", "timesfm", "timesfm3",
          "uni2ts", "gluonts", "tabpfn_time_series", "lightning"]:
    try:
        importlib.import_module(m)
        extra = ""
        if m == "gluonts":
            import gluonts
            extra = gluonts.__version__
        print(f"{m}: OK {extra}")
    except Exception as e:
        print(f"{m}: FAIL {str(e)[:140]}")
