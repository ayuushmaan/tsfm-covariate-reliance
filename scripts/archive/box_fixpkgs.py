"""Workarounds: --no-deps for toto/uni2ts, timesfm3 check, tabpfn version probe."""
import importlib
import subprocess
import sys


def pip(args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args,
                       capture_output=True, text=True, timeout=1200)
    return r.returncode


def imp(mod):
    try:
        importlib.import_module(mod)
        return "OK"
    except Exception as e:
        return f"FAIL {str(e)[:150]}"


import pandas, numpy
print("pandas", pandas.__version__, "| numpy", numpy.__version__)
print("timesfm3:", imp("timesfm3"))

print("--- toto-ts --no-deps ---")
print("rc:", pip(["--no-deps", "toto-ts"]))
print("toto:", imp("toto"))
print("toto.model.toto:", imp("toto.model.toto"))
