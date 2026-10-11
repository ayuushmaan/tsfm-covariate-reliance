"""Round 2: lightning (toto), uni2ts --no-deps + gluonts, tabpfn upgrade."""
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


print("lightning rc:", pip(["lightning"]))
print("toto.model.toto:", imp("toto.model.toto"))
print("toto.inference.forecaster:", imp("toto.inference.forecaster"))
print("--- uni2ts ---")
print("uni2ts rc:", pip(["--no-deps", "uni2ts"]))
print("uni2ts:", imp("uni2ts"))
print("gluonts rc:", pip(["gluonts"]))
print("moirai2 module:", imp("uni2ts.model.moirai2"))
print("--- tabpfn ---")
print("tabpfn upgrade rc:", pip(["-U", "tabpfn-time-series", "tabpfn"]))
print("tabpfn_time_series:", imp("tabpfn_time_series"))
