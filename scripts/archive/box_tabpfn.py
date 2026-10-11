"""TabPFN vs pandas3: full traceback, then guarded downgrade trial with revert."""
import importlib
import subprocess
import sys
import traceback


def pip(args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args,
                       capture_output=True, text=True, timeout=1200)
    return r.returncode


try:
    importlib.import_module("tabpfn_time_series")
    print("tabpfn: OK already")
except Exception:
    traceback.print_exc(limit=6)

print("--- trying pandas<3 (will revert if stack breaks) ---")
print("downgrade rc:", pip(["pandas<3"]))
import pandas
print("pandas now:", pandas.__version__)
try:
    importlib.import_module("tabpfn_time_series")
    print("tabpfn: OK")
    ok_stack = True
except Exception as e:
    print("tabpfn still FAIL:", str(e)[:150])
    ok_stack = False

# verify the working stack survives
checks = {}
try:
    import chronos
    checks["chronos"] = "OK"
except Exception as e:
    checks["chronos"] = f"FAIL {str(e)[:100]}"
try:
    import fev
    checks["fev"] = "OK"
except Exception as e:
    checks["fev"] = f"FAIL {str(e)[:100]}"
print(checks)
if not ok_stack or any(v != "OK" for v in checks.values()):
    print("reverting to pandas==3.0.3, rc:", pip(["pandas==3.0.3"]))
    importlib.reload(pandas)
    print("pandas back:", pandas.__version__)
