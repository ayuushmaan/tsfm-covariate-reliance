"""Install remaining model packages on the GPU box (long pole)."""
import subprocess
import sys

pkgs = ["tirex-ts", "tfc-t0", "toto-ts", "timesfm[torch]", "uni2ts", "tabpfn-time-series"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                   capture_output=True, text=True, timeout=2400)
print("rc:", r.returncode)
tail = (r.stderr or "")[-1500:]
print(tail if tail else "pip quiet OK")
import importlib

for m in ["tirex", "t0", "toto", "timesfm3", "uni2ts", "tabpfn_time_series", "timesfm"]:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL {str(e)[:160]}")
