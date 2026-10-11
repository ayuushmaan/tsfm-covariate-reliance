"""Install model packages one at a time to isolate failures (py3.13 wheel gaps)."""
import subprocess
import sys

pkgs = ["tirex-ts", "tfc-t0", "toto-ts", "timesfm", "uni2ts", "tabpfn-time-series"]
mods = ["tirex", "t0", "toto", "timesfm", "uni2ts", "tabpfn_time_series"]
for pkg, mod in zip(pkgs, mods):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                       capture_output=True, text=True, timeout=1200)
    err = (r.stderr or "")
    try:
        __import__(mod)
        print(f"{pkg}: INSTALLED+IMPORTS OK (rc={r.returncode})")
    except Exception as e:
        print(f"{pkg}: rc={r.returncode} import={str(e)[:120]}")
        if r.returncode != 0:
            print("  tail: " + err[-600:].replace("\n", " | "))
