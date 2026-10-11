"""Toto patch fix (pass cls) + gluonts pin + Moirai2Module direct-call inspect."""
import subprocess
import sys
import traceback


def pip(args):
    r = subprocess.run([sys.executable, "-m", "pip"] + args,
                       capture_output=True, text=True, timeout=1200)
    return r.returncode, (r.stdout or "")[-400:] + (r.stderr or "")[-400:]


print("=== toto (cls fix) ===")
try:
    from toto.model.toto import Toto

    _orig = Toto._from_pretrained.__func__

    @classmethod
    def _patched(cls, *, proxies=None, resume_download=False, **kw):
        return _orig(cls, proxies=proxies, resume_download=resume_download, **kw)

    Toto._from_pretrained = _patched
    _toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to("cuda")
    from toto.inference.forecaster import TotoForecaster
    loaded["toto"] = TotoForecaster(_toto.model)
    print("toto: LOADED", flush=True)
except Exception:
    loaded["toto"] = None
    traceback.print_exc(limit=10)

print("=== gluonts pin ===")
rc, out = pip(["install", "--force-reinstall", "--no-deps", "-q", "gluonts==0.14.3"])
print("rc:", rc, out.replace("\n", " | ")[:300])
import gluonts
print("gluonts version now:", gluonts.__version__)

print("=== moirai module direct API ===")
import inspect
from uni2ts.model.moirai2 import Moirai2Module
print([m for m in dir(Moirai2Module) if not m.startswith("_")][:20])
