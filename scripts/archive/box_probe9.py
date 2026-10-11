"""Toto pristine-reload + patch; gluonts uninstall-all then pin 0.14.3."""
import importlib
import subprocess
import sys
import traceback


def pip(args):
    r = subprocess.run([sys.executable, "-m", "pip"] + args,
                       capture_output=True, text=True, timeout=1200)
    return r.returncode


print("sys.path:", sys.path)
print("=== toto pristine ===")
try:
    import toto.model.toto as _tm
    importlib.reload(_tm)
    from toto.model.toto import Toto as _T2

    _orig2 = _T2._from_pretrained.__func__

    @classmethod
    def _patched2(cls, *, proxies=None, resume_download=False, **kw):
        return _orig2(cls, proxies=proxies, resume_download=resume_download, **kw)

    _T2._from_pretrained = _patched2
    _toto2 = _T2.from_pretrained("Datadog/Toto-Open-Base-1.0").to("cuda")
    from toto.inference.forecaster import TotoForecaster
    loaded["toto"] = TotoForecaster(_toto2.model)
    print("toto: LOADED", flush=True)
except Exception:
    loaded["toto"] = None
    traceback.print_exc(limit=8)

print("=== gluonts reset ===")
print("uninstall rc:", pip(["uninstall", "-y", "gluonts"]))
print("install rc:", pip(["install", "-q", "--no-deps", "gluonts==0.14.3"]))
import gluonts
print("gluonts version now:", gluonts.__file__, gluonts.__version__)
