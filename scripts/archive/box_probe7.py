"""Toto monkeypatch + gluonts 0.14 --no-deps for Moirai."""
import subprocess
import sys
import traceback


def pip(args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + args,
                       capture_output=True, text=True, timeout=1200)
    return r.returncode


print("=== toto monkeypatch ===")
try:
    from toto.model.toto import Toto

    _orig = Toto._from_pretrained.__func__

    @classmethod
    def _patched(cls, *, proxies=None, resume_download=False, **kw):
        return _orig(proxies=proxies, resume_download=resume_download, **kw)

    Toto._from_pretrained = _patched
    _toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0").to("cuda")
    from toto.inference.forecaster import TotoForecaster
    loaded["toto"] = TotoForecaster(_toto.model)
    print("toto: LOADED via monkeypatch", flush=True)
except Exception:
    loaded["toto"] = None
    traceback.print_exc(limit=12)

print("=== gluonts 0.14 ===")
print("rc:", pip(["--no-deps", "gluonts==0.14.*"]))
try:
    import gluonts
    print("gluonts version:", gluonts.__version__)
    from gluonts.dataset.common import ListDataset
    print("ListDataset OK")
except Exception as e:
    print("gluonts FAIL:", str(e)[:200])
