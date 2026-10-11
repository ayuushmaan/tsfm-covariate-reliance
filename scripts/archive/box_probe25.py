"""Function identity + all gluonts locations on sys.path."""
import pathlib
import sys

for d in sys.path:
    try:
        hits = sorted(pathlib.Path(d).glob("gluonts"))
        if hits:
            print(d, "->", [h.name for h in hits])
    except Exception:
        pass
import gluonts.torch.model.predictor as _pp
import gluonts.model.forecast_generator as _fg
print("same fn:", _pp.predict_to_numpy is _fg.predict_to_numpy)
print("predictor file:", _pp.__file__)
print("fg file:", _fg.__file__)
print("registry now:", list(_pp.predict_to_numpy.registry.keys()))
