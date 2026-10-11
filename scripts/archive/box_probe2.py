"""Introspect Toto.from_pretrained signature + timesfm 2.5 entry point."""
import inspect

print("toto sig:")
import tsfm_audit.adapters_extra  # noqa
from toto.model.toto import Toto
print(inspect.signature(Toto.from_pretrained))
print("timesfm attrs:")
import timesfm
print([a for a in dir(timesfm) if "imes" in a or "2p5" in a or "ORECAST" in a.upper()])
