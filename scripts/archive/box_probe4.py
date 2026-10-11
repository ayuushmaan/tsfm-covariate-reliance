"""Inspect Toto.from_pretrained forwarding to _from_pretrained."""
import inspect

from toto.model.toto import Toto

print(inspect.getsource(Toto.from_pretrained))
print("----")
print(inspect.signature(Toto._from_pretrained))
