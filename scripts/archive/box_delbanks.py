"""Delete stale F5B/F6B banks (old metric) so resume reruns them fresh."""
import os as _os

for _f in ["full_F5B.jsonl", "full_F6B.jsonl"]:
    _p = "/marimo/" + _f
    if _os.path.exists(_p):
        _os.remove(_p)
        print("removed", _f)
    else:
        print("absent", _f)
