"""Set MODE in a sweep driver. Usage: python scripts/remote/setmode.py <MODE> [driver.py]"""
import pathlib
import re
import sys

name = sys.argv[2] if len(sys.argv) > 2 else "box_sweep.py"
p = pathlib.Path(__file__).resolve().parent / name
s = p.read_text()
s2, n = re.subn(r'^MODE = "[A-Za-z0-9]+"', f'MODE = "{sys.argv[1]}"', s, count=1, flags=re.M)
assert n == 1, "MODE line not found"
p.write_text(s2)
print(f"{name} MODE =", sys.argv[1])
