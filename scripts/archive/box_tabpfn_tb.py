"""Full tabpfn import traceback to judge shim feasibility."""
import importlib
import traceback

try:
    importlib.import_module("tabpfn_time_series")
    print("tabpfn: OK already")
except Exception:
    traceback.print_exc(limit=30)
