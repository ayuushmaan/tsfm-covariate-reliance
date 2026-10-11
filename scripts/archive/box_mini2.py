"""Read gNMZ cell, then replace its code with the fixed version (same defs)."""
import marimo._code_mode as cm

NEW_CODE = '''import importlib
import tsfm_audit.adapters as _a
importlib.reload(_a)
from tsfm_audit.adapters import FEV_MINI_YAML
from fev import Benchmark

mini_bench = Benchmark.from_yaml(FEV_MINI_YAML)
print(f"mini tasks: {len(mini_bench.tasks)}")
mini_cov_tasks = [
    t for t in mini_bench.tasks
    if getattr(t, "known_dynamic_columns", []) or getattr(t, "past_dynamic_columns", []) or getattr(t, "static_columns", [])
]
print(f"with covariates: {len(mini_cov_tasks)}")
for _t in mini_cov_tasks:
    print(f"- {_t.task_name} H={_t.horizon} known={_t.known_dynamic_columns} past={_t.past_dynamic_columns} static={_t.static_columns}")
'''

async with cm.get_context() as ctx:
    cell = None
    for c in ctx.cells:
        if c.id == "gNMZ":
            cell = c
            break
    print("found:", cell.id if cell else None)
    print("old code:", repr((cell.code or "")[:200]))
    ctx.edit_cell("gNMZ", NEW_CODE)
print("edit queued")
