"""Inspect fev-bench-mini: covariate tasks + first window schema (via cm cell)."""
import marimo._code_mode as cm

INSPECT_CODE = '''import importlib
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

SCHEMA_CODE = '''_t0 = mini_cov_tasks[0]
_w0 = _t0.get_window(0)
_past, _future = _w0.get_input_data()
print("task:", _t0.task_name, "| target:", _t0.target)
print("past type:", type(_past))
print(_past)
print("future type:", type(_future))
print(_future)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(INSPECT_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
