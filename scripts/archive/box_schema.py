"""Dump first-window schema of epf_np (known-covariate task)."""
import marimo._code_mode as cm

SCHEMA_CODE = '''_schema_task = [t for t in mini_cov_tasks if t.task_name == "epf_np"][0]
_schema_window = _schema_task.get_window(0)
schema_past, schema_future = _schema_window.get_input_data()
print("task:", _schema_task.task_name, "| target:", _schema_task.target, "| horizon:", _schema_task.horizon)
print("=== PAST ===")
print("type:", type(schema_past))
print(schema_past)
print("=== FUTURE ===")
print("type:", type(schema_future))
print(schema_future)
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(SCHEMA_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
