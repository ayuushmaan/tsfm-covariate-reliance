"""Inspect ground-truth format for rohlik window 0."""
import marimo._code_mode as cm

GT_CODE = '''_gt_rohlik = [t for t in mini_cov_tasks if t.task_name == "rohlik_orders_1D"][0]
_gt_w = _gt_rohlik.get_window(0)
gt_ds = _gt_w.get_ground_truth()
print("type:", type(gt_ds))
print("columns:", gt_ds.column_names)
_gt_row0 = gt_ds[0]
for _k, _v in _gt_row0.items():
    import numpy as _np2
    _a2 = _np2.asarray(_v)
    print(f"{_k}: shape={_a2.shape} dtype={_a2.dtype} head={_a2.flatten()[:4]!r}")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(GT_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
