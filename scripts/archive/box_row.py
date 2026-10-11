"""Inspect one row's values (lengths, dtypes) to pin the converter."""
import marimo._code_mode as cm

ROW_CODE = '''_row = schema_past[0]
for _k, _v in _row.items():
    import numpy as _np
    _a = _np.asarray(_v)
    print(f"{_k}: shape={_a.shape} dtype={_a.dtype} head={_a.flatten()[:4]!r}")
_schema_frow = schema_future[0]
for _k, _v in _schema_frow.items():
    import numpy as _np
    _a = _np.asarray(_v)
    print(f"FUT {_k}: shape={_a.shape} dtype={_a.dtype} head={_a.flatten()[:4]!r}")
print("n_past_rows:", len(schema_past), "| n_fut_rows:", len(schema_future))
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(ROW_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
print("queued")
