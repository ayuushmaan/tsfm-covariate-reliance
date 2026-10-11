"""Bank fev_rows to JSONL on the box; drop dead scratch cells."""
import marimo._code_mode as cm

BANK_CODE = '''import json as _js8
with open("/marimo/smoke_rohlik.jsonl", "w") as _f8:
    for _r in fev_rows:
        _f8.write(_js8.dumps(_r) + "\\n")
print("banked", len(fev_rows), "rows")
'''

async with cm.get_context() as ctx:
    cid = ctx.create_cell(BANK_CODE, hide_code=False)
    print("created", cid)
    ctx.run_cell(cid)
    for dead in ("nzqo", "TOJb", "PsoV"):
        try:
            ctx.delete_cell(dead)
            print("deleted", dead)
        except Exception as e:
            print(dead, "kept:", str(e)[:100])
print("queued")
