"""Run the fixed gNMZ cell."""
import marimo._code_mode as cm

async with cm.get_context() as ctx:
    ctx.run_cell("HwsB")
print("run queued")
