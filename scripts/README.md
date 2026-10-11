# scripts/

- `remote/` — reusable remote-GPU ops: `molab_exec.py` (marimo session connector),
  `box_sweep*.py` (sweep drivers), `box_setup/loadall/*` (env + checkpoints),
  `box_ceiling*/box_fig*` (analysis), `make_ship.py`/`setmode.py` (helpers).
  Needs `MOLAB_URL`/`MOLAB_TOKEN` env vars; never commit tokens.
- `archive/` — one-off session debug scripts, kept for history, not for reuse.
