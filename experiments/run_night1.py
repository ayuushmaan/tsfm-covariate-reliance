"""Night-1 sweep template: fev-bench covariate tasks x models x seeds.

Prep (daytime, no GPU needed for code):
    pip install fev
    python -c "import fev; print(fev.__version__)"
    # fev-bench-mini (20 tasks) first; graduate to full 46 covariate tasks.

Sweep (GPU box):
    python experiments/run_night1.py --models chronos2 tabpfn_ts bolt --seeds 0 1 2 --out results/night1.jsonl

Banking: appends one JSON line per (model, task, seed) so a box death
costs only the missing lines. Re-run with --resume to skip completed keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ["chronos2", "chronos2_synth", "tirex2", "t0", "toto", "timesfm25", "timesfm3", "moirai2", "tabpfn_ts", "bolt"]
