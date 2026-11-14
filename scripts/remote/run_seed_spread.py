"""Five-seed corruption-draw check on five headline cells (GPU box).

Run on a machine with a GPU and `fev` + `chronos-forecasting` installed:
    python scripts/remote/run_seed_spread.py

Banks one JSON row per (model, task, seed) to /marimo/seedspread.jsonl and
resumes from existing rows. Produces results/seed_spread_5seed.csv after
pulling the bank off the box; bolt is always the covariate-blind control and
must read exactly 0.000.
"""
import json
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from fev import Benchmark

from tsfm_audit.adapters import load_bolt, load_chronos2, make_bolt_predict, make_chronos2_predict
from tsfm_audit.fev_convert import window_to_arrays
from tsfm_audit.harness import run_probe, run_probe_windows

BANK = "/marimo/seedspread.jsonl"
BENCH = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml"
TASKS = [
    ("xwin", "epf_np"),
    ("xwin", "proenfo_gfc12"),
    ("xwin", "uci_air_quality_1D"),
    ("series", "rohlik_sales_1D"),
    ("series", "uk_covid_nation_1D/new"),
]
SEEDS = [0, 1, 2, 3, 4]
MAXCTX, SUB = 512, 64


def log(*args):
    print(*args, flush=True)


def bank(row):
    with open(BANK, "a") as f:
        f.write(json.dumps(row) + "\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
log("device:", device)

done = set()
if os.path.exists(BANK):
    for line in open(BANK):
        try:
            d = json.loads(line)
            done.add((d["model"], d["task"], d["seed"]))
        except Exception:
            pass
log("resume skip", len(done))

tasks = {t.task_name: t for t in Benchmark.from_yaml(BENCH).tasks}
log("tasks loaded:", len(tasks))

models = {}
try:
    models["chronos2"] = (load_chronos2(device_map=device), make_chronos2_predict)
    log("chronos2 loaded")
except Exception:
    log("chronos2 FAILED\n" + traceback.format_exc()[-1200:])
try:
    models["bolt"] = (load_bolt(device_map=device), make_bolt_predict)
    log("bolt loaded")
except Exception:
    log("bolt FAILED\n" + traceback.format_exc()[-1200:])

for kind, name in TASKS:
    if name not in tasks:
        log("MISSING TASK", name)
        continue
    task = tasks[name]
    n_targets = len(getattr(task, "target_columns", ["x"]))
    multi = f"|multi{n_targets}" if n_targets > 1 else ""
    if kind == "series":
        conv = window_to_arrays(task.get_window(0), max_context=MAXCTX)
        idx = list(range(len(conv["ids"])))
        if len(idx) > SUB:
            idx = sorted(random.Random(0).sample(idx, SUB))
        kw = dict(
            Y_past=conv["Y_past"][idx],
            X_past=conv["X_past"][idx],
            X_future=conv["X_future"][idx],
            Y_future=conv["Y_future"][idx],
        )
        meta = dict(
            ids=[conv["ids"][i] for i in idx],
            past_ts=[conv["past_ts"][i] for i in idx],
            fut_ts=[conv["fut_ts"][i] for i in idx],
            col_names=conv["col_names"],
            known_names=conv["known_names"],
            cat_maps=conv["cat_maps"],
        )
        tag = f"{name}/w0{multi}"
        log(f"{tag}: N={len(idx)} C={conv['X_past'].shape[2]} known={conv['n_known']}")
        for seed in SEEDS:
            for model_name, (pipe, make) in models.items():
                if (model_name, tag, seed) in done:
                    continue
                try:
                    row = run_probe(
                        model_name, tag, **kw, predict=make(pipe, **meta), seed=seed
                    ).to_dict()
                    bank(row)
                    log(f"  s={seed} {model_name}: intact={row['mase_intact']:.4f} gap={row['swap_gap']:+.4f}")
                except Exception:
                    log(f"  s={seed} {model_name}: FAIL\n" + traceback.format_exc()[-800:])
    else:
        windows = []
        for w in range(task.num_windows):
            a = window_to_arrays(task.get_window(w), max_context=MAXCTX)
            windows.append(
                dict(
                    Y_past=a["Y_past"][0],
                    X_past=a["X_past"][0],
                    X_future=a["X_future"][0],
                    Y_future=a["Y_future"][0],
                )
            )
        tag = f"{name}/xwin{multi}"
        log(f"{tag}: {len(windows)} windows")
        for seed in SEEDS:
            for model_name, (pipe, make) in models.items():
                if (model_name, tag, seed) in done:
                    continue
                try:
                    row = run_probe_windows(
                        model_name, tag, windows, predict=make(pipe), seed=seed
                    ).to_dict()
                    bank(row)
                    log(f"  s={seed} {model_name}: intact={row['mase_intact']:.4f} gap={row['swap_gap']:+.4f}")
                except Exception:
                    log(f"  s={seed} {model_name}: FAIL\n" + traceback.format_exc()[-800:])
log("ALL DONE")
