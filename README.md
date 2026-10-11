# Do Time-Series Foundation Models Read Their Covariates?

**A benchmark-wide, model-agnostic audit of covariate use in time-series foundation models (TSFMs).**

Benchmarks report accuracy *with covariates present* but never check whether the model *uses* them. This repo measures **covariate reliance** (swap gap): accuracy lost when each series' covariates are swapped with another series' covariates at evaluation time.

> Working paper target: ICLR 2027 workshop (submission ~Feb 1, 2027) + arXiv on completion. 4 pages + references. Inference-only — no pretraining.

## Why this matters (hire me pitch)

- **Real gap:** `fev-bench` (100 tasks, 46 with covariates, Sep 2025) shows covariates drive SOTA — Chronos-2 jumps 40.9% → 47.0% skill with covariates — yet most TSFMs ignore them entirely.
- **Cheap, decisive probe:** inference-only corruptions (cross-series swap, time permutation) work on any published checkpoint. No training, one GPU-night.
- **What I built:** a model-agnostic harness + planted-signal dose-response + ceiling regression, reproducible end-to-end.

**Skills demonstrated:** zero-shot TSFM evaluation (Chronos-2, TimesFM-3, Moirai-2, Toto, TabPFN-TS, TiRex), `fev` benchmark integration, robust metrics (MASE/WQL/SQL), experimental design (swap seeds, ablations), Python packaging, figure/table generation for papers.

## Key idea in 30 seconds

```
intact:      y_hat = model(y_past, X_own_past, X_own_future)
swapped:     y_hat = model(y_past, X_other_past, X_other_future)  # same marginals, wrong alignment
time-perm:   y_hat = model(y_past, permute(X_own))                # breaks alignment, keeps marginals

covariate reliance (swap gap) = score(swapped) - score(intact)   # >0 means model used covariates
```

- **Control:** Chronos-Bolt (univariate) should have gap ≈ 0.
- **Ceiling:** per-series ridge of target on lagged covariates = how much covariates *could* contribute.
- **Planted signal:** `y_t = AR Dynamics + β·x_{t-k}`, β ∈ [0,1] → dose-response curve per model: at what β does use appear?

## Repo layout

```
src/tsfm_audit/      # importable library (the hireable artifact)
  probes.py          # swap_across_series, permute_time, truncate_context
  planted.py         # synthetic generator: y = dynamics + β·lagged x
  metrics.py         # MASE, WQL, skill scores
  ceiling.py         # ridge-regression ceiling
  harness.py         # run_probe(): intact vs corrupted, multi-seed banking
  adapters.py        # thin wrappers: chronos2, timesfm, moirai, toto, tabpfn_ts, bolt
experiments/
  run_planted_demo.py  # GPU-free demo: proves harness works (run this first)
  run_night1.py        # full fev-bench sweep (needs checkpoints + GPU)
paper/
  outline.md         # 4-page workshop structure + figures/tables spec
tests/
  test_probes.py     # unit tests for probes + metrics
results/             # banked per-model JSONL (gitignored except demo)
```

## Quickstart (2 min, no GPU, zero downloads)

```bash
pip install -e ".[demo]"
python experiments/run_planted_demo.py --betas 0 0.2 0.4 0.6 0.8 1.0 --n-series 32 --seed 0
# writes results/demo_planted.json + results/demo_figure.png
python -m pytest tests/ -q
```

Expected demo output: a `covariate-blind` baseline has flat dose-response (gap ≈ 0 for all β); a `covariate-using` oracle tracks the ceiling. That contrast *is* Figure 2 of the paper in miniature.

## Smoke test: Chronos-2 vs Bolt (remote GPU only — laptop stays clean)

Local validation first (mocks, no downloads — `chronos`/`fev` are NOT installed here):

```bash
python experiments/smoke_mini.py --dry-run   # expect chronos2(fake) gap >0, bolt(fake) gap ~0
```

Then on your marimo/Modal GPU box (checkpoints + data download THERE, never here):

```bash
# option A — script
python experiments/smoke_mini.py --remote --max-tasks 2 --seeds 0 --out results/smoke_mini.jsonl
# option B — interactive notebook (recommended: planted dose-response through REAL checkpoints + Fig)
marimo edit notebooks/smoke_chronos_bolt_marimo.py
```

`src/tsfm_audit/adapters.py` imports stay light by design — `torch`/`chronos`/`fev`
are imported lazily inside `load_chronos2()`/`load_bolt()`/`run_remote()` only.
A test (`test_import_is_light`) fails the suite if that rule ever breaks.

## Full benchmark (Night 1, GPU box)

```bash
pip install -e ".[full]"
pip install fev chronos-forecasting timesfm moirai-moe  # + toto, tabpfn-ts per adapters.py
python experiments/run_night1.py --models chronos2 tabpfn_ts bolt --benchmark fev-bench --seeds 0 1 2
```

See `paper/outline.md` for the Night-1 / Night-2 schedule and `src/tsfm_audit/adapters.py` for per-model covariate API notes.

## Verified facts (Oct 2026)

| Claim | Status |
|---|---|
| fev-bench: 100 tasks / 46 with covariates, `pip install fev` | ✅ verified — arXiv 2509.26468, github.com/autogluon/fev |
| Chronos-2 (120M): past + known covariates, group attention | ✅ verified — HF `amazon/chronos-2` |
| TimesFM 2.5: covariates via XReg (`timesfm[xreg]`) | ✅ verified — google-research/timesfm |
| TimesFM 3.0 (330M, Aug 2026): native multivariate + covariates | ✅ verified — research.google blog 2026-08-31 |
| ICLR 2027 workshops: submission ~Feb 1, notify Feb 26 | ✅ verified — iclr.cc/Conferences/2027 |
| GIFT-Eval has no covariates → context-probe only or drop | ✅ confirmed by fev-bench paper Table 1 |
| Only Chronos-2 + TabPFN-TS (+ COSMIC) use covariates per fev-bench eval | ✅ verified — fev-bench §6 |

Position vs concurrent work: Won (shuffle gain, 3 models), Berthelier et al. (planted same-time, 2 models), t0 (shift/perm on 1 task) — cite & contrast; we do benchmark-wide + dose-response. Term: **"covariate reliance" / "swap gap"** (not "dependence" — Pan & Ezzat use that for with-vs-without).

## Roadmap to impactful + hireable

1. [x] Harness + planted demo (this week, no GPU) → portfolio-ready repo
2. [ ] fev-bench-mini (20 tasks) smoke test, 1 model → `numbers.tex` draft
3. [ ] Night 1: 6+ models × 46 tasks × 3 cond × 3 seeds → Figure 1 (reliance per model + ceiling)
4. [ ] Night 2: ceiling regression + GIFT-Eval context probe → Table 1
5. [ ] arXiv post immediately, then ICLR 2027 workshop submit

Recruiters / collaborators: the demo + tests run anywhere. The full sweep is a single command. Paper outline maps 1:1 to code outputs.

## License

MIT — benchmarks/models keep their own licenses (TimesFM-3, Moirai weights are non-commercial; fine for research).
