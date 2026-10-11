# Workshop paper outline (4 pages + refs) — maps 1:1 to code outputs

**Title:** Do Time-Series Foundation Models Read Their Covariates?
**Term:** covariate reliance / swap gap. **Venue:** ICLR 2027 workshop (~Feb 1 submit) + arXiv immediately on done.

## 1. Introduction (0.75p)
- TSFMs now accept covariates (Chronos-2, TimesFM-3, TabPFN-TS…) but benchmarks report with-covariates accuracy without checking use.
- fev-bench: 46/100 tasks have covariates; covariates drive SOTA (Chronos-2 40.9→47.0 skill) yet most models ignore them.
- Contribution: benchmark-wide, model-agnostic usage audit + planted dose-response + ceiling.

## 2. Probe definition (0.75p)
- Swap gap: `score(swap X across series) − score(intact)`; time-perm variant; 3 swap seeds.
- Why swap > with-vs-without: keeps marginals, isolates alignment; catches models that silently ignore X.
- Context probe (if time): full vs last-season truncation on GIFT-Eval.
- Ceiling: per-series ridge on lagged X.

## 3. Results (1.5p)
- **Fig 1:** reliance per model + ceiling mark (from `results/night1.jsonl`).
- **Fig 2:** planted dose-response β∈[0,1] per model (from `run_planted_demo.py` → real checkpoints).
- **Table 1:** per-task intact/corrupted MASE+WQL.
- Report replication/agreement with Won / Berthelier / t0 where overlapping.

## 4. Related work + Limitations (1p)
- Concurrent probes (cite & contrast scope: 3 models / 2 models / 1 task vs our benchmark-wide).
- Limits: swap is pessimistic if models rely on cross-series priors; fev leakage policy; non-commercial weights (TimesFM-3, Moirai) research-only.
- Hand-off: harness + corruptions become §4 of conference paper unchanged.

## Night schedule (from PLAN_WORKSHOP.md)
- **Prep daytime:** `pip install fev`, download checkpoints to HF vault, test harness on fev-bench-mini (20 tasks) × 1 model.
- **Night 1:** 6+ models × 46 tasks × {intact, swap, timeperm} × 3 seeds → bank per model; planted grid on 2nd box.
- **Between:** `numbers.tex` + Fig 1 + Table 1 draft.
- **Night 2:** ceiling + GIFT-Eval context probe + write.
