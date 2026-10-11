# Scoop / verification check (2026-10-11, supersedes [verify] items in PLAN_WORKSHOP.md)

- **fev-bench:** 100 tasks / 96 datasets / 7 domains, 46 with covariates (30 known-dynamic, 24 past-dynamic, 19 static, non-exclusive). `pip install fev`, Apache-2.0, github.com/autogluon/fev. Source: arXiv 2509.26468.
- **fev-bench eval finding:** only Chronos-2 + TabPFN-TS (+COSMIC) use covariates; Chronos-2 47.0% vs 40.9% skill with/without; TabPFN-TS 42.5% vs 34.1%. Most pretrained models ignore covariates.
- **Chronos-2:** 120M, encoder-only, group attention, past-only + known (real/categorical) covariates, ctx 8192. HF `amazon/chronos-2`.
- **TimesFM 2.5:** covariates via XReg (`timesfm[xreg]`, `forecast_with_covariates`, modes `xreg+timesfm` / `timesfm+xreg`).
- **TimesFM 3.0:** 330M, Aug 2026, native multivariate + past-only & past-future covariates, single-pass horizon. SOTA on Gift-Eval/FEV/Time in multivariate mode.
- **GIFT-Eval:** 0 covariate tasks (fev-bench paper Table 1) → context-probe only or drop.
- **ICLR 2027 workshops:** proposal deadline Oct 9 2026 (closed), workshop list out Nov 29 2026, suggested contribution deadline Feb 1 2027, mandatory notify + OpenReview public Feb 26 2027. Post to arXiv as soon as done.
