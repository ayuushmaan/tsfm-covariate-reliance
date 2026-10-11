# Workshop paper plan: two all-nighters

**Working title:** Do Time-Series Foundation Models Read Their Covariates?

**Format:** 4 pages plus references, workshop track. Candidate venues: an ICLR 2027 or ICML 2027 workshop on time series, foundation models or synthetic data [verify which exist and their deadlines; ICLR workshop deadlines usually fall in early February].

**Scope rule:** inference only. No pretraining, no new generator. Everything runs on published checkpoints and public benchmarks, so two nights of compute and writing are enough.

## The claim

Time-series foundation models increasingly accept covariates and multiple channels, but benchmarks report accuracy with the covariates present and never check whether the model uses them. We measure **covariate dependence**: the accuracy lost when each series' covariates are swapped with another series' covariates at evaluation (the time-series analogue of the foreign-key corruption probe in `../new_paper`). Expected finding: several models lose almost nothing, so their covariate-aware accuracy comes from the target's own history.

Fallback if every model uses covariates well: the paper reports which models do and how much, plus the planted-signal test below, which shows how strong a covariate effect must be before each model picks it up. Either outcome is a result.

## Models (open checkpoints) [verify availability and covariate support]

| model | why |
|---|---|
| Chronos-2 | built for covariates and multivariate input (group attention) |
| Moirai (1.x and 2.0) | any-variate attention over channels |
| Toto | multivariate, observability data |
| TimesFM 2.x | covariate support through in-context regression |
| TabPFN-TS | tabular prior applied to series; covariates as features |
| Chronos-Bolt | univariate control: covariate dependence should be zero |

## Data

- **fev-bench** tasks with covariates [verify]: the main benchmark, because its tasks are chosen to have informative covariates.
- **GIFT-Eval** multivariate subset: a second benchmark, no covariate guarantee.
- **Planted-signal series** (synthetic, generated on the fly): target = own dynamics + β · lagged covariate, β on a grid from 0 to 1. Gives a dose-response curve per model: the β at which covariate use appears.

## Measurements

1. **Covariate dependence:** score intact minus score with covariates swapped across series (also with covariates permuted in time, which keeps marginals and breaks alignment). Metrics: MASE and WQL as the benchmarks define them.
2. **Context dependence (second probe if time allows):** score with full context minus score with context truncated to the last season.
3. **Ceiling:** a simple regression of the target on lagged covariates, fitted per series, to show how much the covariates could contribute on each task. A model whose covariate dependence sits far below this ceiling is leaving signal unused.

## Schedule

**Daytime before night 1 (prep, no box needed):**
- literature and scoop check, 2 to 3 hours: has anyone published this probe? [run first; it can kill or reshape the paper]
- install the model packages in one environment, download checkpoints and benchmark data to the HF vault
- write the probe harness: load task → predict intact → predict with covariates swapped and time-permuted → score

**Night 1 (compute):**
- 20:00 to 22:00: harness test on one model and two tasks; fix loaders
- 22:00 to 06:00: run all six models on fev-bench covariate tasks, intact and two corruptions, three random swap seeds; results bank to the vault per model
- in parallel on a second box: planted-signal grid, all models

**Day between nights:** pull results, generate `numbers.tex`, draft one figure (covariate dependence per model per task) and one table.

**Night 2 (compute and writing):**
- GIFT-Eval multivariate subset, the ceiling regression, context probe if time allows
- write the paper: introduction (0.75 page), probe definition (0.75), results (1.5), related work and limitations (1)

**After night 2:** one review pass, citation verification, submit.

## Figures and tables

- Figure 1: covariate dependence per model, with the ceiling marked (the one-picture pitch).
- Figure 2: planted-signal dose-response, one line per model.
- Table 1: per-task intact and corrupted scores.

## Compute

Inference only: roughly 6 models × ~30 tasks × 3 conditions × 3 seeds. One GPU box per night is enough; a second box halves the risk of a box death costing the night.

## Risks

| risk | what we do |
|---|---|
| already published | fold the probe into the conference plan as a section; drop the workshop paper |
| a model's covariate API differs per model | the harness wraps each; budget the daytime prep for this |
| box dies overnight | results bank per model; rerun only the missing model |
| benchmark has too few covariate tasks | lean on the planted-signal experiment |

## Hand-off to the conference paper

The probe harness, the corruption definitions and the per-model results become Section 4 of the conference paper (published models), unchanged.

## Revisions after the scoop check (2026-10-10; supersedes the [verify] items above)

See `SCOOP_CHECK.md` for sources.

- **Positioning:** a benchmark-wide, model-agnostic usage audit. Concurrent work to cite and contrast: Won (arXiv 2610.08118, shuffle gain on 3 models in forced engineering systems), Berthelier et al. (arXiv 2605.12200, planted same-time relations, 2 models), t0 (arXiv 2609.24559, time shift and permutation on one task). If we agree with Won that TimesFM-2.5 and TabPFN-TS ignore covariates, report it as replication at benchmark scale.
- **Metric name:** "covariate reliance" (or "swap gap"); "covariate dependence" is already used by Pan and Ezzat (arXiv 2607.02623) for with-vs-without.
- **Models:** Chronos-2 (and its synthetic-only variant, a free extra data point), TiRex-2, t0, Toto 1.0 (exogenous support), TimesFM 2.5 (XReg) and 3.0, Moirai 2.0, TabPFN-TS (future covariates only), Chronos-Bolt as the univariate control. Licenses: TimesFM 3.0 and Moirai weights are non-commercial, fine for research.
- **Benchmark:** fev-bench, 46 covariate tasks (`pip install fev`); fev-bench-mini (20 tasks) for the night-1 harness test. GIFT-Eval has no covariates: use it only for the context probe, or drop it.
- **Venue:** an ICLR 2027 workshop (papers about Feb 1, 2027; the list is out Nov 29). Because the area is moving fast, **post to arXiv as soon as the paper is done** and submit to the workshop later.
- **Timing:** run the two nights within the next two weeks.
