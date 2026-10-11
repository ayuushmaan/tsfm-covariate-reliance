# Do Time-Series Foundation Models Read Their Covariates?

**A benchmark-wide, model-agnostic usage audit across 10 open checkpoints and 46 covariate tasks.**

> Draft v0.1 (workshop target: 4 pages + refs; ICLR 2027 workshop / arXiv-first). Numbers below are measured on fev-bench via the repo harness (`src/tsfm_audit/`); banks: `sweep_*.jsonl`, `full_F*.jsonl`, `numbers_capture.json` on the GPU boxes; figures `figure1.png`, `figure_full.png`, `heatmap_full.png`.

---

## Abstract

Time-series foundation models (TSFMs) increasingly accept covariates, and benchmarks report their accuracy *with covariates present* — but never check whether the model *uses* them. We introduce **covariate reliance** (swap gap): the accuracy lost when each series' covariates are replaced, at evaluation time only, with another series' (or another window's) covariates. Marginals are preserved; only target–covariate alignment is broken. Across **10 open checkpoints** (Chronos-2 + synthetic variant, TimesFM 3.0/2.5, t0, Toto 1.0, TabPFN-TS, TiRex, Moirai 2.0, Chronos-Bolt) and **46 fev-bench covariate tasks**, we find: (1) reliance spans **two orders of magnitude by task** (energy-price tasks up to +1.5 MASE, retail tasks ~0.00); (2) **accuracy decouples from reliance** — TimesFM-3 wins rossmann/solar yet shows ~zero gap there, and is actively *harmed* by covariates on rohlik (0.92 vs 0.79 univariate); (3) three covariate-blind controls sit at **exactly zero in 70+ cells**, calibrating the instrument; (4) a linear reference is systematically *exceeded* 2–22×, so covariate effects are overwhelmingly nonlinear. Inference-only, no pretraining, fully reproducible.

---

## 1. Introduction (0.75 page)

Real-world forecasts run on context: promotions move retail demand, load and wind forecasts set electricity prices, weather drives solar output. Modern TSFMs advertise exactly this — Chronos-2 (group attention over targets and covariates), TimesFM-3 (native past-only and past–future covariates), t0 (alternating time/variate attention), TabPFN-TS (known-future covariates as tabular features). And fev-bench, the first general benchmark with serious covariate coverage (46/100 tasks), shows covariates drive state-of-the-art results.

But there is a gap between *accepting* covariates and *reading* them. Benchmarks score models with covariates present; a model that silently ignores them keeps its score. The fev-bench evaluation itself hints at the problem: only a minority of pretrained models use covariates at all. We make the check explicit, cheap, and model-agnostic.

**Contributions.** (1) The swap-gap probe: intact vs cross-series-swapped vs time-permuted covariates, multi-seed, plus a cross-window variant for single-series tasks. (2) A planted-signal dose-response that bounds the probe's sensitivity. (3) A 10-model × 46-task audit with three independent zero-controls. (4) Evidence that accuracy ≠ reliance, with concrete winners on both sides.

---

## 2. Probe definition (0.75 page)

**Swap gap.** For a task with target history `y` and covariates `X` (past + known-future stacked):

- *intact:* `ŷ = model(y, X_own)`
- *swapped:* `ŷ = model(y, X_other)` — each series (derangement) or window receives another's full covariate trajectory. Marginals identical, alignment destroyed.
- *time-permuted:* `ŷ = model(y, permute_time(X_own))` — keeps each series' marginals, breaks alignment.

`reliance = MASE(swapped) − MASE(intact)` (swap gap; time-perm gap reported alongside). Three swap seeds; means banked per (model, task, seed).

**Why swap beats with-vs-without.** Dropping covariates changes input dimensionality and marginals; a gap there confounds *use* with robustness-to-distribution-shift. Swapping isolates alignment — the signature of reading.

**Cross-window variant.** Single-series tasks (energy, solar, air quality) have no siblings to swap with. fev-bench gives them up to 20 evaluation windows; we swap trajectories *across windows*. Caveat (stated in the paper): windows come from different periods, so marginals match only approximately — the planted experiment bounds this concern.

**Planted-signal dose-response.** `y_t = 0.6·y_{t−1} + β·x_{t−3} + noise`, β ∈ [0, 1]. A model that reads covariates must show a rising gap in β; a blind model stays flat. This validates the instrument before touching real data.

**Ceiling.** Per-series direct multi-output ridge (own lags + covariate lags incl. known-future values, per-horizon alpha selection) fit on history, applied at the cutoff; `capture = (bolt − model)/(bolt − ceiling)`. Honest methods notes: shared-alpha CV collapses on multi-horizon targets (far-horizon noise dominates selection); known-future covariates must enter the design at *target* time; categorical codes are excluded from linear designs.

**Metric hygiene.** MASE returns NaN for unmeasurable series (seasonal-naive scale ~0, e.g. intermittent all-zero sales); task means use nanmean. Without this, one degenerate series (m5 weekly: MASE ~2×10⁷) dominates a whole task. This changed m5/favorita-transactions numbers and nothing else.

**Controls.** Chronos-Bolt (univariate), TiRex (no covariate support), Moirai 2.0 (support explicitly dropped) — all must read exactly 0.00.

---

## 3. Results (1.5 pages)

### 3.1 The instrument works (planted, β = 1.0, N = 8)

| model | gap | | model | gap |
|---|---|---|---|---|
| Chronos-2 | +1.8 | | Chronos-2-synth | +1.93 |
| TimesFM-3 | +1.89 | | Toto 1.0 | +0.27/+0.35 |
| naive / Bolt / TiRex / Moirai-2 | 0.000 | | t0 | −0.10 (nondeterministic — see §3.4) |
| TimesFM-2.5 (XReg) | −0.03 | | TabPFN-TS | −0.05 |

Strong readers rise with β (Chronos-2 dose-response verified β = 0 → 1); blind models are flat at 0.000. t0 returns different answers on identical inputs (verified), so it is averaged over seeds everywhere below.

### 3.2 Reliance spans two orders of magnitude by task (Figure 1)

Mean swap gap (MASE), best intact in brackets:

| task | top readers | ~zero |
|---|---|---|
| epf (5× energy price) | TabPFN **+1.47**, c2synth +0.98, c2 +0.83, t0 +0.77 (c2 intact 1.35★) | — |
| proenfo (gridBattery) | c2 +0.58, c2synth +0.52, tfm3 +0.43 (tfm3 0.67★) | toto +0.005 |
| uci air quality | c2 +0.34, c2synth +0.26, tfm3 +0.13 (tfm3 1.62★) | — |
| solar | tfm3 +0.18 (0.91★), c2synth +0.15 | — |
| rohlik grocery | c2 +0.047, c2synth +0.051 (c2 0.61★) | tfm3 **−0.067**, toto −0.018 |
| ukcovid | c2 +0.048, toto +0.055 (toto 1.34★) | tfm25 0.0†, noisy signs at n=4 |
| rossmann (1W+1D) | +0.002…+0.020 (tfm3 0.28/0.21★) | ≈ everything |
| restaurant (static-only) | — | **all models 0** |
| kdd/redset/gvar/m5/entsoe/favorita/walmart/hermes/fred | mixed small (see Table 1) | controls 0 throughout |

† by construction (XReg needs known-future covariates; ukcovid is past-only).

### 3.3 Accuracy decouples from reliance

- **TimesFM-3** wins rossmann (0.278), solar (0.907), uci (1.619) with gaps ≈ 0 — then is *harmed* by covariates on rohlik: 0.924 with vs 0.792 univariate, negative swap gap on all 3 seeds. A leaderboard win that owes nothing to (or despite) covariates.
- **Toto** is best *among covariate-users* on ukcovid (1.36, with reliance +0.06) — yet the outright winner is univariate TiRex (1.04): on this task every covariate pathway loses to ignoring covariates. Contrast rohlik, where Toto is strong (0.69) *without* reliance (−0.02). Same model, opposite stories per task.
- **TabPFN-TS** is the strongest reader on epf (+1.47, best of all 10) yet flat zero on rohlik/rossmann — its accuracy there comes from target history plus auto-generated calendar features, not the provided covariates.
- Static covariates are decorative: on restaurant (static-only), all 10 gaps are 0.000.

### 3.4 Noise discipline

t0 samples (verified nondeterministic on identical inputs); Toto's sample-median jitters (±0.02 across seeds at 32 samples); n=4 tasks (ukcovid weekly) show derangement-lottery sign flips. All reported gaps are seed means; spreads are in Table 1. Controls (Bolt/TiRex/Moirai-2) are bit-identical across seeds at exactly 0.000 in 70+ cells.

### 3.5 The linear ceiling is a floor

capture = (bolt − model)/(bolt − ceiling): epf shows +6…+23 (models beat the linear reference up to 22×), rohlik +1.5…+3, ukcovid mixed. Where the linear model loses to naive (solar/uci/proenfo/rossmann), no linear signal is detectable — yet foundation models still extract small positive gaps there. Covariate effects in the wild are overwhelmingly nonlinear. (The ceiling also needed per-horizon alpha selection and target-time covariate alignment; shared-alpha CV over-shrinks by 2–4×. Details in repo.)

---

## 4. Related work and limitations (1 page)

**Concurrent probes** (cite and contrast): Won (shuffle gain, 3 models, forced engineering systems); Berthelier et al. (planted same-time relations, 2 models); t0 paper (with-vs-without ablation, incl. the +6.3 known / +2.7 past skill finding we replicate in direction). Ours: benchmark-wide (46 tasks), model-agnostic (10 checkpoints), dose-response validated, with three independent zero-controls. Where we overlap (t0 on fev-bench covariates) we agree.

**Limitations.** (1) MASE-only; probabilistic (WQL) reliance needs quantile adapters + full rerun — deferred. (2) Swap is pessimistic where siblings are near-identical (rohlik grocery) and approximate across windows (different periods). (3) Giants subsampled to 64 series; multivariate tasks probed on first target (flagged `|multiN`). (4) t0/Toto sampling noise folded into seed means. (5) Non-commercial weights (TimesFM-3, Moirai) — research use only. (6) TiRex-2 did not exist publicly at writing; TiRex v1 used.

**Hand-off.** Harness, corruptions, and per-model results become §4 of the conference paper unchanged.

---

## References (to verify on write-up)

fev-bench (arXiv 2509.26468); Chronos-2 (2510.15821); TimesFM-3 (research.google, Aug 2026); t0 (arXiv 2609.24559); TabPFN-TS (2501.02945); TiRex (2505.23719); Moirai-2.0 (2511.11698); Toto (Datadog); Won (2610.08118); Berthelier et al. (2605.12200); Pan & Ezzat — dependence terminology (2607.02623).

## Venue
arXiv on completion, then ICLR 2027 workshop (contribution deadline ~Feb 1 2027; notify Feb 26).
