# AAAI 2027 Workshop AI4TS — OpenReview submission fields

Paste-ready values. Venue facts: **7 pages excluding references** (AAAI-27 author kit),
double-blind PDF, non-archival (no proceedings), optional appendix after references.
Deadline on venue site: **Nov 20, 2026, 23:59 AoE** (OpenReview form displays
Nov 21, 4:29 PM local — treat Nov 20 as the real target and submit a day early).

## Title

```
Do Time-Series Foundation Models Read Their Covariates?
```

Alternative (if a subtitle is wanted):

```
Do Time-Series Foundation Models Read Their Covariates? A Benchmark-Wide Usage Audit
```

## Keywords

```
time series foundation models, covariate utilization, evaluation and benchmarking, interpretability, forecasting, robustness, model auditing
```

## TL;DR

```
State-of-the-art time-series foundation models often ignore the covariates they are given; a marginal-preserving swap probe, validated on a planted signal and applied to 10 checkpoints over 46 tasks, measures exactly how much they use.
```

## Abstract

```
Across ten open time-series foundation models (TSFMs) and all 46 covariate tasks of fev-bench, covariate reliance spans two orders of magnitude and is only weakly related to accuracy. We define reliance as the swap gap: the accuracy lost when each series' covariates are replaced, at inference only, by another series' covariates. Marginals are preserved and only target-covariate alignment is broken, so the gap isolates use rather than robustness to distribution shift. Three covariate-blind controls read exactly $0.000$ in over 70 cells, calibrating the instrument, and a planted-signal dose-response, $y_t = 0.6\,y_{t-1} + \beta\,x_{t-3} + \varepsilon$, recovers a monotone rise for every checkpoint that self-declares covariate support. Reliance ranges from $+1.47$ MASE on electricity-price forecasting to $\approx 0.00$ on retail. Accuracy decouples from reliance: TimesFM-3 leads on Rossmann and solar with small gaps, yet is actively harmed by covariates on Rohlik grocery ($0.924$ vs $0.792$ univariate, negative gap on every seed). A per-series linear ridge reference is exceeded by up to $22\times$, implying that covariate effects in the wild are overwhelmingly nonlinear. The audit covers Chronos-2 (and a synthetic variant), TimesFM 3.0/2.5, $t_0$, Toto, TabPFN-TS, TiRex, Moirai 2.0, and Chronos-Bolt. The probe is model-agnostic, inference-only, and reuses any benchmark's native covariates, so reliance can be reported alongside error in future TSFM evaluations.
```

## Form fields

- **License:** `CC BY 4.0` (most reusable; non-archival workshop keeps copyright with you).
- **Email Sharing:** authorize (required to submit).
- **Data Release:** authorize (required to submit).
- **PDF:** must be anonymized — see checklist below.

## Anonymization checklist (double-blind PDF)

- [ ] No author names, affiliations, acknowledgments, or email addresses.
- [ ] No GitHub URL. Replace "details in repo" / "repo history" with "details in the released code".
- [ ] Cite the harness as "our released code"; keep model citations normal.
- [ ] Figures must not carry watermark text with the repo path.
- [ ] PDF metadata: set a neutral `\hypersetup{pdfauthor={},pdftitle={...}}` if using hyperref.

## Figure mapping for the PDF

| Draft slot | Local file |
|---|---|
| Fig 1 reliance by domain | `results/study2_reliance_by_domain.png` |
| Fig 2 feature distraction (Rohlik) | `results/study2_feature_distraction_rohlik.png` |
| Fig 3 domain reliance | `results/study2_domain_reliance.png` |
| Fig 4 accuracy vs reliance decoupling | `results/study3_decoupling_scatter.png` |
| Fig 5 linear ceiling / capture | `results/study3_linear_ceiling_comparison.png` |

Draft mentions `figure1.png`, `figure_full.png`, `heatmap_full.png` — those lived on the
GPU boxes and are gone; use the five above instead and adjust captions.

## Build notes

Files ready in `paper/`: `main.tex`, `references.bib`, `tab_planted.tex`,
`tab_tasks.tex`, `tab_decoupling.tex`, `tab_inventory.tex`, `tab_positioning.tex`.
Anonymized supplementary bundle: `paper/build/supplementary_anon.zip` (43 files,
~89 KB; ignored by git). Upload it in the OpenReview supplementary field.

No local LaTeX (`pdflatex`, `tectonic`, `pandoc` all missing). Two routes:
1. **Overleaf (recommended):** new project from the AAAI-27 author kit (brings
   `aaai2027.sty` / `aaai2027.bst`), upload `main.tex`, the six `tab_*.tex`, the
   `.bib`, and the five PNGs from `results/`, compile, download PDF.
2. **Local:** `winget install TectonicProject.Tectonic`, then `tectonic main.tex`.

Before submitting: check every figure caption against the rendered PNG (the captions
were written from filenames, not from viewing the images), and confirm the PDF has no
author metadata.
