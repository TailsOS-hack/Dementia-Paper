# Post-hoc analysis excluding Harvard — October 18, 2026

This descriptive subset excludes Harvard after its poor performance was observed. No independent data-quality reason for exclusion has been established. It does not replace the completed experiment or its original external evaluation. The Harvard raw MRI and participant records were already absent from this public repository; none were deleted locally by this analysis.

The recomputation uses saved aggregate confusion matrices for every one of the eight fixed models/ensembles in the two remaining clinical cohorts. No models were retrained, predictions rerun, thresholds adjusted or new patients evaluated. All 16 model/cohort count-based summaries match the previously reported results.

| Remaining cohort | Old fixed ensemble | New fixed ensemble | Old → new balanced accuracy |
|---|---:|---:|---:|
| Reused Besga test, 11 patients | 5/11 (45.5%) | 10/11 (90.9%) | 57.1% → 87.5% |
| Reused OpenNeuro, 15 patients | 12/15 (80.0%) | 14/15 (93.3%) | 42.9% → 50.0% |

The new ensemble still misses the only AD participant in OpenNeuro and predicts control for all 15 people: AD sensitivity is 0%, despite 93.3% accuracy. The 26 remaining participants are from reused cohorts. **There are zero fresh external clinical patients remaining in this subset.** Removing the fresh cohort cannot establish improved generalization. Cohorts are not pooled into a headline accuracy, and the separate four-class image benchmark is unchanged.

The excluded Harvard result remains 9/19 for the new ensemble versus 10/19 for the older ensemble, with 0% new-ensemble specificity. See the [original full results](../../docs/kevin_paper_20261011/round2_publication/RESULTS.md). Its exclusion is a post-hoc presentation choice, not evidence that its observations were invalid.

Run `python3 analyses/harvard_excluded_20261018/recompute.py` from the repository root. `recomputed_metrics.csv` includes all eight models in both retained cohorts. `verification.json` records the input SHA-256, checks and limitations. Probability-based metrics, confidence intervals and patient-paired inference were not recomputed from aggregate counts; their original evidence remains in the full report.
