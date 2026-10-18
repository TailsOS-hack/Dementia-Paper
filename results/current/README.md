# Current dementia results — October 18, 2026

This is the active dataset scope. Earlier broader snapshots remain in Git history. No model was retrained or selected from these refreshed outcomes.

## Retained clinical comparisons

All eight fixed clinical models/ensembles and seven scenarios were recomputed from saved patient predictions: 112 summaries, 1,456 binary-evaluable prediction records, 26 patients. There are no fresh external clinical patients in this active scope.

| Cohort | Model | Correct / total | Accuracy | AD sensitivity | CN specificity | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| besga_reused_test | round1_c2 | 5/11 | 45.45% | 14.29% | 100.00% | 57.14% |
| besga_reused_test | round1_c4 | 7/11 | 63.64% | 57.14% | 75.00% | 66.07% |
| besga_reused_test | round1_c6 | 4/11 | 36.36% | 14.29% | 75.00% | 44.64% |
| besga_reused_test | clinical_seed20261021 | 10/11 | 90.91% | 100.00% | 75.00% | 87.50% |
| besga_reused_test | clinical_seed20261022 | 10/11 | 90.91% | 100.00% | 75.00% | 87.50% |
| besga_reused_test | clinical_seed20261023 | 9/11 | 81.82% | 85.71% | 75.00% | 80.36% |
| besga_reused_test | round1_clinical_ensemble | 5/11 | 45.45% | 14.29% | 100.00% | 57.14% |
| besga_reused_test | new_clinical_ensemble | 10/11 | 90.91% | 100.00% | 75.00% | 87.50% |
| openneuro_reused_external | round1_c2 | 14/15 | 93.33% | 0.00% | 100.00% | 50.00% |
| openneuro_reused_external | round1_c4 | 7/15 | 46.67% | 100.00% | 42.86% | 71.43% |
| openneuro_reused_external | round1_c6 | 13/15 | 86.67% | 0.00% | 92.86% | 46.43% |
| openneuro_reused_external | clinical_seed20261021 | 14/15 | 93.33% | 0.00% | 100.00% | 50.00% |
| openneuro_reused_external | clinical_seed20261022 | 10/15 | 66.67% | 0.00% | 71.43% | 35.71% |
| openneuro_reused_external | clinical_seed20261023 | 15/15 | 100.00% | 100.00% | 100.00% | 100.00% |
| openneuro_reused_external | round1_clinical_ensemble | 12/15 | 80.00% | 0.00% | 85.71% | 42.86% |
| openneuro_reused_external | new_clinical_ensemble | 14/15 | 93.33% | 0.00% | 100.00% | 50.00% |

The new fixed ensemble predicts CN for all 15 reused OpenNeuro participants and misses the sole AD participant. Its 93.3% accuracy is not evidence of reliable AD detection. Exact intervals and all paired 10,000-draw diagnosis-stratified bootstrap comparisons are in `clinical/analysis.json`; intervals are conditional on these small reused cohorts.

## Mendeley source-alias evaluation

The downloaded 6,400-image release contains only images already present in the legacy collection. All ten frozen stage models were rerun on its 1,279-image test split. These are descriptive image-level reused-data scores, not independent validation.

| Frozen model | Correct / 1,279 | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| alzheimers_classifier | 1279 | 100.00% | 100.00% | 1.0000 |
| kevin_20261011__multitask_seed20261011 | 1278 | 99.92% | 99.94% | 0.9990 |
| kevin_20261011__multitask_seed20261012 | 1279 | 100.00% | 100.00% | 1.0000 |
| kevin_20261011__multitask_seed20261013 | 1278 | 99.92% | 99.94% | 0.9995 |
| kevin_20261011__stage_only_seed20261011 | 1278 | 99.92% | 99.96% | 0.9995 |
| kevin_20261011__stage_only_seed20261012 | 1279 | 100.00% | 100.00% | 1.0000 |
| kevin_20261011__stage_only_seed20261013 | 1278 | 99.92% | 99.94% | 0.9995 |
| kevin_round2_20261011__stage_seed20261011 | 1279 | 100.00% | 100.00% | 1.0000 |
| kevin_round2_20261011__stage_seed20261012 | 1277 | 99.84% | 99.92% | 0.9991 |
| kevin_round2_20261011__stage_seed20261013 | 1279 | 100.00% | 100.00% | 1.0000 |

## Graphs and reproducibility

[Figure index](FIGURES.md) covers every regenerated current plot: 12 clinical plots, 11 stage comparison plots, and 12 raw-history plots. Existing historical stage/attribution evidence remains unchanged where the model and inputs are unchanged. Attribution is exploratory, not validated anatomical localization.

OASIS is not used: approved access is pending. No OASIS results are fabricated. Dataset audit and intake instructions are under `datasets/`.

Run the scripts in `analysis_refresh/`; clinical replay requires the locally retained private saved predictions. Patient-level clinical records are not redistributed. The raw histories were replotted, not retrained.
