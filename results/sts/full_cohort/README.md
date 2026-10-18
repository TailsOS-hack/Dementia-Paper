# Full 5,450-image six-candidate benchmark

Recomputed from all six archived prediction CSVs, with exact source-manifest and prediction SHA256 checks, frozen model identity checks, and independent agreement with the archived confusion matrices and metrics.

| Candidate | Condition | Seed | Correct / images | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- |
| candidate_1 | stage_only | 20261011 | 5438 / 5450 | 99.7798% | 99.7678% | 99.7691% |
| candidate_2 | multitask | 20261011 | 5413 / 5450 | 99.3211% | 99.2706% | 99.2884% |
| candidate_3 | stage_only | 20261012 | 5443 / 5450 | 99.8716% | 99.8660% | 99.8653% |
| candidate_4 | multitask | 20261012 | 5435 / 5450 | 99.7248% | 99.7120% | 99.7088% |
| candidate_5 | stage_only | 20261013 | 5444 / 5450 | 99.8899% | 99.8837% | 99.8844% |
| candidate_6 | multitask | 20261013 | 5428 / 5450 | 99.5963% | 99.5768% | 99.5755% |

The denominator is 5,450 images in 4,273 prepared similarity groups: 1,342 MildDemented, 1,538 ModerateDemented, 1,324 NonDemented and 1,246 VeryMildDemented.

- These six runs are the first six-candidate experiment archived under kevin_paper_20261011; they are not the earlier single historical baseline or later round-2 retraining.
- All 5,450 images belong to the frozen candidate test split. The clean full-split replay was post hoc and contains the historical primary 1,058-image comparison; it is not an additional independent cohort or a retrospectively prespecified primary analysis.
- The 5,450-image split contains 3,860 historical-baseline training and 532 validation rows. The old baseline must not be represented as independently held out on this full cohort.
- The legacy source was augmented upstream. Participant and original augmentation-family identifiers are unavailable. Similarity-group separation does not establish patient independence.
- The four supplied image labels do not establish a clinically validated CDR-stage mapping. These are descriptive image-level results, not clinical diagnostic validation.
- No model was trained, selected, calibrated or tuned during this extraction. Reported point estimates are recomputed from preserved CSV probabilities and labels; no new confidence intervals are claimed.

## Files

- `metrics.json`: complete metrics, frozen identities, provenance hashes and limitations.
- `performance.csv`: six-model performance table.
- `per_class.csv`: class support, recall, precision, specificity and F1.
- `confusion_matrices.csv`: true classes in rows, predicted classes in columns.

Regenerate from the repository root: `python3 -m analysis.sts_results.full_cohort`.
