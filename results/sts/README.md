# STS manuscript results and reproducible evidence

[Expanded Results](manuscript/RESULTS.md) · [Word](manuscript/STS_Expanded_Results.docx) · [PDF](manuscript/STS_Expanded_Results.pdf) · [Loss curve audit](training/LOSS_AUDIT.md)

The four-page manuscript contains three tables, two graphs and commit-pinned links to the original figures. It now includes verified numerical results for the complete 5,450-image split, recovered from all six saved prediction files. All original result artifacts remain unchanged.

## What the study supports

- The 305 training epochs belong to **two MobileNetV3-Large conditions × three seeds**, six runs total, archived under the October 11 Round 1 study. They are separate from the older specialist and later Round 2 runs.
- The complete 5,450-image replay reports six-model accuracy from **99.32% to 99.89%**. Its frozen protocol explicitly calls it a **post-hoc supplementary analysis**, not a replacement for the prespecified 1,058-image primary comparison.
- The primary 1,058 images are nested within the full test split; the 756-image sensitivity analysis is nested within the primary. These are not independent cohorts.
- All 27 model/cohort rows for the matched, sensitivity and reused-100 benchmarks reproduce from saved predictions. The stage ensemble scores 1,058/1,058 and 756/756; these observations do not establish population certainty.
- The old baseline saw 3,860 training and 532 validation rows within the full 5,450. Its fairer exact-row comparison remains the 1,058 intersection, subject to the recorded similarity-exposure limits.

These image-level legacy results do not establish patient independence, clinically validated severity labels, clinical superiority, or clinical readiness. No model was selected on test outcomes. The source artifacts and actual clinical endpoints remain distinct.

## Evidence files

| Directory | Contents |
| --- | --- |
| `training/` | Six-run table, checkpoint/selection audit, raw epoch figures and separate legacy-history lineage |
| `full_cohort/` | Full-test accuracy, balanced accuracy, macro F1, per-class metrics, confusion matrices and source hashes |
| `supporting/` | Recomputed point metrics, paired discordant counts, preserved historical intervals, condition summaries and comparison plot |
| `manuscript/` | Editable Word, PDF and Markdown Results section |

Historical uncertainty intervals were preserved from the frozen analysis, not rerun or relabeled. Point estimates, class counts, confusion matrices, ensemble averaging and paired discordances were independently checked. The full-test summary has no invented paired interval.

## Reproduce the historical analysis

From the repository root, with the dependencies in `requirements.txt` installed:

```bash
python -m analysis.sts_results.run
python -m pytest tests/test_sts_full_cohort.py tests/test_sts_training.py tests/test_sts_supporting.py tests/test_training_curve_schema.py
python -m analysis.sts_results.build_manuscript --source-commit 84c1d731154d12103abf1adce5a57ad6f7bb6b8a
```

The manuscript builder needs `python-docx`; plots need `matplotlib`. Render the Word document to PDF with LibreOffice and inspect all four pages before delivering a regenerated file. The build verifies every supporting-table numerator against the recomputed audit. Source links are pinned to the audited public commit; new report files are linked locally.

No model binaries or raw clinical MRI are needed to reproduce these historical tables. The training audit cross-checks checkpoint hashes in the frozen identity records rather than loading weight files. Public-export history hashes and historical original-file hashes are identified separately.

After finalizing public code and generated files, refresh the repository inventory with `python -m analysis.sts_results.refresh_repository_manifest --write`, then verify it with `--check`. The default command only previews changes. LFS entries use the referenced payload SHA256 and size; raw/private working bundles are excluded.

## Corrected loss plots and new training

The [current training plots](../current/training/index.json) now separate different endpoints and units. Their historical values were not changed. The [loss audit](training/LOSS_AUDIT.md) explains the online augmented loss, clean validation NLL, clinical slice-versus-subject distinction and optimization objective.

[The new training implementation](../../research/sts_retrain_20261018/README.md) records clean full-training and validation NLL at the same epoch-end weights, with identical deterministic transforms and aggregation. Its runs and curves are a new experiment and must not be substituted into the historical 305-epoch results.
