# New retraining learning curves

All six new October 18 runs completed and passed report validation: 329 total epochs. This experiment is separate from the historical six-run study.

Checkpoint verification: **all_12_binaries_verified** (12/12 available binaries hash-verified). All histories, per-epoch evaluation denominators, stopping decisions and validation-only selections were checked. Missing binaries are not claimed as verified. No weights were loaded or inference run for this report.

| Run | Completed epochs | Selected epoch | Clean train NLL | Clean validation NLL |
| --- | ---: | ---: | ---: | ---: |
| stage_only_seed20261011 | 42 | 27 | 0.034431 | 0.036154 |
| multitask_seed20261011 | 67 | 52 | 0.041157 | 0.045978 |
| stage_only_seed20261012 | 55 | 40 | 0.035196 | 0.037724 |
| multitask_seed20261012 | 75 | 60 | 0.042489 | 0.048810 |
| stage_only_seed20261013 | 40 | 25 | 0.035889 | 0.038450 |
| multitask_seed20261013 | 50 | 35 | 0.044689 | 0.049141 |

`matched_clean_nll.png/.pdf` plots unsmoothed raw clean training and validation NLL for all six runs. Each pair uses the same epoch-end weights, transform, evaluation mode and unweighted unsmoothed NLL definition; dashed lines identify validation-selected epochs. `online_and_objectives.png/.pdf` separately labels online augmented NLL and weighted/smoothed objectives. `clinical_clean_nll.png/.pdf` compares 32 training and 11 validation subjects after averaging slice probabilities per subject. Clinical loss does not select checkpoints.

`summary.csv` contains selected/final NLL and epoch counts; `epochs.csv` retains every logged observation; `condition_summary.csv` reports three-seed means and sample SDs (not confidence intervals). `report.json` records source hashes, split identity checks, per-checkpoint verification and definitions. Its `figure_sha256` map pins all six generated PNG/PDF files; a `--no-plots` build records an empty map and cannot supply figures for the manuscript or public package, even if older plot files remain.

Stage split verification: `manifest_ids_and_groups_verified`. Clinical split verification: `manifest_ids_and_groups_verified`. Manifest checks do not re-read image pixels or establish patient independence.

Rebuild from the repository root:

```sh
python -m research.sts_retrain_20261018.report --input /path/to/retrieved/sts_retrain_20261018 --output results/sts/retraining
```

Add `--require-checkpoints` to require all 12 checkpoint binaries. Add `--clinical-manifest /path/to/audited/minimized.csv` to verify clinical membership/group separation against the recorded digest. The bundled public stage manifest is verified automatically when available; `--stage-manifest` can specify its path.

Trainer identity is pinned to the local `train.py` by default. For an archived rebuild with a different reviewed trainer version, pass its independently verified digest using `--expected-trainer-sha256`; never copy an unverified digest solely from the suite metadata.

Training is resubstitution and validation was used for selection. These curves contain no held-out performance evidence and do not establish clinical superiority or readiness. No smoothing, interpolation or manual adjustment was applied.
