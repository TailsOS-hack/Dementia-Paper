# Learning-curve audit

The historical Round 1 training losses do not need numerical alteration. Their logging uses valid per-observation denominators, but the former current-results plotting code mixed different endpoints and aggregation units on one axis. The refreshed plots correct that presentation. Separately requested retraining can record matched clean training and validation measurements at every new epoch; those measurements cannot be reconstructed from the old aggregate histories.

## Confirmed plotting fixes

`analysis_refresh/refresh_results.py` previously selected columns by whether their names contained `loss`, `nll` or `accuracy`. For multitask runs this combined four-class image losses, binary clinical slice losses, and binary clinical subject losses. It also omitted the Round 2 weighted/smoothed training objective because its column name lacked `loss` or `nll`.

`analysis_refresh/training_curves.py` now uses four explicit schemas. It separates stage and clinical endpoints, separates Round 1 clinical sampled-slice training from subject-level validation, distinguishes raw NLL from the optimization objective and validation-selection score, and labels stress-family versus stress-scenario aggregation. Round 2 clean clinical training and validation share a panel because both use the same subject-level mean-logit rule at epoch-end weights. Online augmented measurements have a separate panel. Every recorded value is preserved, including genuine fluctuations. Missing epoch-0 training values remain absent. Loss axes start at zero; accuracy axes span 0–1; clinical loss panels share their scale within each run.

The source histories remain unchanged. `results/current/training/index.json` records their SHA-256 hashes and observed epoch ranges; `results/current/training/loss_semantics.json` records each panel's endpoint, unit and source fields. The twelve retained figure filenames remain stable.

## Round 1 loss meanings and arithmetic

Source: `docs/kevin_paper_20261011/training/train_candidate.py` (`train`, `evaluate`, `metrics`, `dataset`, and `forward`) and the six CSV histories under `docs/kevin_paper_20261011/publication/training/`.

| Logged field | Meaning | Aggregation and state |
| --- | --- | --- |
| `train_stage_loss` | Unweighted cross-entropy against unsmoothed four-class labels | Sum of batch mean CE × actual batch size, divided by the observed stage draw count; online predictions before each optimizer update; augmentation, dropout and training-mode batch normalization active |
| `val_stage_loss` | Unweighted four-class NLL | All stage validation images, epoch-end model in evaluation mode, deterministic resize and normalization; probabilities normalized before NLL |
| `train_aux_loss` | Unweighted unsmoothed binary CE | Sampled augmented clinical slices, balanced by diagnosis and sampled by subject; batch-size-weighted numerator divided by observed auxiliary draws; online training mode |
| `val_aux_loss` | Binary subject NLL | One prediction per subject after averaging nine slice probabilities; epoch-end evaluation mode |

The optimized stage objective is the mean of per-image, label-smoothed CE multiplied by inverse group-size/class-group weights normalized to global mean one. Multitask steps add auxiliary label-smoothed CE with weight 0.25 every fourth stage batch. The old raw CE fields are not that weighted/smoothed objective and do not include its auxiliary contribution. The global weight normalization and the sample-weighted raw-loss denominators are internally consistent with the documented objective; the audit did not find a double-division or partial-batch denominator error.

All six histories record 19,023 stage draws each epoch. The three multitask histories record 4,768 auxiliary draws each epoch: 595 stage batches at batch size 32, with 149 auxiliary updates including step zero, times 32 sampled slices. These auxiliary draws repeatedly sample the training subjects and are not independent patients. The three stage-only histories have zero auxiliary draws.

Training mode is explicitly restored at the beginning of each epoch. Validation uses `model.eval()` and inference mode. Both routes apply the same ImageNet normalization, while only training applies random augmentation. Online train loss is accumulated while the model is changing; validation loss uses the epoch's final parameters. A lower validation loss than online training loss can therefore occur without a normalization defect. The shared backbone also receives clinical gradients and batch-normalization updates during auxiliary steps, so multitask stage behavior need not match stage-only behavior.

Clinical slice CE and NLL of mean subject probabilities measure different quantities. Even for the same images and fixed parameters, NLL after averaging probabilities need not equal mean per-slice NLL. Their historical curve gap is not a matched training/validation generalization gap. The elevated or fluctuating clinical validation loss is retained as an observed result and is not cosmetically corrected.

## New-curve requirement

For new training runs, retain online stage/auxiliary raw CE and optimization-objective logs, then evaluate the full training and validation splits with identical deterministic transforms, evaluation mode, epoch-end weights and endpoint-specific aggregation. Plot clean image-level training/validation NLL together; plot clean subject-level training/validation NLL together. Keep online optimization curves separate. Preserve validation-only checkpoint selection and untouched held-out test data. New runs must be labeled as a new experiment and must not replace historical curves or retrospectively change the 305-epoch count.

## Separate legacy script issue

`src/train_alzheimers.py` defines a validation transform but uses `random_split` on a single `ImageFolder` constructed with the training transform. Consequently both subsets would receive training augmentation if that script were executed as written. The current audit does not establish that this script generated a particular legacy saved history or checkpoint. It is separate from the October 11 trainer, which constructs independent deterministic validation datasets. This finding does not justify relabeling or numerically changing the historical six-run curves. The script was not changed during this scoped curve repair.

## Verification and limits

Seven schema tests cover all twelve histories, reject unknown schemas, unclassified new metrics and missing post-initialization metrics, enforce separation of slice/subject losses and raw/objective losses, and confirm matched Round 2 subject aggregation. The seven existing STS training tests verify the six-run identity, 305 completed epochs, frozen selection and stopping-rule replay. Plot regeneration reads saved summaries only; it performs no inference, retraining, recalibration or checkpoint selection. Historical aggregate losses cannot independently verify every per-image logit or the original arithmetic without the original epoch checkpoints and per-image records.
