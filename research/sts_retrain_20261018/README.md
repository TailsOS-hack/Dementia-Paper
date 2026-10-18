# New retraining with comparable learning curves

This is a separate October 18 experiment. It retains the original two conditions
(stage only and multitask), three seeds (20261011–20261013), ImageNet initialization,
full 19,023-image training passes, optimizer, augmentation, weighting, label
smoothing and validation-only checkpoint-selection rules. It never loads an old
study checkpoint as initialization. The frozen October 11 source remains unchanged.

Every epoch now evaluates **all 19,023 training images and all 2,705 validation
images using the same clean transform, unweighted unsmoothed NLL, evaluation mode,
and end-of-epoch parameters**. `train_loss` and `val_loss` are comparable in those
respects. Training resubstitution still is not a generalization estimate.

The original online train NLL is retained separately as `train_online_nll`.
`train_stage_objective` records the weighted smoothed stage objective, and
`train_total_objective_per_step` records the actual scalar objective averaged over
optimizer steps, including the scheduled auxiliary term. These are separately
labelled because they do not measure the same quantity. Multitask clinical
diagnostics distinguish online balanced slices from clean per-subject aggregation.

Only validation NLL selects checkpoints. Neither training diagnostics nor test
outcomes select epochs, conditions or seeds. Added diagnostic passes preserve
Python, NumPy, Torch CPU, CUDA and available MPS RNG states. No test or external
image is evaluated by these scripts.

Run from the repository root, with the original frozen data available locally:

```sh
python -m research.sts_retrain_20261018.suite \
  --data-root /path/to/Neuro-Radiology-Database \
  --output /workspace/sts_retrain_20261018 \
  --device cuda --workers 4
```

Optional `--imagenet-weights /path/to/mobilenet_v3_large-8738ca79.pth` uses the official
cached ImageNet weights. Otherwise torchvision downloads its official weights.
Default manifests are the original `training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv`
and `data/external/clinical_t1_preprocessed_v1/clinical_slices.csv`; both are checked
against their expected SHA256. A minimized clinical manifest that retains the exact
original training/validation membership, order, labels and image paths can be used
with `--clinical-manifest-sha256 <audited-minimized-file-sha256>`. Its membership
audit must establish the match to the original manifest before running; an override
does not itself establish scientific equivalence. No test/external images are
needed. Clinical image paths are rooted under `data` by default.

Six runs execute sequentially in the original seed-major order. Maximum epochs
100, minimum epochs 20, patience 15 and minimum validation-NLL improvement 0.0001
remain unchanged. Output paths must be inside a new `sts_retrain_20261018*` directory.
Existing outputs are refused unless `--resume` is explicitly supplied; resume
accepts only this study's own matching checkpoint/configuration, never historical
checkpoints. The suite writes per-run logs, `suite_status.json`, completion records,
and `validation_only_preference.json` after all six runs complete.

`last.pt` is the committed epoch record. Resume validates its complete loss history,
replays validation selection and patience, reconstructs missing history/selection
outputs, and recovers `best.pt` from `last.pt` when the last epoch was selected.
Later epochs pin the prior selected checkpoint's exact hash. An interruption before
the first committed checkpoint is restartable only for the suite's verified current
run: its incomplete configuration/temporary files are archived, never deleted.
Completed runs carry hashes for both checkpoints and history/selection/curve files;
suite resume verifies them before skipping a run. Worker count is part of resume
identity because it changes augmentation RNG assignment. Ties between condition
means use the original lexicographic condition-name rule, then ascending seed.

The frozen helper is imported by its full repository module name, so its dataset
class can be reconstructed by spawned DataLoader workers as well as Linux forked
workers. This changes importability without changing the frozen helper's bytes.

Each run contains `history.csv`, `history.json`, per-epoch clean evaluation summaries,
raw three-panel `learning_curves.png/.pdf`, `selection.json`, `best.pt`, and `last.pt`.
No smoothing, interpolation, manual loss adjustment or held-out evaluation occurs.
The new observations must be reported separately from the historical experiment.

Tests: `python -m unittest tests.test_sts_retrain -v`.

## Prepare minimized inputs

`prepare_inputs.py` reproduces the verified training-only bundle from separately held source data and official ImageNet initialization. It checks frozen stage pixel identities, clinical development membership, image metadata and every archive member hash. The unchanged stage manifest retains test-row metadata for split verification; no test pixels are included. Private bundle output must be outside this repository. The script performs no upload.

```sh
python -m research.sts_retrain_20261018.prepare_inputs \
  --source-root /path/to/Neuro-Radiology-Database \
  --imagenet-initialization /path/to/mobilenet_v3_large-8738ca79.pth \
  --output-dir /private/path/outside/repository
```
