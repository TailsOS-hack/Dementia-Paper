# New retraining with comparable learning curves

The separate October 18 experiment completed **six runs and 329 epochs**. The
[verified report](../../results/sts/retraining/report.json),
[selected-checkpoint summary](../../results/sts/retraining/summary.csv),
[public run evidence](../../results/sts/retraining/raw/) and
[six inference states](../../models/sts_retrain_20261018/README.md) are included.
The [combined six-page manuscript](../../results/sts/README.md) has four tables and
four embedded graphs. These new observations remain separate from the historical
305-epoch experiment and add no held-out performance estimates.

The experiment retains the original two conditions
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
identity because it changes augmentation RNG assignment. Condition preference
minimizes mean selected validation NLL across three seeds; exact condition-mean
ties use lexicographic condition-name order. Within the preferred condition,
the representative minimizes selected validation NLL, with ascending seed used
only for exact NLL ties. The archived `selection_basis` says “lowest seed”; that
description is imprecise and is preserved as source metadata. The frozen code and
report verification implement the NLL-based rule. The completed suite preferred
stage-only, represented by seed 20261011 at epoch 27.

The frozen helper is imported by its full repository module name, so its dataset
class can be reconstructed by spawned DataLoader workers as well as Linux forked
workers. This changes importability without changing the frozen helper's bytes.

Each run contains `history.csv`, `history.json`, per-epoch clean evaluation summaries,
raw three-panel `learning_curves.png/.pdf`, `selection.json`, `best.pt`, and `last.pt`.
No smoothing, interpolation, manual loss adjustment or held-out evaluation occurs.
The new observations must be reported separately from the historical experiment.

Tests: `python -m unittest tests.test_sts_retrain -v`.

## Rebuild from public metadata

The public records are sufficient to rebuild numeric summaries and curves without
training, image inference or checkpoint files. Keep the output separate from the
published report:

```sh
python -m research.sts_retrain_20261018.report \
  --input results/sts/retraining/raw \
  --output /tmp/sts_retraining_rebuild
```

This checks histories, per-epoch denominators, stopping, selection and available
public stage-manifest evidence. It reports metadata-only verification for absent
original checkpoints and the private clinical manifest. The publication report
records a separate local check of all twelve original best/last binaries and both
manifests. Public inference derivatives have distinct file hashes and cannot stand
in for the original resumable files. Their [provenance](../../models/sts_retrain_20261018/provenance.json)
maps each selected source hash and epoch to its derivative and unchanged tensor
bytes. No new held-out estimates follow from this rebuild.

## Prepare minimized inputs

`prepare_inputs.py` reproduces the verified training-only bundle from separately held source data and official ImageNet initialization. It checks frozen stage pixel identities, clinical development membership, image metadata and every archive member hash. The unchanged stage manifest retains test-row metadata for split verification; no test pixels are included. Private bundle output must be outside this repository. The script performs no upload.

```sh
python -m research.sts_retrain_20261018.prepare_inputs \
  --source-root /path/to/Neuro-Radiology-Database \
  --imagenet-initialization /path/to/mobilenet_v3_large-8738ca79.pth \
  --output-dir /private/path/outside/repository
```

## Reproduce public evidence packaging

The completed publication package was built after all six runs passed the
aggregate verifier. To reproduce packaging from separately held originals,
generate the aggregate report with
`python -m research.sts_retrain_20261018.report --require-checkpoints` against
the full retrieved suite, supplying the
same private clinical manifest used below. Packaging independently verifies all
twelve original checkpoint binaries and checks that the supplied report matches
the histories, selection, split checks and source hashes. Existing target folders
are refused. This command stages files locally; it does not commit or publish.
The report also pins all six generated PNG/PDF files by SHA256. Packaging checks
those hashes, and the manuscript checks its two figure hashes before embedding.
Generate the publication report with plots enabled: `--no-plots` records an empty
figure map and cannot reuse plot files left by a previous build.

```sh
python -m research.sts_retrain_20261018.package \
  --repo . \
  --suite /private/path/sts_retrain_20261018 \
  --report results/sts/retraining/report.json \
  --clinical-manifest /private/path/clinical_manifest.csv \
  --runtime /private/path/sts_retrain_20261018/runtime.json \
  --export-inference-states --trust-local-checkpoint-pickle
```

Only named public metadata, numeric histories, aggregate per-epoch records and
curves enter `results/sts/retraining/raw/`; private manifests, input inventories,
images, logs and original checkpoint binaries are excluded. The clinical manifest
is read only for verification. Runtime and the frozen protocol are checked before
copying. Omit the last two flags to package metadata without models.

Optional inference states enter `models/sts_retrain_20261018/` under Git LFS.
Their hashes differ from the original resumable `best.pt` files. Provenance maps
each derivative to its original SHA256 and selected epoch and verifies unchanged
model tensor bytes, `weights_only=True` read-back and strict loading with the
frozen constructor. Multitask auxiliary weights are retained; unused stage-only
auxiliary weights and all optimizer, scheduler and RNG state are omitted. Original
selection/completion hashes continue to identify the local resumable files.
No image inference or new held-out performance estimate occurs during packaging.

Packaging helper tests use temporary synthetic data only:
`python -m unittest tests.test_sts_retrain_package -v`.
