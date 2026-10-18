# Dementia research repository

[STS manuscript and full test results](results/sts/README.md) · [Current results and tables](results/current/README.md) · [Regenerated figures](results/current/FIGURES.md) · [Mendeley audit](datasets/mendeley_xx9zzz6t54_v1/README.md) · [OASIS access status](datasets/OASIS_STATUS.md)

The current scope contains legacy four-class image research, retained Besga/OpenNeuro clinical comparisons, trained checkpoints, methods, raw training histories, attribution controls and future-study software. Earlier broader snapshots remain in Git history. The active repository has no fresh external clinical validation and does not establish clinical superiority or readiness.

## October 18 refresh

- Added the six-page combined [STS Results manuscript](results/sts/README.md), with four tables and four embedded graphs. It retains the historical six-run, 305-epoch study and its 5,450-image, matched, sensitivity and reused-image benchmarks, then presents the separate completed retraining experiment.
- Completed six new October 18 runs across **329 epochs**. The [verified report](results/sts/retraining/report.json), [raw run evidence](results/sts/retraining/raw/) and [six inference states](models/sts_retrain_20261018/README.md) are included. New train/validation curves use the same epoch-end weights, clean transform, evaluation mode and unweighted unsmoothed NLL; no new held-out estimates were produced.
- Recomputed all 112 retained clinical model/scenario summaries from saved predictions, including uncertainty and paired comparisons.
- Downloaded and audited Mendeley DOI 10.17632/xx9zzz6t54.1: all 6,400 images exactly match existing legacy pixels. Registered it as an alias; added no duplicate training data.
- Reran all ten frozen stage classifiers on its 1,279-image supplied test split, explicitly labeled as reused-data evaluation.
- Regenerated 35 current plots, including twelve historical training histories. Unchanged historical stage and native attribution evidence remains available.
- Added a validated local OASIS intake command. Access approval is pending, so no OASIS training or results are claimed.

## Organization

- `results/sts/` and `analysis/sts_results/`: STS manuscript, recovered historical metrics, source audits and regeneration code.
- `results/current/`: current tables, figures, verification and reproducible outputs.
- `analysis_refresh/`: refreshed evaluation/statistics/plotting scripts.
- `datasets/`: source audit, provenance, overlap and access status.
- `docs/kevin_paper_20261011/`: study methods, historical stage results, training code and future research design.
- `models/`: baseline, six Round 1 and six Round 2 research checkpoints, initialization weights, and six October 18 inference-state derivatives.
- `research/dementia_round3/`: proposed patient/site-level study implementation; not completed clinical research.
- `research/sts_retrain_20261018/`: completed retraining implementation, report validation and public packaging code; the 329 new epochs remain separate from the historical 305 epochs.
- `paper_package/Dementia_Only_2026-10-09/`: historical dementia figures and native maps.

Install Python dependencies from `requirements.txt`. Use Git LFS (`git lfs install`, then `git lfs pull`) to download model binaries. Raw clinical data, private access records and the large original training collection are not public. Some exact historical reruns require separately obtained source data and the earlier frozen code snapshot.

The retraining publication report verified all twelve original best/last checkpoints locally. Public inference states have their own hashes and omit resume state; [model provenance](models/sts_retrain_20261018/provenance.json) maps each one to its selected original checkpoint and verifies unchanged tensor bytes. A [public metadata-only rebuild](results/sts/README.md#reproduce-the-new-retraining-report) needs neither these model binaries nor private clinical images.

The existing augmented JPEG data lack verified patient/upstream-family independence. The original100 exercise is reused internal data. Grad-CAM is exploratory. MCI and four-class image labels must not be silently converted into clinical AD/CN diagnoses. Tests are software checks, not clinical evidence.

`REPOSITORY_MANIFEST.json` records current public file payload hashes. For Git LFS files, the size and SHA256 refer to the model payload, including when the checkout contains only its pointer. Preview an update with `python -m analysis.sts_results.refresh_repository_manifest`; run with `--write` after finalizing public files, then `--check` to verify coverage. Raw-data and private working archives are excluded. Legacy package indexes describe historical layouts and are not a current inventory.
