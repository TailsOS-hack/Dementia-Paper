# Dementia research repository

[Current results and tables](results/current/README.md) · [Regenerated figures](results/current/FIGURES.md) · [Mendeley audit](datasets/mendeley_xx9zzz6t54_v1/README.md) · [OASIS access status](datasets/OASIS_STATUS.md)

The current scope contains legacy four-class image research, retained Besga/OpenNeuro clinical comparisons, trained checkpoints, methods, raw training histories, attribution controls and future-study software. Earlier broader snapshots remain in Git history. The active repository has no fresh external clinical validation and does not establish clinical superiority or readiness.

## October 18 refresh

- Recomputed all 112 retained clinical model/scenario summaries from saved predictions, including uncertainty and paired comparisons.
- Downloaded and audited Mendeley DOI 10.17632/xx9zzz6t54.1: all 6,400 images exactly match existing legacy pixels. Registered it as an alias; added no duplicate training data.
- Reran all ten frozen stage classifiers on its 1,279-image supplied test split, explicitly labeled as reused-data evaluation.
- Regenerated 35 current plots, including all twelve stored training histories. Unchanged historical stage and native attribution evidence remains available.
- Added a validated local OASIS intake command. Access approval is pending, so no OASIS training or results are claimed.

## Organization

- `results/current/`: current tables, figures, verification and reproducible outputs.
- `analysis_refresh/`: refreshed evaluation/statistics/plotting scripts.
- `datasets/`: source audit, provenance, overlap and access status.
- `docs/kevin_paper_20261011/`: study methods, historical stage results, training code and future research design.
- `models/`: baseline, six Round 1 and six Round 2 research checkpoints, plus initialization weights.
- `research/dementia_round3/`: proposed patient/site-level study implementation; not completed clinical research.
- `paper_package/Dementia_Only_2026-10-09/`: historical dementia figures and native maps.

Install Python dependencies from `requirements.txt`. Use Git LFS (`git lfs install`, then `git lfs pull`) to download model binaries. Raw clinical data, private access records and the large original training collection are not public. Some exact historical reruns require separately obtained source data and the earlier frozen code snapshot.

The existing augmented JPEG data lack verified patient/upstream-family independence. The original100 exercise is reused internal data. Grad-CAM is exploratory. MCI and four-class image labels must not be silently converted into clinical AD/CN diagnoses. Tests are software checks, not clinical evidence.

`REPOSITORY_MANIFEST.json` records current file hashes. Legacy package indexes describe historical layouts and are not a current inventory.
