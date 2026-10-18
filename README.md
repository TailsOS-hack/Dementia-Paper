# Dementia paper research repository

Dedicated dementia/Alzheimer research code, model checkpoints, figures, raw learning histories, statistical comparisons, methods, dataset-source notes, and study plans extracted on October 17, 2026.

**For Abid or an AI writing assistant, begin with [AI_START_HERE.md](AI_START_HERE.md).** See [source coverage](SOURCE_COVERAGE.md) for the scope and explicit omissions.

## Start here

- [Current scientific handoff](handoff/START_HERE.md)
- [Model and dataset comparison](handoff/MODEL_AND_DATASET_COMPARISON.md)
- [Round 2 results and methods](docs/kevin_paper_20261011/round2_publication/README.md)
- [Latest 49-page graph atlas](docs/kevin_paper_20261011/round2_publication/combined_graphs/Round2_All_Graphs_Atlas.pdf)
- [All 115 latest Grad-CAM cases](docs/kevin_paper_20261011/round2_publication/artifacts/stage/attribution_diagnostics/gallery.html) — download and open locally.
- [Earlier publication results and 40-page atlas](docs/kevin_paper_20261011/publication/README.md)
- [Complete historical dementia-only evidence](paper_package/Dementia_Only_2026-10-09/START_HERE.txt)
- [Round 3 proposed study and software](docs/kevin_paper_20261011/round3/README.md)

## Checkpoints and code

`models/` includes the original dementia checkpoint, six Round 1 checkpoints, six Round 2 checkpoints, and the recorded initialization weights. Checkpoint identity records remain beside the weights. Binaries use Git LFS; run `git lfs install` and `git lfs pull` after cloning. GitHub's ordinary ZIP download may contain LFS pointers instead of the weights.

Training, analysis, evaluation, reporting and figure scripts retain their paths under `docs/kevin_paper_20261011/`. The proposed next-study implementation is in `research/dementia_round3/`, with its tests under `tests/`. Supporting attribution utilities are in `src/`. There are no tumor models, tumor datasets, routing models, or mixed-disease GUI applications. The generic provenance module was reduced to functions needed by dementia attribution; original scientific result files and checkpoints are preserved.

Create a Python virtual environment and install `requirements.txt`. Recorded tested versions are in `research/dementia_round3/requirements-local-tested.txt` and `docs/kevin_paper_20261011/training/requirements-analysis.txt`. Full replay needs the original approved data and preprocessing assets; see [reconstruction boundaries](docs/kevin_paper_20261011/publication/RECONSTRUCTION.md). Missing clinical inputs must not be substituted with synthetic outcomes.

## Evidence boundaries

This is research-only. Clinical superiority and clinical readiness were not demonstrated. The latest clinical ensemble scored 9/19 on the fresh clinical cohort versus 10/19 for the older ensemble. The reused 100-image four-class benchmark contains training/validation overlap and is not independent clinical validation. Grad-CAM is exploratory attribution, not validated anatomical localization. Round 3 is a proposed workflow with software checks, not a new completed clinical study.

Raw clinical MRI, participant records, private access attestations, credentials and personal correspondence are excluded. The full augmented training image collection is not duplicated; source inventories, manifests and acquisition/preprocessing code are supplied. Historical benchmark assets already present in the source project are retained for the paper's comparisons; source-use terms still apply. This repository grants no new dataset license. Historical document references to the original mixed project identify provenance, not additional scope.

`REPOSITORY_MANIFEST.json` records every delivered file's SHA-256 and size (except itself). `handoff/FILE_MANIFEST.csv` and other original manifests describe their original packages and are retained as historical provenance.
