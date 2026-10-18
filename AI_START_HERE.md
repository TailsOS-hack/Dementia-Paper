# Start here: understanding and writing the dementia paper

This repository is a scientific evidence archive, analysis/code handoff, and writing source for Abid and an AI assistant. Read the evidence before drafting. The material spans different experiments, dates, endpoints and levels of maturity. Do not merge them into one performance claim.

## Recommended reading order

1. `README.md` — scope and installation.
2. `handoff/START_HERE.md` and `handoff/MODEL_AND_DATASET_COMPARISON.md` — the latest consolidated scientific account.
3. `docs/kevin_paper_20261011/round2_publication/README.md`, `Methods.md`, `RESULTS.md`, and `DATA_AND_LIMITATIONS.md` — completed Round 2.
4. `docs/kevin_paper_20261011/publication/reports/` — completed Round 1, original methods and statistical handoff.
5. `docs/kevin_paper_20261011/publication/DATA_AND_REPRODUCIBILITY.md` and `RECONSTRUCTION.md` — what can and cannot be reproduced from the shared artifacts.
6. `handoff/FIGURE_CAPTION_GUIDANCE.md` and `handoff/FIGURE_INDEX.csv` — figure selection and defensible captions.
7. `docs/kevin_paper_20261011/round3/` — proposed future study; read separately from completed results.
8. `SOURCE_COVERAGE.md` and `REPOSITORY_MANIFEST.json` — scope and file integrity.

## Study map

| Component | Question and unit | Status | Primary evidence |
|---|---|---|---|
| Historical dementia classifier | Four-class augmented JPEG image classification | Historical internal results; patient independence unknown | `paper_package/Dementia_Only_2026-10-09/`, baseline checkpoint |
| Original 100-image reader exercise | Same-case four-class image/reader comparisons | Reused internal benchmark with known old-development overlap | Round 1 and Round 2 publication result tables, `data/evaluation/` |
| Round 1 models | Stage-only and multitask image-model comparisons across fixed seeds | Completed research runs | `publication/reports/`, `publication/training/`, `models/kevin_20261011/` |
| Round 2 stage models | Four-class stage task, attribution and shortcut controls | Completed; original100 scores remain internal | `round2_publication/artifacts/stage/`, `models/kevin_round2_20261011/` |
| Round 2 clinical ensemble | Separate patient-level AD/control task on a fresh cohort | Completed external generalization failure | `round2_publication/RESULTS.md`, clinical report and comparison tables |
| Round 3 | Future structural-MRI patient/site-level study with frozen preprocessing and evaluation gates | Proposed workflow and software tests; no completed real training/outcomes | `round3/`, `research/dementia_round3/`, `tests/test_round3*` |

## Claims that must remain attached to the results

- The reused original100 benchmark includes 66 old-training, nine old-validation and 25 old-test images. A 100/100 score is not independent clinical validation or superiority over radiologists in practice.
- The latest clinical ensemble predicted AD for all 19 fresh Harvard participants, including all 10 controls: 9/19 correct, compared with 10/19 for the older clinical ensemble. Preserve this negative result.
- The four-class stage task and binary clinical AD/control task have different labels and evaluation units. Do not substitute one for the other or create stage labels from AD/MCI/control labels.
- Grad-CAM is exploratory attribution. Preserve all cases, native maps, randomization controls, flat/failed maps, and probability-replay warnings. A visually central map does not establish anatomical validity.
- The local augmented JPEG dataset lacks adequate patient/upstream-family provenance for a patient-independent claim. Separate image counts from patient counts and acquired cohorts from proposed/access-pending sources.
- Synthetic checks validate software behavior only. Round 3 readiness or passing tests is not clinical evidence.
- Historical files are dated evidence. When statements conflict, prioritize the latest scoped Round 2 audit, investigate numerical source files, and document the discrepancy rather than averaging claims.

## How to draft the manuscript

Create a claim/evidence table first: proposed sentence, exact source path, experiment/endpoint, denominator, uncertainty, and limitation. Use that table to draft an abstract, introduction, methods, results, discussion, limitations, data/code availability, and figure legends. Use existing completed methods for completed experiments; label Round 3 as future work only. Cite exact tables/CSVs and retain uncertainty and failed checks. Verify external references against original publications before submission. Do not invent author affiliations, ethics/IRB determinations, consent, funding, dataset permission, journal requirements or missing measurements; mark those as author questions.

For each table, state dataset, split, unit of analysis, model/seed or fixed ensemble, denominator, metric, interval and whether the comparison was prespecified or post hoc. For each figure, state the originating round and whether it is quantitative evidence, a diagnostic control or illustrative attribution. Keep original values and image pixels intact.

## Suggested prompt to give another AI

> Read AI_START_HERE.md, README.md, SOURCE_COVERAGE.md, and the listed scientific sources. Explain the whole dementia study to me, distinguish the completed rounds from planned work, and build a claim-to-evidence table with exact repository paths. Then help draft a transparent dementia research paper, including negative results and limitations. Do not claim clinical superiority, independent validation for original100, anatomical validation from Grad-CAM, or new Round 3 results. Ask me for authorship, ethics, permissions and target-journal details that are not in the evidence. Verify every reported number against the source tables and flag unresolved discrepancies.

## Working with the files

Clone with Git LFS to get weights. Open galleries locally; use PDF atlases for rapid visual review. Index text/CSV/JSON and source code before ingesting binaries. The manifest has SHA-256 hashes and sizes. Older package manifests describe their original delivery layouts; the repository manifest describes this delivery. Exact reruns may require separately approved clinical inputs and pinned preprocessing tools. The repository is sufficient for understanding the supplied evidence, but it does not contain every raw dataset or establish missing approvals.
