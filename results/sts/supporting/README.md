# Supporting dementia image-benchmark audit

These are the frozen October 11 Round 1 results: two training conditions with three seeds each. The historical baseline is an additional model. They are distinct from older baseline training and later development rounds.

All 27 model/cohort point estimates were recomputed from public prediction CSVs. Six ensemble tables were reconstructed by equal-weight averaging of case-aligned member probabilities. All 36 paired contrasts retain their frozen 2,000-resample conditional bootstrap intervals; point differences and discordant counts were verified. Intervals were preserved, not regenerated. No McNemar test was present in these source records and none was added.

| Model | Matched 1,058 | Sensitivity 756 | Reused 100 |
| --- | --- | --- | --- |
| Historical baseline | 1057/1058 (99.905%) | 755/756 (99.868%) | 100/100 (100.000%) |
| Stage only / seed 11 | 1055/1058 (99.716%) | 753/756 (99.603%) | 99/100 (99.000%) |
| Multitask / seed 11 | 1054/1058 (99.622%) | 752/756 (99.471%) | 100/100 (100.000%) |
| Stage only / seed 12 | 1057/1058 (99.905%) | 755/756 (99.868%) | 100/100 (100.000%) |
| Multitask / seed 12 | 1056/1058 (99.811%) | 754/756 (99.735%) | 99/100 (99.000%) |
| Stage only / seed 13 | 1057/1058 (99.905%) | 755/756 (99.868%) | 100/100 (100.000%) |
| Multitask / seed 13 | 1055/1058 (99.716%) | 754/756 (99.735%) | 99/100 (99.000%) |
| Stage ensemble* | 1058/1058 (100.000%) | 756/756 (100.000%) | 100/100 (100.000%) |
| Multitask ensemble* | 1056/1058 (99.811%) | 754/756 (99.735%) | 100/100 (100.000%) |

The matched cohort has 966 image-similarity groups; 302 images have an old training/validation group mate. The nested sensitivity cohort retains 733 groups after excluding those 302 images. Hash groups do not establish patient identity or independence. Original100 has historical old-model exposure of 66 training, 9 validation and 25 test images. It remains a reused descriptive benchmark.

The stage ensemble corrects the baseline's one error: +0.095 percentage points on matched images (conditional 95% interval 0.000 to 0.290) and +0.132 on sensitivity images (0.000 to 0.399). Both intervals include zero. Perfect observed predictions and degenerate bootstrap intervals do not establish certainty. These results do not establish clinical superiority or readiness.

`metrics.csv` includes accuracy, balanced accuracy, macro F1, raw numerators and preserved intervals. `paired_differences.csv` includes all paired estimates, source fields and discordant counts. `condition_summaries.csv` reports means and sample SDs for three seeds; those SDs are not confidence intervals. `audit.json` records confusion matrices, source hashes and limitations.

Reproduce from the repository root: `python -m analysis.sts_results.supporting --output-dir results/sts/supporting`.
