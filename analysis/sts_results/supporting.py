"""Audit the frozen Round 1 image benchmarks without training or inference.

Point estimates are recomputed from public, case-aligned predictions. Existing
conditional bootstrap intervals are preserved with their precise source; this
module neither changes the original analysis hierarchy nor generates p-values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

CLASSES = ("MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
PRIMARY = "internal_same_case"
SENSITIVITY = "old_hash_group_unexposed_sensitivity"
COHORTS = (PRIMARY, SENSITIVITY, "original100")
SINGLES = ("old_baseline",) + tuple(f"candidate_{i}" for i in range(1, 7))
ENSEMBLES = {"ensemble_stage_only": ("candidate_1", "candidate_3", "candidate_5"),
             "ensemble_multitask": ("candidate_2", "candidate_4", "candidate_6")}
MODELS = SINGLES + tuple(ENSEMBLES)
DISPLAY = {"old_baseline": "Historical baseline", "candidate_1": "Stage only / seed 11",
           "candidate_2": "Multitask / seed 11", "candidate_3": "Stage only / seed 12",
           "candidate_4": "Multitask / seed 12", "candidate_5": "Stage only / seed 13",
           "candidate_6": "Multitask / seed 13", "ensemble_stage_only": "Stage ensemble*",
           "ensemble_multitask": "Multitask ensemble*"}
PUBLICATION = Path("docs/kevin_paper_20261011/publication")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context, tolerance=1e-12):
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f"{context}: recomputed {actual!r}, frozen {expected!r}")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def index_predictions(rows, original100=False):
    """Normalize known public ID schemas and reject duplicates or bad labels."""
    result = {}
    for row in rows:
        identifiers = [row[k] for k in ("case_id", "path", "FileName") if row.get(k)]
        require(bool(identifiers), "Prediction row lacks a case ID")
        ids = [Path(v).name if original100 else v for v in identifiers]
        require(len(set(ids)) == 1, f"Conflicting case IDs: {identifiers}")
        key = ids[0]
        require(key not in result, f"Duplicate case ID: {key}")
        truth, pred = row["TrueCategory"], row["ModelPrediction"]
        require(truth in CLASSES and pred in CLASSES, f"Unknown label for {key}")
        probs = tuple(float(row[f"P_{label}"]) for label in CLASSES)
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in probs),
                f"Invalid probability for {key}")
        close(sum(probs), 1.0, f"Probability sum for {key}", tolerance=1e-5)
        require(CLASSES[max(range(4), key=lambda i: probs[i])] == pred,
                f"Argmax disagrees with saved prediction for {key}")
        result[key] = {"truth": truth, "prediction": pred, "probabilities": probs}
    require(bool(result), "Empty prediction table")
    return result


def align(reference, candidate):
    require(set(reference) == set(candidate),
            f"Case ID mismatch: missing={len(set(reference)-set(candidate))}, "
            f"extra={len(set(candidate)-set(reference))}")
    for key in reference:
        require(reference[key]["truth"] == candidate[key]["truth"],
                f"Reference-label mismatch for {key}")


def ensemble(members):
    for member in members[1:]:
        align(members[0], member)
    result = {}
    for key in sorted(members[0]):
        probs = tuple(sum(m[key]["probabilities"][i] for m in members) / len(members)
                      for i in range(4))
        result[key] = {"truth": members[0][key]["truth"], "probabilities": probs,
                       "prediction": CLASSES[max(range(4), key=lambda i: probs[i])]}
    return result


def metrics(rows):
    confusion = [[0] * 4 for _ in CLASSES]
    for row in rows.values():
        confusion[CLASSES.index(row["truth"])][CLASSES.index(row["prediction"])] += 1
    counts = list(map(sum, confusion))
    require(all(counts), "All four reference classes are required")
    correct, n = sum(confusion[i][i] for i in range(4)), sum(counts)
    precision_denominators = [sum(confusion[i][j] for i in range(4)) for j in range(4)]
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "balanced_accuracy": sum(confusion[i][i] / counts[i] for i in range(4)) / 4,
            "macro_f1": sum(2 * confusion[i][i] / (counts[i] + precision_denominators[i])
                            for i in range(4)) / 4,
            "class_counts": counts, "confusion_matrix": confusion}


def paired_counts(candidate, reference):
    align(reference, candidate)
    result = {"n_paired": len(reference), "candidate_correct_reference_wrong": 0,
              "candidate_wrong_reference_correct": 0, "both_correct": 0, "both_wrong": 0}
    for key in reference:
        a = candidate[key]["truth"] == candidate[key]["prediction"]
        b = reference[key]["truth"] == reference[key]["prediction"]
        field = ("both_correct" if a and b else "both_wrong" if not a and not b else
                 "candidate_correct_reference_wrong" if a else "candidate_wrong_reference_correct")
        result[field] += 1
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_plot(output, metric_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13.7, 6.0), sharey=True)
    titles = ("Matched comparison · 1,058 images", "Similarity sensitivity · 756 images",
              "Reused historical · 100 images")
    lookup = {(r["cohort"], r["model"]): r for r in metric_rows}
    for ax, cohort, title in zip(axes, COHORTS, titles):
        for y, model in enumerate(MODELS):
            row = lookup[cohort, model]
            value = row["accuracy"] * 100
            color = "#333333" if model == "old_baseline" else (
                "#0072B2" if model in ("candidate_1", "candidate_3", "candidate_5", "ensemble_stage_only")
                else "#D55E00")
            ax.plot(value, y, "D" if model in ENSEMBLES else "o", color=color, markersize=6)
            ax.text(97.75, y, f"{row['correct']}/{row['n']}", fontsize=8, va="center", color="#444444")
        ax.set(title=title, xlabel="Image-label accuracy (%)", xlim=(97.65, 100.15),
               xticks=(98, 99, 100), yticks=range(len(MODELS)))
        ax.set_yticklabels([DISPLAY[m] for m in MODELS])
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    figure.suptitle("Round 1 supporting dementia image benchmarks", fontsize=16, x=.01, ha="left")
    figure.text(.01, .055, "* Fixed equal-weight probability ensembles; each uses three seeds. "
                "Counts are correct images / evaluated images. Narrowed x-axis.", fontsize=9)
    figure.text(.01, .025, "Sensitivity is nested in the matched cohort. Hash groups are not patients; "
                "the historical 100 images were reused. Points show estimates; intervals are in metrics.csv.", fontsize=9)
    figure.tight_layout(rect=(0, .09, 1, .94))
    figure.savefig(output / "accuracy_comparison.png", dpi=200)
    figure.savefig(output / "accuracy_comparison.pdf")
    plt.close(figure)


def run(repo_root: Path, output_dir: Path, make_plot=True):
    repo_root, output_dir = Path(repo_root).resolve(), Path(output_dir).resolve()
    publication = repo_root / PUBLICATION
    sources = {}

    def source(path):
        path = publication / path
        sources[str(path.relative_to(repo_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_predictions(path, cohort):
        return index_predictions(csv_rows(source(path)), original100=cohort == "original100")

    primary = json.loads(source("results/uncertainty_analysis.json").read_text())
    supplement = json.loads(source("results/ensemble_supplement.json").read_text())
    specifications = json.loads(source("results/ensemble_specifications.json").read_text())
    sensitivity_audit = json.loads(source("evidence/sensitivity_audit.json").read_text())
    manifests = {}
    for cohort, filename in ((PRIMARY, "same_case_test_intersection.csv"),
                             (SENSITIVITY, "old_hash_group_unexposed_sensitivity.csv")):
        rows = csv_rows(source("prepared/" + filename))
        manifests[cohort] = {r["path"]: r for r in rows}
        require(len(rows) == len(manifests[cohort]), f"Duplicate {cohort} manifest ID")
        require(len(rows) == (1058 if cohort == PRIMARY else 756), "Unexpected cohort count")
        require(len({r["group_id"] for r in rows}) == (966 if cohort == PRIMARY else 733),
                "Unexpected image-similarity group count")
        require(all(r["split"] == r["baseline_strict_split"] == "test" for r in rows),
                "Non-held-out manifest row")
    require(set(manifests[SENSITIVITY]) < set(manifests[PRIMARY]), "Sensitivity is not nested")
    require(len(manifests[PRIMARY]) - len(manifests[SENSITIVITY]) == sensitivity_audit["excluded"]["rows"],
            "Sensitivity exclusion count mismatch")
    for key, row in manifests[SENSITIVITY].items():
        require(row == manifests[PRIMARY][key], f"Sensitivity manifest row changed: {key}")
    key_rows = csv_rows(source("results/original100/radiologist_test_key.csv"))
    truth100 = {r["FileName"]: r["TrueCategory"] for r in key_rows}
    require(len(key_rows) == len(truth100) == 100, "Original100 key must have 100 unique images")

    all_predictions = {cohort: {} for cohort in COHORTS}
    for model in SINGLES:
        main = read_predictions(f"results/legacy_predictions/{model}_{PRIMARY}_predictions.csv", PRIMARY)
        require(set(main) == set(manifests[PRIMARY]), "Primary prediction/manifest ID mismatch")
        for key, row in main.items():
            require(row["truth"] == manifests[PRIMARY][key]["subtype"], f"Manifest label mismatch: {key}")
        all_predictions[PRIMARY][model] = main
        all_predictions[SENSITIVITY][model] = {k: main[k] for k in manifests[SENSITIVITY]}
        historical = read_predictions(f"results/original100/{model}_original100_predictions.csv", "original100")
        require(set(historical) == set(truth100), "Original100 prediction/key ID mismatch")
        require(all(r["truth"] == truth100[k] for k, r in historical.items()), "Original100 key-label mismatch")
        all_predictions["original100"][model] = historical

    for cohort in COHORTS:
        for model, member_names in ENSEMBLES.items():
            spec = specifications[model]
            require(tuple(m["evaluation_name"] for m in spec["members"]) == member_names,
                    "Ensemble membership changed")
            require(tuple(spec["class_order"]) == CLASSES, "Ensemble class order changed")
            require(all(m["weight"] == 1/3 for m in spec["members"]), "Ensemble weights changed")
            reconstructed = ensemble([all_predictions[cohort][m] for m in member_names])
            folder = "original100" if cohort == "original100" else "legacy_predictions"
            saved = read_predictions(f"results/{folder}/{model}_{cohort}_predictions.csv", cohort)
            align(reconstructed, saved)
            for key in reconstructed:
                require(reconstructed[key]["prediction"] == saved[key]["prediction"], "Ensemble argmax mismatch")
                for a, b in zip(reconstructed[key]["probabilities"], saved[key]["probabilities"]):
                    close(a, b, f"Ensemble probability {model}/{cohort}/{key}")
            all_predictions[cohort][model] = reconstructed

    metric_rows, pair_rows, summary_rows, details = [], [], [], {}
    verified_points = verified_pairs = 0
    for cohort in COHORTS:
        points = {model: metrics(rows) for model, rows in all_predictions[cohort].items()}
        details[cohort] = {"models": points, "n_resampling_units": primary["stage"][cohort]["n_resampling_units"]}
        for frozen, filename in ((primary, "uncertainty_analysis.json"), (supplement, "ensemble_supplement.json")):
            section = frozen["stage"][cohort]
            require(tuple(section["class_order"]) == CLASSES, "Frozen class order mismatch")
            for model, record in section["models"].items():
                for key, expected in record["point"].items():
                    if isinstance(expected, float):
                        close(points[model][key], expected, f"{cohort}/{model}/{key}")
                    else:
                        require(points[model][key] == expected, f"Frozen point mismatch {cohort}/{model}/{key}")
                verified_points += 1
            for name, record in section["paired_comparisons"].items():
                candidate, reference = record["candidate"], record["reference"]
                counts = paired_counts(all_predictions[cohort][candidate], all_predictions[cohort][reference])
                for key, expected in counts.items():
                    require(record[key] == expected, f"Paired count mismatch {cohort}/{name}/{key}")
                for metric in METRICS:
                    difference = points[candidate][metric] - points[reference][metric]
                    interval = record["difference"][metric]
                    close(difference, interval["estimate"], f"Paired difference {cohort}/{name}/{metric}")
                    require(interval["valid_resamples"] == interval["requested_resamples"] == 2000,
                            "Unexpected bootstrap record")
                    pair_rows.append({"cohort": cohort, "comparison": name, "metric": metric,
                                      "estimate": difference, "ci95_low": interval["ci95"][0],
                                      "ci95_high": interval["ci95"][1], **counts,
                                      "resampling_unit": section["resampling_unit"],
                                      "ci_status": "preserved_frozen_not_recomputed",
                                      "source": str(PUBLICATION / "results" / filename),
                                      "source_json_field": f"stage.{cohort}.paired_comparisons.{name}.difference.{metric}"})
                verified_pairs += 1
        for model in MODELS:
            frozen = supplement if model in ENSEMBLES else primary
            filename = "ensemble_supplement.json" if model in ENSEMBLES else "uncertainty_analysis.json"
            section, point = frozen["stage"][cohort], points[model]
            row = {"cohort": cohort, "model": model, "correct": point["correct"], "n": point["n"],
                   "n_resampling_units": section["n_resampling_units"]}
            for metric in METRICS:
                row[metric] = point[metric]
                ci = section["models"][model]["bootstrap"][metric]["ci95"]
                row[metric + "_ci95_low"], row[metric + "_ci95_high"] = ci
            row.update(ci_status="preserved_frozen_not_recomputed", source=str(PUBLICATION / "results" / filename))
            metric_rows.append(row)
        for mode, names in ENSEMBLES.items():
            for metric in METRICS:
                values = [points[m][metric] for m in names]
                summary_rows.append({"cohort": cohort, "condition": mode.removeprefix("ensemble_"),
                                     "metric": metric, "seeds": 3, "mean": mean(values), "sample_sd": stdev(values),
                                     "min": min(values), "max": max(values), "sd_is_confidence_interval": False})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics.csv", metric_rows)
    write_csv(output_dir / "paired_differences.csv", pair_rows)
    write_csv(output_dir / "condition_summaries.csv", summary_rows)
    audit = {"status": "verified", "experiment": "kevin_20261011 Round 1",
             "scope": "Supporting four-stage image benchmarks, not the older baseline training or later rounds",
             "new_training": False, "new_inference": False,
             "cohort_model_rows": len(metric_rows), "frozen_point_records_verified": verified_points,
             "paired_contrasts_verified": verified_pairs,
             "ensemble_probability_tables_reconstructed": len(COHORTS) * len(ENSEMBLES),
             "interval_status": "Frozen 2000-resample conditional paired bootstrap intervals preserved; not rerun",
             "mcnemar_status": "No McNemar statistic occurs in these frozen primary/supplement records; no new significance test added",
             "baseline_group_exposure": primary["baseline_group_exposure"],
             "historical100_old_exposure": primary["protocol"]["original100_old_exposure"],
             "class_order": CLASSES, "cohorts": details,
             "limitations": primary["limitations"] + supplement["protocol"]["limitations"],
             "source_sha256": sources}
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    lines = ["# Supporting dementia image-benchmark audit", "",
             "These are the frozen October 11 Round 1 results: two training conditions with three seeds each. "
             "The historical baseline is an additional model. They are distinct from older baseline training and later development rounds.", "",
             "All 27 model/cohort point estimates were recomputed from public prediction CSVs. "
             "Six ensemble tables were reconstructed by equal-weight averaging of case-aligned member probabilities. "
             "All 36 paired contrasts retain their frozen 2,000-resample conditional bootstrap intervals; "
             "point differences and discordant counts were verified. Intervals were preserved, not regenerated. "
             "No McNemar test was present in these source records and none was added.", "",
             "| Model | Matched 1,058 | Sensitivity 756 | Reused 100 |", "| --- | --- | --- | --- |"]
    lookup = {(r["cohort"], r["model"]): r for r in metric_rows}
    for model in MODELS:
        cells = [f"{lookup[c,model]['correct']}/{lookup[c,model]['n']} ({lookup[c,model]['accuracy']*100:.3f}%)" for c in COHORTS]
        lines.append("| " + " | ".join([DISPLAY[model]] + cells) + " |")
    lines += ["", "The matched cohort has 966 image-similarity groups; 302 images have an old training/validation group mate. "
              "The nested sensitivity cohort retains 733 groups after excluding those 302 images. "
              "Hash groups do not establish patient identity or independence. Original100 has historical old-model exposure of "
              "66 training, 9 validation and 25 test images. It remains a reused descriptive benchmark.", "",
              "The stage ensemble corrects the baseline's one error: +0.095 percentage points on matched images "
              "(conditional 95% interval 0.000 to 0.290) and +0.132 on sensitivity images (0.000 to 0.399). "
              "Both intervals include zero. Perfect observed predictions and degenerate bootstrap intervals do not establish certainty. "
              "These results do not establish clinical superiority or readiness.", "",
              "`metrics.csv` includes accuracy, balanced accuracy, macro F1, raw numerators and preserved intervals. "
              "`paired_differences.csv` includes all paired estimates, source fields and discordant counts. "
              "`condition_summaries.csv` reports means and sample SDs for three seeds; those SDs are not confidence intervals. "
              "`audit.json` records confusion matrices, source hashes and limitations.", "",
              "Reproduce from the repository root: `python -m analysis.sts_results.supporting --output-dir results/sts/supporting`."]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")
    if make_plot:
        draw_plot(output_dir, metric_rows)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("results/sts/supporting"))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    audit = run(args.repo_root, args.output_dir, make_plot=not args.no_plot)
    print(json.dumps({k: audit[k] for k in ("status", "cohort_model_rows", "paired_contrasts_verified")}))


if __name__ == "__main__":
    main()
