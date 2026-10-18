"""Recover the six frozen candidates' full 5,450-image descriptive results.

This reads archived predictions, never runs inference, trains, or selects a model.
The full split was evaluated post hoc and contains the historical 1,058-image
primary comparison. Its presentation does not change that analysis chronology.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PUBLICATION = Path("docs/kevin_paper_20261011/publication")
CLASSES = ("MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented")
MODELS = tuple("candidate_%d" % i for i in range(1, 7))
MANIFEST_SHA256 = "66a0cd9409d61edd32ee137154f769c749664f255ddebe461cf7683c1f4ba7b9"
EXPECTED_SPLITS = {"train": 19023, "val": 2705, "test": 5450}
EXPECTED_CLASSES = dict(zip(CLASSES, (1342, 1538, 1324, 1246)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_digest(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError("Source SHA256 mismatch: %s" % path)
    return actual


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_predictions(rows, expected_labels, classes=CLASSES):
    """Validate exact membership, labels, probabilities and argmax; recompute metrics.

    expected_labels maps every expected image path to its supplied class label.
    No image paths or image-level probability rows are included in the result.
    """
    paths = [r["path"] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate prediction path")
    if set(paths) != set(expected_labels):
        raise ValueError("Prediction membership does not match the frozen test manifest")
    cm = [[0 for _ in classes] for _ in classes]
    total_loss = 0.0
    for row in rows:
        label = expected_labels[row["path"]]
        if label not in classes or row["TrueCategory"] != label:
            raise ValueError("Prediction truth label disagrees with the frozen manifest")
        probabilities = [float(row["P_" + label]) for label in classes]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities):
            raise ValueError("Invalid probability")
        probability_sum = sum(probabilities)
        if abs(probability_sum - 1) > 2e-6:
            raise ValueError("Probability vector does not sum to one")
        predicted_index = probabilities.index(max(probabilities))
        if row["ModelPrediction"] != classes[predicted_index]:
            raise ValueError("Prediction label disagrees with probability argmax")
        true_index = classes.index(label)
        cm[true_index][predicted_index] += 1
        total_loss -= math.log(max(probabilities[true_index] / probability_sum, 1e-15))
    n = len(rows)
    if not n:
        raise ValueError("Empty prediction cohort")
    per_class = []
    for index, label in enumerate(classes):
        support = sum(cm[index])
        if support == 0:
            raise ValueError("A required class is absent from the cohort")
        tp = cm[index][index]
        fp = sum(row[index] for row in cm) - tp
        fn = support - tp
        tn = n - tp - fp - fn
        per_class.append({
            "class": label, "support": support,
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn),
        })
    correct = sum(cm[i][i] for i in range(len(classes)))
    return {
        "n": n, "correct": correct, "errors": n - correct,
        "accuracy": correct / n,
        "balanced_accuracy": sum(r["recall"] for r in per_class) / len(classes),
        "macro_f1": sum(r["f1"] for r in per_class) / len(classes),
        "negative_log_likelihood": total_loss / n,
        "confusion_matrix": cm, "per_class": per_class,
    }


def generate(repo_root, output_dir=None):
    """Validate public source artifacts and return (optionally save) the report."""
    root = Path(repo_root).resolve()
    base = root / PUBLICATION
    source_files = {}

    def source(relative, pinned=None):
        path = base / relative
        digest = check_digest(path, pinned) if pinned else sha256(path)
        source_files[str(PUBLICATION / relative)] = digest
        return path

    manifest_path = source("prepared/stage_manifest.csv", MANIFEST_SHA256)
    manifest = read_csv(manifest_path)
    if len({r["path"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate paths in the frozen source manifest")
    if dict(Counter(r["split"] for r in manifest)) != EXPECTED_SPLITS:
        raise ValueError("Frozen source split counts changed")
    groups = defaultdict(set)
    for row in manifest:
        if row["subtype"] != CLASSES[int(row["label"])]:
            raise ValueError("Frozen manifest label encoding disagrees with class order")
        groups[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("A prepared similarity group spans candidate splits")
    test = [row for row in manifest if row["split"] == "test"]
    if dict(Counter(r["subtype"] for r in test)) != EXPECTED_CLASSES:
        raise ValueError("Frozen test class counts changed")
    expected_labels = {r["path"]: r["subtype"] for r in test}
    selection = json.loads(source("results/validation_selection.json").read_text())
    replay = json.loads(source("figures/clean_replay/replay_metadata.json").read_text())
    protocol = json.loads(source("figures/clean_replay/protocol_before_inference.json").read_text())
    verification = json.loads(source("evidence/clean_replay_verification.json").read_text())
    audit = json.loads(source("prepared/split_audit.json").read_text())
    complete = json.loads(source("figures/clean_replay/replay_complete.json").read_text())
    if complete["status"] != "completed_posthoc_clean_split_replay" or not complete["local_readback_passed"]:
        raise ValueError("Archived replay is not marked complete")
    for record in (selection, replay, protocol):
        if record["stage_manifest_sha256"] != MANIFEST_SHA256:
            raise ValueError("Frozen analysis uses a different stage manifest")
    if audit["manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Split audit uses a different stage manifest")
    if replay["class_order"] != list(CLASSES) or set(replay["checkpoints"]) != set(MODELS):
        raise ValueError("Unexpected archived class order or candidate set")
    if not selection["frozen_before_heldout_evaluation"]:
        raise ValueError("Selection was not recorded as frozen before held-out evaluation")
    if replay["checkpoint_reselection"] or replay["training"]:
        raise ValueError("Replay unexpectedly records training or checkpoint reselection")
    primary = read_csv(source("prepared/same_case_test_intersection.csv", audit["same_case_test_intersection_sha256"]))
    sensitivity = read_csv(source("prepared/old_hash_group_unexposed_sensitivity.csv",
                                  selection["old_hash_unexposed_sensitivity"]["manifest_sha256"]))
    primary_paths = {r["path"] for r in primary}
    sensitivity_paths = {r["path"] for r in sensitivity}
    if len(primary) != 1058 or len(primary_paths) != 1058 or primary_paths != {r["path"] for r in test if r["baseline_strict_split"] == "test"}:
        raise ValueError("Matched primary membership differs from the full-test historical-baseline intersection")
    if len(sensitivity) != 756 or len(sensitivity_paths) != 756 or not sensitivity_paths <= primary_paths:
        raise ValueError("Sensitivity cohort is not the expected subset of the primary cohort")
    for row in primary + sensitivity:
        if row["subtype"] != expected_labels[row["path"]]:
            raise ValueError("Overlapping cohort labels disagree")
    archived_checks = {r["model"]: r for r in verification["checks"] if r["split"] == "full_new_test"}
    models = []
    for name in MODELS:
        identity = selection["candidates"][name]
        checkpoint = replay["checkpoints"][name]
        if identity != replay["frozen_candidate_identities"][name] or identity != protocol["frozen_candidate_identities"][name]:
            raise ValueError("Frozen candidate identity mismatch: " + name)
        for key in ("checkpoint_sha256", "mode", "seed"):
            if identity[key] != checkpoint[key]:
                raise ValueError("Replayed checkpoint identity mismatch: " + name)
        if identity["epoch"] != checkpoint["selected_epoch"]:
            raise ValueError("Replayed checkpoint epoch mismatch: " + name)
        split = checkpoint["splits"]["full_new_test"]
        expected_filename = name + "_full_new_test_predictions.csv"
        if split["predictions_file"] != expected_filename:
            raise ValueError("Unexpected prediction artifact: " + name)
        prediction_file = source("figures/clean_replay/" + expected_filename, split["predictions_sha256"])
        metrics = audit_predictions(read_csv(prediction_file), expected_labels)
        for prior in (split["metrics"], archived_checks[name]):
            if prior["n"] != metrics["n"] or prior["confusion_matrix"] != metrics["confusion_matrix"]:
                raise ValueError("Recomputed denominator or confusion matrix differs: " + name)
            for key in ("accuracy", "balanced_accuracy", "macro_f1"):
                if abs(prior[key] - metrics[key]) > 1e-10:
                    raise ValueError("Recomputed metric differs: " + name + " " + key)
            if abs(prior["loss"] - metrics["negative_log_likelihood"]) > 1e-10:
                raise ValueError("Recomputed NLL differs: " + name)
        models.append({"model": name, "condition": identity["mode"], "seed": identity["seed"],
                       "selected_epoch": identity["epoch"], "checkpoint_sha256": identity["checkpoint_sha256"],
                       "prediction_file": str(prediction_file.relative_to(root)), **metrics})

    historical_split_counts = dict(Counter(r["baseline_strict_split"] for r in test))
    if historical_split_counts != {"train": 3860, "val": 532, "test": 1058}:
        raise ValueError("Historical baseline exposure counts changed")
    report = {
        "schema_version": 1,
        "status": "verified_from_archived_prediction_csvs",
        "cohort": "full_new_test", "analysis_role": "post_hoc_descriptive_overlapping_test_split",
        "n_images": len(test), "n_prepared_similarity_groups": len({r["group_id"] for r in test}),
        "class_order": list(CLASSES), "class_counts": EXPECTED_CLASSES,
        "split_image_counts": EXPECTED_SPLITS,
        "source_image_count": audit["source_rows"], "excluded_image_count": audit["excluded_rows"],
        "exclusion_counts": audit["exclusion_counts"],
        "candidate_hash_groups_disjoint_across_splits": True,
        "historical_baseline_split_counts_within_full_cohort": historical_split_counts,
        "primary_matched_comparison_n": 1058,
        "sensitivity_n": 756,
        "cohort_nesting_verified": "sensitivity756_subset_primary1058_subset_full5450",
        "validation_preferred_condition": selection["preferred_mode"],
        "validation_exploratory_representative": selection["exploratory_representative_candidate"],
        "models": models,
        "source_sha256": source_files,
        "generator_sha256": sha256(Path(__file__)),
        "interpretation": [
            "These six runs are the first six-candidate experiment archived under kevin_paper_20261011; they are not the earlier single historical baseline or later round-2 retraining.",
            "All 5,450 images belong to the frozen candidate test split. The clean full-split replay was post hoc and contains the historical primary 1,058-image comparison; it is not an additional independent cohort or a retrospectively prespecified primary analysis.",
            "The 5,450-image split contains 3,860 historical-baseline training and 532 validation rows. The old baseline must not be represented as independently held out on this full cohort.",
            "The legacy source was augmented upstream. Participant and original augmentation-family identifiers are unavailable. Similarity-group separation does not establish patient independence.",
            "The four supplied image labels do not establish a clinically validated CDR-stage mapping. These are descriptive image-level results, not clinical diagnostic validation.",
            "No model was trained, selected, calibrated or tuned during this extraction. Reported point estimates are recomputed from preserved CSV probabilities and labels; no new confidence intervals are claimed.",
        ],
    }
    if output_dir is not None:
        write_report(report, Path(output_dir))
    return report


def write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    performance_fields = ("model", "condition", "seed", "selected_epoch", "n", "correct", "errors",
                          "accuracy", "balanced_accuracy", "macro_f1", "negative_log_likelihood")

    def table(name, fieldnames, rows):
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    table("performance.csv", performance_fields, report["models"])
    per_class = [{"model": m["model"], **r} for m in report["models"] for r in m["per_class"]]
    table("per_class.csv", list(per_class[0]), per_class)
    matrices = [{"model": m["model"], "true_class": CLASSES[i],
                 **{label: row[j] for j, label in enumerate(CLASSES)}}
                for m in report["models"] for i, row in enumerate(m["confusion_matrix"])]
    table("confusion_matrices.csv", ["model", "true_class", *CLASSES], matrices)
    lines = ["# Full 5,450-image six-candidate benchmark", "",
             "Recomputed from all six archived prediction CSVs, with exact source-manifest and prediction SHA256 checks, frozen model identity checks, and independent agreement with the archived confusion matrices and metrics.", "",
             "| Candidate | Condition | Seed | Correct / images | Accuracy | Balanced accuracy | Macro-F1 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for model in report["models"]:
        lines.append("| {model} | {condition} | {seed} | {correct} / {n} | {accuracy:.4%} | {balanced_accuracy:.4%} | {macro_f1:.4%} |".format(**model))
    lines.extend(["", "The denominator is 5,450 images in 4,273 prepared similarity groups: 1,342 MildDemented, 1,538 ModerateDemented, 1,324 NonDemented and 1,246 VeryMildDemented.", ""])
    lines.extend("- " + note for note in report["interpretation"])
    lines.extend(["", "## Files", "", "- `metrics.json`: complete metrics, frozen identities, provenance hashes and limitations.",
                  "- `performance.csv`: six-model performance table.", "- `per_class.csv`: class support, recall, precision, specificity and F1.",
                  "- `confusion_matrices.csv`: true classes in rows, predicted classes in columns.", "",
                  "Regenerate from the repository root: `python3 -m analysis.sts_results.full_cohort`.", ""])
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = generate(args.root, args.output or args.root / "results/sts/full_cohort")
    print("Verified %d candidates, each on %d images." % (len(report["models"]), report["n_images"]))


if __name__ == "__main__":
    main()
