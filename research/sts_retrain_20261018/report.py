"""Validate and plot a completed new six-run retraining suite, without inference.

Checkpoint binaries are optional for paper-only rebuilds; available binaries are
always SHA256-verified. Metadata and split-manifest checks are reported separately
from binary checks. Synthetic fixtures require explicit opt-in and are labelled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .train import CODE_ROOT, STUDY, LEGACY_SHA, STAGE_SHA, METRIC_DEFINITIONS, require
from .suite import MODES, SEEDS, validation_preference

EXPECTED_NAMES = tuple(f"{mode}_seed{seed}" for seed in SEEDS for mode in MODES)
EXPECTED_CONFIG = dict(epochs=100, min_epochs=20, patience=15, min_delta=1e-4,
                       batch_size=32, image_size=224, learning_rate=1e-4,
                       weight_decay=.001, label_smoothing=.05, aux_weight=.25, aux_every=4)
TRAIN_N, VAL_N = 19023, 2705
COMPLETION_ARTIFACTS = frozenset(("selection.json", "history.json", "history.csv", "learning_curves.png", "learning_curves.pdf"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def close(a, b, context):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-11), f"{context}: {a} != {b}")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def replay_selection(history, config):
    require([int(r["epoch"]) for r in history] == list(range(1, len(history)+1)),
            "Epochs must be contiguous and unique, starting at 1")
    best, selected, bad, stop = math.inf, None, 0, None
    for row in history:
        epoch, loss = int(row["epoch"]), float(row["val_loss"])
        require(math.isfinite(loss) and loss >= 0, "Invalid validation NLL")
        require(stop is None, "History continues after first stopping epoch")
        if loss < best - config["min_delta"]:
            best, selected, bad = loss, epoch, 0
        else:
            bad += 1
        if (epoch >= config["min_epochs"] and bad >= config["patience"]) or epoch >= config["epochs"]:
            stop = epoch
    require(stop == len(history), "Completed history does not reach its stopping rule")
    return dict(selected_epoch=selected, val_loss=best, completed_epochs=len(history), first_stopping_epoch=stop,
                mathematical_minimum_epoch=min(history, key=lambda r: r["val_loss"])["epoch"])


def check_clean_metrics(record, n, loss, accuracy, label):
    require(record["n"] == n, f"{label}: incorrect clean evaluation denominator")
    require(loss >= 0 and 0 <= accuracy <= 1, f"{label}: invalid metric value")
    close(record["loss"], loss, label + " NLL")
    close(record["accuracy"], accuracy, label + " accuracy")
    matrix = record["confusion_matrix"]
    require(len(matrix) == (4 if "stage" in label else 2) and all(len(r) == len(matrix) for r in matrix),
            f"{label}: invalid confusion matrix shape")
    require(all(isinstance(x, int) and x >= 0 for row in matrix for x in row), f"{label}: invalid confusion count")
    require(sum(map(sum, matrix)) == n, f"{label}: confusion denominator mismatch")
    close(sum(matrix[i][i] for i in range(len(matrix)))/n, accuracy, label + " confusion accuracy")
    require(record.get("unit") == ("legacy_image" if "stage" in label else "subject"), f"{label}: incorrect unit")


def verify_manifests(stage_manifest=None, clinical_manifest=None, clinical_digest=None):
    result = {"stage": {"status": "metadata_identity_only", "sha256": STAGE_SHA},
              "clinical": {"status": "metadata_identity_only", "sha256": clinical_digest}}
    if stage_manifest:
        require(sha(stage_manifest) == STAGE_SHA, "Stage manifest identity mismatch")
        rows = read_csv(stage_manifest)
        require(len({r["path"] for r in rows}) == len(rows), "Duplicate stage image ID")
        groups, counts = {}, {}
        for split, n in (("train", TRAIN_N), ("val", VAL_N), ("test", 5450)):
            selected = [r for r in rows if r["split"] == split]
            require(len(selected) == n, f"Stage {split} count mismatch")
            groups[split] = {r["group_id"] for r in selected}; counts[split] = n
        require(not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]),
                "Stage image-similarity group crosses splits")
        result["stage"].update(status="manifest_ids_and_groups_verified", split_counts=counts,
                                image_pixels_read=False, patient_independence_established=False)
    if clinical_manifest:
        require(sha(clinical_manifest) == clinical_digest, "Clinical manifest identity mismatch")
        rows = read_csv(clinical_manifest); selected = [r for r in rows if r["split"] in ("train", "val")]
        require(len({r.get("image_path") or r["path"] for r in selected}) == len(selected), "Duplicate clinical image ID")
        groups = {}; split_counts = {}; subject_counts = {}
        for split, count, subjects in (("train", 288, 32), ("val", 99, 11)):
            subset = [r for r in selected if r["split"] == split]
            require(len(subset) == count, f"Clinical {split} slice count mismatch")
            ids = set()
            for row in subset:
                cohort = row.get("dataset") or row.get("cohort")
                identity = (row.get("group_key") or row.get("subject_key") or
                            cohort+":"+(row.get("subject_id") or row["source_subject_id"]))
                label = {"AD": "1", "CN": "0"}.get(row["diagnosis"])
                require(label is not None and str(row["binary_label"]) == label, "Clinical diagnosis/label mismatch")
                require(identity not in groups or groups[identity] == (split, label), "Clinical subject crosses split or label")
                groups[identity] = (split, label); ids.add(identity)
            require(len(ids) == subjects, f"Clinical {split} subject count mismatch")
            split_counts[split], subject_counts[split] = count, subjects
        result["clinical"].update(status="manifest_ids_and_groups_verified", slice_counts=split_counts,
                                   subject_counts=subject_counts, image_pixels_read=False)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def figures(output, runs, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prefix = "SYNTHETIC TEST FIXTURE — NOT RESEARCH RESULTS\n" if synthetic else ""

    def save(fig, name):
        fig.savefig(output / (name+".png"), dpi=200)
        fig.savefig(output / (name+".pdf"))
        plt.close(fig)

    for kind in ("matched_clean_nll", "online_and_objectives"):
        fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), sharey=True)
        for i, mode in enumerate(MODES):
            for j, seed in enumerate(SEEDS):
                run = runs[f"{mode}_seed{seed}"]; hist = run["history"]; epochs = [r["epoch"] for r in hist]
                ax = axes[i,j]
                if kind == "matched_clean_nll":
                    ax.plot(epochs, [r["train_loss"] for r in hist], label="Clean training", color="#0072B2")
                    ax.plot(epochs, [r["val_loss"] for r in hist], label="Clean validation", color="#D55E00")
                else:
                    ax.plot(epochs, [r["train_online_nll"] for r in hist], label="Online train NLL", color="#666666")
                    ax.plot(epochs, [r["train_stage_objective"] for r in hist], label="Stage objective / draw", color="#009E73")
                    if mode == "multitask":
                        ax.plot(epochs, [r["train_total_objective_per_step"] for r in hist], label="Total objective / step", color="#CC79A7")
                selected = run["summary"]["selected_epoch"]
                ax.axvline(selected, color="#333333", linestyle="--", linewidth=1, label=f"Selected epoch {selected}")
                ax.set_title(f"{'Stage only' if mode=='stage_only' else 'Multitask'} · seed {seed}")
                ax.set_xlabel("Epoch"); ax.set_ylabel("NLL" if kind == "matched_clean_nll" else "Loss / stated denominator")
                ax.grid(alpha=.18); ax.legend(fontsize=8)
        fields = ("train_loss", "val_loss") if kind == "matched_clean_nll" else (
            "train_online_nll", "train_stage_objective", "train_total_objective_per_step")
        ymax=max(row[field] for item in runs.values() for row in item["history"] for field in fields)
        for ax in axes.flat:
            ax.set_ylim(0, max(ymax*1.05, 1e-6))
        fig.suptitle(prefix + ("New retraining: comparable clean training and validation NLL" if kind == "matched_clean_nll" else
                              "New retraining: online NLL and separately defined optimization objectives"), fontsize=15)
        note = ("Same end-epoch weights, clean transform and evaluation mode; unweighted, unsmoothed NLL. "
                "Training is resubstitution. Dashed lines: validation-only selections." if kind == "matched_clean_nll" else
                "Online NLL: augmented draws and evolving weights. Objectives include sample weights/smoothing; "
                "multitask total includes scheduled auxiliary loss. Definitions differ.")
        fig.text(.02, .024, note, fontsize=8.5)
        fig.tight_layout(rect=(0, .055, 1, .93 if synthetic else .95)); save(fig, kind)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        run = runs[f"multitask_seed{seed}"]; hist=run["history"]
        epochs = [r["epoch"] for r in hist]
        ax.plot(epochs, [r["train_aux_eval_nll"] for r in hist], label="Clean training subjects", color="#0072B2")
        ax.plot(epochs, [r["val_aux_nll"] for r in hist], label="Clean validation subjects", color="#D55E00")
        ax.axvline(run["summary"]["selected_epoch"], color="#333333", linestyle="--", linewidth=1, label="Stage-selected epoch")
        ax.set(title=f"Multitask · seed {seed}", xlabel="Epoch", ylabel="Subject NLL")
        ax.grid(alpha=.18); ax.legend(fontsize=8)
    ymax=max(row[field] for name,item in runs.items() if name.startswith("multitask")
             for row in item["history"] for field in ("train_aux_eval_nll", "val_aux_nll"))
    for ax in axes:
        ax.set_ylim(0,max(ymax*1.05,1e-6))
    fig.suptitle(prefix+"Clinical auxiliary head: comparable clean subject-level NLL", fontsize=15)
    fig.text(.02, .035, "Slice probabilities averaged per subject before scoring; 32 training and 11 validation subjects. "
             "Auxiliary performance never selects checkpoints. No held-out evaluation.", fontsize=8.5)
    fig.tight_layout(rect=(0, .08, 1, .88 if synthetic else .92)); save(fig, "clinical_clean_nll")


def run(input_root, output_dir, make_plots=True, require_checkpoints=False,
        stage_manifest=None, clinical_manifest=None, allow_synthetic=False, expected_trainer_sha256=None):
    root, output = Path(input_root).resolve(), Path(output_dir).resolve()
    require(root != output and root not in output.parents, "Report output must be outside the immutable input suite")
    sources = {}

    def source(relative):
        path = root / relative
        require(path.is_file(), f"Missing source file: {relative}")
        sources[str(relative)] = sha(path)
        return path

    def read(relative):
        return json.loads(source(relative).read_text())

    suite = read("suite_status.json")
    synthetic = suite.get("synthetic_fixture") is True
    require(not synthetic or allow_synthetic, "Synthetic fixture requires explicit allow_synthetic")
    require(suite.get("study") == STUDY and suite.get("state") == "all_six_completed_no_heldout_evaluation", "Suite is not completed")
    require(suite.get("heldout_inference_performed") is False, "Unexpected held-out evaluation flag")
    require(suite.get("source_helper_sha256") == LEGACY_SHA, "Different historical helper identity")
    require(suite.get("completed") == list(EXPECTED_NAMES), "Suite must contain all six runs in the original order")
    identity = suite["input_identity"]
    trainer_reference = expected_trainer_sha256 or sha(Path(__file__).with_name("train.py"))
    require(isinstance(trainer_reference, str) and len(trainer_reference) == 64 and
            all(c in "0123456789abcdef" for c in trainer_reference), "Invalid expected trainer digest")
    require(identity.get("trainer_sha256") == trainer_reference, "Trainer identity does not match expected reviewed source")
    runs, summary_rows, epoch_rows, selections, binary_checks, artifact_checks = {}, [], [], {}, [], []
    for name in EXPECTED_NAMES:
        mode, seed = name.rsplit("_seed", 1); seed = int(seed)
        config, complete, selection = (read(Path(name)/filename) for filename in ("config.json", "run_complete.json", "selection.json"))
        require(config.get("study") == complete.get("study") == selection.get("study") == STUDY, f"{name}: wrong study")
        require(config.get("mode") == mode and config.get("seed") == seed, f"{name}: condition/seed mismatch")
        require(complete.get("status") == "completed" and complete.get("heldout_inference_performed") is False, f"{name}: incomplete")
        require(config.get("test_used_for_selection") is False and selection.get("test_used_for_selection") is False and
                config.get("heldout_inference_performed") is False, f"{name}: selection/evaluation flags changed")
        for key, expected in EXPECTED_CONFIG.items():
            close(config[key], expected, f"{name} configured {key}")
        require(config.get("source_helper_sha256") == LEGACY_SHA and config.get("stage_manifest_sha256") == STAGE_SHA,
                f"{name}: source or stage split identity mismatch")
        require(config.get("trainer_sha256") == identity["trainer_sha256"], f"{name}: trainer identity differs within suite")
        require(config.get("torch_threads") == config.get("torch_threads_actual") == identity.get("torch_threads") == 1,
                f"{name}: CPU thread setting differs from the reviewed one-thread suite")
        require(config.get("train_rows") == TRAIN_N and config.get("validation_rows") == VAL_N, f"{name}: split denominators changed")
        require(config.get("metric_definitions") == METRIC_DEFINITIONS, f"{name}: metric definitions differ")
        if mode == "multitask":
            require(config.get("clinical_manifest_sha256") == identity["clinical_manifest_sha256"], f"{name}: clinical split identity changed")
            require(config.get("training_aux_subjects") == 32 and config.get("validation_aux_subjects") == 11,
                    f"{name}: clinical subject counts changed")
        else:
            require(config.get("clinical_manifest_sha256") is None and config.get("training_aux_subjects") == 0 and
                    config.get("validation_aux_subjects") == 0, f"{name}: unexpected clinical training")
        history = read(Path(name)/"history.json")
        csv_history = read_csv(source(Path(name)/"history.csv"))
        require(len(history) == len(csv_history) == complete["epochs"], f"{name}: history length mismatch")
        for row, csv_row in zip(history, csv_history):
            require(set(row) == set(csv_row), f"{name}: CSV/JSON history schema mismatch")
            for key, value in row.items():
                close(value, csv_row[key], f"{name}: CSV/JSON {key}")
            require(row["stage_draws"] == TRAIN_N and row["optimizer_steps"] == 595, f"{name}: incorrect stage draws/steps")
            require(row["aux_draws"] == (4768 if mode == "multitask" else 0), f"{name}: incorrect clinical draws")
            for key in ("train_loss", "val_loss", "train_online_nll", "train_stage_objective", "train_total_objective_per_step"):
                require(math.isfinite(row[key]) and row[key] >= 0, f"{name}: invalid {key}")
            require(all(0 <= row[key] <= 1 for key in ("train_eval_accuracy", "val_accuracy", "train_online_accuracy")),
                    f"{name}: invalid accuracy")
            evaluation = read(Path(name)/f"epoch_{row['epoch']:03d}_evaluation.json")
            check_clean_metrics(evaluation["train_stage"], TRAIN_N, row["train_loss"], row["train_eval_accuracy"], name+" train stage")
            check_clean_metrics(evaluation["val_stage"], VAL_N, row["val_loss"], row["val_accuracy"], name+" val stage")
            if mode == "multitask":
                check_clean_metrics(evaluation["train_aux"], 32, row["train_aux_eval_nll"], row["train_aux_eval_accuracy"], name+" train aux")
                check_clean_metrics(evaluation["val_aux"], 11, row["val_aux_nll"], row["val_aux_accuracy"], name+" val aux")
            else:
                require(not evaluation["train_aux"] and not evaluation["val_aux"], f"{name}: unexpected clinical evaluation")
            epoch_rows.append(dict(run=name, mode=mode, seed=seed, **row))
        replay = replay_selection(history, config)
        require(selection["selected_epoch"] == replay["selected_epoch"], f"{name}: checkpoint epoch disagrees with stopping replay")
        close(selection["val_loss"], replay["val_loss"], f"{name}: selected loss")
        close(complete["best_validation_nll"], replay["val_loss"], f"{name}: completion best loss")
        require(complete.get("selected_epoch") == selection["selected_epoch"], f"{name}: missing or mismatched completion selection")
        require(selection.get("selected_checkpoint_sha256") == complete["best_sha256"], f"{name}: missing or mismatched selected checkpoint hash")
        completion_artifacts = complete.get("artifact_sha256")
        require(isinstance(completion_artifacts, dict) and set(completion_artifacts) == COMPLETION_ARTIFACTS,
                f"{name}: completion must pin all five expected artifacts")
        for filename, expected in completion_artifacts.items():
            require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                    f"Invalid completion artifact digest: {filename}")
            relative = Path(name)/filename
            if (root/relative).is_file():
                source(relative)
                require(sources[str(relative)] == expected, f"Completion artifact hash mismatch: {relative}")
                status="artifact_sha256_verified"
            else:
                # Raw per-run plots can be omitted from a paper-only rebuild;
                # the required history/selection files have already been read.
                require(filename.startswith("learning_curves."), f"Missing completion artifact: {relative}")
                status="metadata_only_optional_raw_plot_absent"
            artifact_checks.append(dict(path=str(relative), expected_sha256=expected, status=status))
        require(suite["records"][name] == selection, f"{name}: suite selection record differs")
        selections[name] = selection
        for checkpoint in ("best", "last"):
            relative = Path(name)/(checkpoint+".pt"); expected = complete[checkpoint+"_sha256"]
            require(len(expected) == 64 and all(c in "0123456789abcdef" for c in expected), "Invalid checkpoint digest")
            exists = (root/relative).is_file()
            require(exists or not require_checkpoints, f"Missing required checkpoint: {relative}")
            if exists:
                source(relative)
                require(sources[str(relative)] == expected, f"Checkpoint hash mismatch: {relative}")
            binary_checks.append(dict(path=str(relative), expected_sha256=expected,
                                      status="binary_sha256_verified" if exists else "metadata_only_binary_absent"))
        selected = history[selection["selected_epoch"]-1]
        summary = dict(run=name, mode=mode, seed=seed, completed_epochs=len(history),
                       selected_epoch=selection["selected_epoch"], selected_train_nll=selected["train_loss"],
                       selected_validation_nll=selected["val_loss"],
                       selected_validation_minus_train_nll=selected["val_loss"]-selected["train_loss"],
                       selected_online_train_nll=selected["train_online_nll"],
                       selected_stage_objective=selected["train_stage_objective"],
                       selected_train_accuracy=selected["train_eval_accuracy"], selected_validation_accuracy=selected["val_accuracy"],
                       final_train_nll=history[-1]["train_loss"], final_validation_nll=history[-1]["val_loss"],
                       train_images=TRAIN_N, validation_images=VAL_N,
                       mathematical_minimum_epoch=replay["mathematical_minimum_epoch"])
        summary_rows.append(summary); runs[name] = dict(history=history, summary=summary, stopping=replay)
    preference = validation_preference(selections)
    require(read("validation_only_preference.json") == preference and suite["validation_only_preference"] == preference,
            "Suite condition preference does not match validation-only recomputation")
    if stage_manifest is None:
        candidate = CODE_ROOT/"docs/kevin_paper_20261011/publication/prepared/stage_manifest.csv"
        stage_manifest = candidate if candidate.is_file() else None
    split_checks = verify_manifests(stage_manifest, clinical_manifest, identity["clinical_manifest_sha256"])
    condition_rows = []
    for mode in MODES:
        rows = [r for r in summary_rows if r["mode"] == mode]
        for metric in ("selected_train_nll", "selected_validation_nll", "selected_validation_minus_train_nll"):
            values = [r[metric] for r in rows]
            condition_rows.append(dict(condition=mode, metric=metric, seeds=3, mean=mean(values), sample_sd=stdev(values),
                                       minimum=min(values), maximum=max(values), sd_is_confidence_interval=False))
    verified = sum(r["status"] == "binary_sha256_verified" for r in binary_checks)
    report = dict(study=STUDY, status="verified_synthetic_fixture" if synthetic else "verified_completed_retraining",
                  synthetic_fixture=synthetic, completed_runs=6, total_epochs=sum(r["completed_epochs"] for r in summary_rows),
                  checkpoint_verification="all_12_binaries_verified" if verified==12 else "metadata_only" if verified==0 else "partial_binary_verification",
                  checkpoint_binaries_verified=verified, checkpoint_checks=binary_checks, split_checks=split_checks,
                  completion_artifact_checks=artifact_checks,
                  source_helper_sha256=LEGACY_SHA, trainer_sha256=identity["trainer_sha256"],
                  trainer_verification={"status": "expected_source_sha256_verified", "expected_sha256": trainer_reference,
                                        "reference": "explicit_archived_source_digest" if expected_trainer_sha256 else "local_train.py"},
                  hyperparameters=EXPECTED_CONFIG.copy(),
                  runtime={"torch_threads": 1},
                  denominators={"stage_train_images": TRAIN_N, "stage_validation_images": VAL_N,
                                "clinical_train_subjects": 32, "clinical_validation_subjects": 11,
                                "clinical_train_slices": 288, "clinical_validation_slices": 99},
                  metric_definitions=METRIC_DEFINITIONS, validation_only_preference=preference, runs=summary_rows,
                  source_sha256=sources, image_pixels_read=False, new_training=False, new_inference=False,
                  heldout_performance_estimated=False,
                  limitations=["New retraining is separate from the historical 305-epoch experiment.",
                               "Clean training NLL is resubstitution; validation selected checkpoints and is not an independent test estimate.",
                               "No held-out performance, clinical superiority or readiness claim follows from these learning curves.",
                               "Image-similarity groups do not establish patient independence; clinical data remain small.",
                               "Source hashes verify retrieved artifacts, not GPU runtime behavior or the scientific equivalence of a reduced manifest.",
                               "Absent checkpoint binaries are explicitly metadata-only; no checkpoint deserialization occurs."])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"summary.csv", summary_rows); write_csv(output/"condition_summary.csv", condition_rows)
    write_csv(output/"epochs.csv", epoch_rows)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    if make_plots:
        figures(output, runs, synthetic)
    heading = "# SYNTHETIC TEST FIXTURE — NOT RESEARCH RESULTS" if synthetic else "# New retraining learning curves"
    lines = [heading, "", f"All six new October 18 runs completed and passed report validation: {report['total_epochs']} total epochs. "
             "This experiment is separate from the historical six-run study.", "",
             f"Checkpoint verification: **{report['checkpoint_verification']}** ({verified}/12 available binaries hash-verified). "
             "All histories, per-epoch evaluation denominators, stopping decisions and validation-only selections were checked. "
             "Missing binaries are not claimed as verified. No weights were loaded or inference run for this report.", "",
             "| Run | Completed epochs | Selected epoch | Clean train NLL | Clean validation NLL |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in summary_rows:
        lines.append(f"| {r['run']} | {r['completed_epochs']} | {r['selected_epoch']} | {r['selected_train_nll']:.6f} | {r['selected_validation_nll']:.6f} |")
    lines += ["", "`matched_clean_nll.png/.pdf` plots unsmoothed raw clean training and validation NLL for all six runs. "
              "Each pair uses the same epoch-end weights, transform, evaluation mode and unweighted unsmoothed NLL definition; "
              "dashed lines identify validation-selected epochs. `online_and_objectives.png/.pdf` separately labels online augmented NLL "
              "and weighted/smoothed objectives. `clinical_clean_nll.png/.pdf` compares 32 training and 11 validation subjects after "
              "averaging slice probabilities per subject. Clinical loss does not select checkpoints.", "",
              "`summary.csv` contains selected/final NLL and epoch counts; `epochs.csv` retains every logged observation; "
              "`condition_summary.csv` reports three-seed means and sample SDs (not confidence intervals). "
              "`report.json` records source hashes, split identity checks, per-checkpoint verification and definitions.", "",
              f"Stage split verification: `{split_checks['stage']['status']}`. Clinical split verification: `{split_checks['clinical']['status']}`. "
              "Manifest checks do not re-read image pixels or establish patient independence.", "",
              "Rebuild from the repository root:", "", "```sh",
              "python -m research.sts_retrain_20261018.report --input /path/to/retrieved/sts_retrain_20261018 --output results/sts/retraining",
              "```", "", "Add `--require-checkpoints` to require all 12 checkpoint binaries. "
              "Add `--clinical-manifest /path/to/audited/minimized.csv` to verify clinical membership/group separation against the recorded digest. "
              "The bundled public stage manifest is verified automatically when available; `--stage-manifest` can specify its path.", "",
              "Trainer identity is pinned to the local `train.py` by default. For an archived rebuild with a different reviewed trainer version, "
              "pass its independently verified digest using `--expected-trainer-sha256`; never copy an unverified digest solely from the suite metadata.", "",
              "Training is resubstitution and validation was used for selection. These curves contain no held-out performance evidence "
              "and do not establish clinical superiority or readiness. No smoothing, interpolation or manual adjustment was applied."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, default=Path("results/sts/retraining"))
    p.add_argument("--require-checkpoints", action="store_true"); p.add_argument("--stage-manifest", type=Path)
    p.add_argument("--clinical-manifest", type=Path); p.add_argument("--no-plots", action="store_true")
    p.add_argument("--expected-trainer-sha256", help="Independently reviewed trainer digest for an archived rebuild; defaults to local train.py")
    args=p.parse_args()
    result=run(args.input, args.output, make_plots=not args.no_plots, require_checkpoints=args.require_checkpoints,
               stage_manifest=args.stage_manifest, clinical_manifest=args.clinical_manifest,
               expected_trainer_sha256=args.expected_trainer_sha256)
    print(json.dumps({key:result[key] for key in ("status", "completed_runs", "total_epochs", "checkpoint_verification")}))


if __name__ == "__main__":
    main()
