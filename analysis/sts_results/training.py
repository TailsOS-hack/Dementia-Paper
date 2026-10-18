"""Audit the six October 11 Round 1 runs without retraining or loading weights.

The 305 completed epochs belong to two conditions with three seeds each. Earlier
legacy dementia histories are inventoried separately and are never pooled here.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PUBLICATION = Path("docs/kevin_paper_20261011/publication")
CHECKPOINTS = Path("models/kevin_20261011/CHECKPOINTS.json")
TRAINER = Path("docs/kevin_paper_20261011/training/train_candidate.py")
LEGACY = Path("paper_package/Dementia_Only_2026-10-09/source")


def _json(path: Path):
    return json.loads(path.read_text())


def _source(root: Path, relative: Path) -> dict:
    data = (root / relative).read_bytes()
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def stopping_defaults(path: Path) -> dict:
    """Read the published trainer's numeric CLI defaults without executing it."""
    wanted = {"--epochs": "maximum_epochs", "--min-epochs": "minimum_epochs",
              "--patience": "patience", "--min-delta": "minimum_improvement"}
    defaults = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flag = node.args[0]
        if isinstance(flag, ast.Constant) and flag.value in wanted:
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[wanted[flag.value]] = ast.literal_eval(keyword.value)
    if len(defaults) != len(wanted):
        raise ValueError("Cannot recover all four stopping defaults from published trainer")
    return defaults


def replay_selection(rows: list[dict], *, minimum_improvement: float = 0.0001,
                     minimum_epochs: int = 20, patience: int = 15,
                     maximum_epochs: int = 100) -> dict:
    """Replay the trainer's strict NLL improvement and patience rules.

    This rejects missing/duplicated epochs and histories continuing after the
    first stopping epoch. It intentionally does not equate selection to argmin.
    """
    if not rows or [int(row["epoch"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Training epochs must be contiguous, unique and start at one")
    if minimum_improvement < 0 or patience < 1 or minimum_epochs < 1 or maximum_epochs < minimum_epochs:
        raise ValueError("Invalid stopping controls")
    losses = [float(row["val_stage_loss"]) for row in rows]
    if any(not math.isfinite(loss) or loss < 0 for loss in losses):
        raise ValueError("Validation NLL must be finite and nonnegative")
    best, selected, bad, stop = math.inf, None, 0, None
    for row, loss in zip(rows, losses):
        epoch = int(row["epoch"])
        if stop is not None:
            raise ValueError("History continues beyond its first stopping epoch")
        if loss < best - minimum_improvement:
            best, selected, bad = loss, epoch, 0
        else:
            bad += 1
        if (epoch >= minimum_epochs and bad >= patience) or epoch >= maximum_epochs:
            stop = epoch
    return {"selected_epoch": selected, "selected_validation_nll": best,
            "completed_epochs": len(rows), "first_stopping_epoch": stop,
            "epochs_since_selected": bad,
            "mathematical_minimum_epoch": losses.index(min(losses)) + 1,
            "mathematical_minimum_nll": min(losses),
            "selection_to_minimum_nll_gap": best - min(losses)}


def audit_legacy_lineage(root: Path) -> dict:
    """Inventory legacy histories; do not infer checkpoint ancestry from names."""
    primary = LEGACY / "training_logs/Dementia_Specialist_history.json"
    duplicate = LEGACY / "models/Collab_Train/trained_models/training_logs/Dementia_Specialist_history.json"
    histories = [("Legacy dementia specialist", primary),
                 ("Archived Collab_Train dementia specialist", duplicate)]
    for family, run in (("experiments_regularized", "20260511_221548"),
                        ("experiments_dedup_regularized", "20260512_022515"),
                        ("experiments_perceptual_regularized", "20260513_163515")):
        histories.append((family, LEGACY / f"training_logs/{family}/dementia/{run}/history.json"))
    entries = []
    for label, relative in histories:
        history = _json(root / relative)
        if isinstance(history, dict):
            lengths = {len(history[key]) for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
            if len(lengths) != 1:
                raise ValueError(f"Unequal legacy history lengths: {relative}")
            epochs = lengths.pop()
        else:
            epochs = len(history)
            if [row["epoch"] for row in history] != list(range(1, epochs + 1)):
                raise ValueError(f"Noncontiguous legacy history: {relative}")
        entries.append({"label": label, "completed_epochs": epochs,
                        "source": _source(root, relative), "seed": None,
                        "selected_epoch": None,
                        "checkpoint_lineage": "Not established by this history alone"})
    primary_data, duplicate_data = (root / primary).read_bytes(), (root / duplicate).read_bytes()
    return {
        "scope": "Historical dementia histories predating the October 11 Round 1 study",
        "histories": entries,
        "same_filename_comparison": {"source": _source(root, duplicate),
                                     "compared_with": primary.as_posix(),
                                     "identical": primary_data == duplicate_data},
        "limitations": [
            "These five history artifacts are separate from the six October 11 runs and are not included in the 305-epoch total.",
            "History length is observed; seed, validation selection and exact baseline-checkpoint ancestry are not established by these history files alone.",
            "Two files named Dementia_Specialist_history.json have different contents (32 and 5 epochs); their shared filename does not establish common run or checkpoint identity.",
            "No six-architecture or eighteen-run original STS experiment is established by the supplied six-run evidence.",
        ],
    }


def audit_training(root: Path) -> dict:
    root = Path(root)
    records = _json(root / CHECKPOINTS)["checkpoints"]
    selection_path = PUBLICATION / "results/validation_selection.json"
    evidence_path = PUBLICATION / "evidence/completed_methods_evidence.json"
    selection, evidence = _json(root / selection_path), _json(root / evidence_path)
    controls = stopping_defaults(root / TRAINER)
    if not selection["frozen_before_heldout_evaluation"]:
        raise ValueError("Selection is not recorded as frozen before evaluation")
    pairs = [(record["condition"], record["seed"]) for record in records]
    protocol = selection["analysis_protocol"]
    expected = {(mode, seed) for mode in protocol["planned_modes"] for seed in protocol["planned_seeds"]}
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("Checkpoint inventory differs from the planned condition-by-seed design")
    sources = [_source(root, p) for p in (CHECKPOINTS, selection_path, evidence_path, TRAINER,
                                        PUBLICATION / "reports/Completed_Methods.md")]
    evidence_runs = {row["candidate"]: row for row in evidence["completed_runs"]}
    result = []
    for record in sorted(records, key=lambda row: int(row["candidate"].split("_")[-1])):
        candidate, condition, seed = record["candidate"], record["condition"], record["seed"]
        frozen, original = selection["candidates"][candidate], evidence_runs[candidate]
        path = PUBLICATION / f"training/{condition}_seed{seed}/history.csv"
        with (root / path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        replay = replay_selection(rows, **controls)
        if not (record["selected_epoch"] == frozen["epoch"] == original["selected_epoch"] == replay["selected_epoch"]):
            raise ValueError(f"Selected-epoch disagreement for {candidate}")
        if not (record["sha256"] == frozen["checkpoint_sha256"] == original["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint-metadata hash disagreement for {candidate}")
        if frozen["mode"] != condition or frozen["seed"] != seed:
            raise ValueError(f"Frozen condition or seed disagreement for {candidate}")
        for value in (record["validation_nll"], frozen["stage_validation_nll"], original["validation_nll"]):
            if not math.isclose(value, replay["selected_validation_nll"], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Selected NLL disagreement for {candidate}")
        if replay["first_stopping_epoch"] != replay["completed_epochs"] or original["completed_epochs"] != len(rows):
            raise ValueError(f"Completed-epoch disagreement for {candidate}")
        source = _source(root, path)
        sources.append(source)
        result.append({"candidate": candidate, "condition": condition, "seed": seed,
                       "architecture": "MobileNetV3-Large", "initialization": "ImageNet",
                       **replay, "checkpoint_metadata_sha256": record["sha256"],
                       "history_path": path.as_posix(), "history_sha256": source["sha256"],
                       "historical_source_history_sha256": original["history_sha256"],
                       "original_loss_figure": path.with_name("loss_curves.png").as_posix(),
                       "original_accuracy_figure": path.with_name("accuracy_curves.png").as_posix()})
    total = sum(run["completed_epochs"] for run in result)
    if total != evidence["total_completed_training_epochs"]:
        raise ValueError("Computed total does not match completed-methods evidence")
    means = {mode: sum(run["selected_validation_nll"] for run in result if run["condition"] == mode)
             / sum(run["condition"] == mode for run in result) for mode in protocol["planned_modes"]}
    preferred = min(means, key=lambda mode: (means[mode], mode))
    representative = min((run for run in result if run["condition"] == preferred),
                         key=lambda run: (run["selected_validation_nll"], run["seed"]))["candidate"]
    if preferred != selection["preferred_mode"] or representative != selection["exploratory_representative_candidate"]:
        raise ValueError("Recomputed validation preference differs from the frozen selection")
    for mode, mean in means.items():
        if not math.isclose(mean, selection["mode_mean_stage_validation_nll"][mode], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Mean validation NLL disagreement for {mode}")
    return {
        "study": "October 11, 2026 Round 1 dementia training experiment",
        "design": "Two training conditions, three shared seed values per condition; six fitted checkpoints total",
        "architecture": "ImageNet-initialized MobileNetV3-Large in both conditions",
        "run_count": len(result), "conditions": dict(Counter(row["condition"] for row in result)),
        "distinct_seed_count": len(set(row["seed"] for row in result)),
        "total_completed_epochs": total, "stopping_controls": controls,
        "selection_rule": "Strict improvement: current stage-validation NLL < selected NLL - minimum_improvement",
        "selection_frozen_utc": selection["created_utc"],
        "validation_preferred_condition": selection["preferred_mode"],
        "mean_selected_validation_nll_by_condition": means,
        "exploratory_representative": selection["exploratory_representative_candidate"],
        "runs": result, "legacy_lineage": audit_legacy_lineage(root), "sources": sources,
        "limitations": [
            "The 305 epochs describe the October 11 Round 1 experiment, not a documented original six-architecture STS training experiment.",
            "There are six condition-by-seed runs, not eighteen runs. The same three seed values occur in both conditions.",
            "Stopping controls are read from the published trainer defaults, corroborated by Completed_Methods.md and exact replay of all six histories; raw run configurations are not in this portable training directory.",
            "Checkpoint hashes are cross-checked between metadata records; model binaries are not loaded or independently rehashed by this audit.",
            "Public CSV byte hashes and historical original-history hashes refer to different exported files and are retained separately.",
            "Stage train loss is unweighted unsmoothed cross-entropy with augmentation/dropout active; optimized loss includes weights and smoothing, while validation is evaluated without training augmentation.",
            "The multitask condition adds a distinct AD/control head; its clinical endpoint and sampled-slice training loss must not be merged with the four-class stage endpoint.",
        ],
    }


def _plot(root: Path, output_dir: Path, summary: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 220})
    runs, figures = summary["runs"], []
    fig, ax = plt.subplots(figsize=(9, 4.9))
    x = list(range(len(runs)))
    ax.bar([i - .19 for i in x], [r["completed_epochs"] for r in runs], .38,
           color="#24466B", label="Completed epochs")
    ax.bar([i + .19 for i in x], [r["selected_epoch"] for r in runs], .38,
           color="#D49339", label="Validation-selected epoch")
    for i, run in enumerate(runs):
        ax.text(i-.19, run["completed_epochs"]+1, str(run["completed_epochs"]), ha="center")
        ax.text(i+.19, run["selected_epoch"]+1, str(run["selected_epoch"]), ha="center")
    ax.set_xticks(x, [f"{r['candidate'].replace('candidate_', 'C')} · {r['condition'].replace('_', ' ')}\nSeed {r['seed']}" for r in runs])
    ax.set_ylabel("Epoch")
    ax.set_ylim(0, max(r["completed_epochs"] for r in runs)+13)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("October 11 Round 1 training: six runs, 305 completed epochs", loc="left", pad=18)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    fig.text(.01, .01, "Two conditions × three seed values; one MobileNetV3-Large backbone family. Selection used validation NLL.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ("png", "pdf"):
        name = f"completed_vs_selected_epochs.{extension}"
        fig.savefig(output_dir / name)
        figures.append({"file": name, "kind": "epoch_summary"})
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True)
    for ax, run in zip(axes.flat, sorted(runs, key=lambda r: (r["condition"] != "stage_only", r["seed"]))):
        with (root / run["history_path"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        ax.plot(epochs, [float(row["train_stage_loss"]) for row in rows], color="#24466B", label="Stage train loss")
        ax.plot(epochs, [float(row["val_stage_loss"]) for row in rows], color="#D49339", label="Stage validation NLL")
        ax.axvline(run["selected_epoch"], color="#5A6069", linestyle="--", linewidth=1, label="Selected epoch")
        ax.set_title(f"{run['candidate'].replace('candidate_', 'C')} · {run['condition'].replace('_', ' ')} · {run['seed']}", fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel("Completed epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Raw stage loss (log scale)")
    fig.suptitle("October 11 Round 1 training: raw stage loss histories", x=.02, ha="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .055), ncols=3, frameon=False)
    fig.text(.02, .015, "No smoothing. Training includes augmentation/dropout; logged stage loss is unsmoothed cross-entropy.\nTraining loss differs from the weighted, label-smoothed optimization objective; clinical loss is not plotted.", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .94))
    for extension in ("png", "pdf"):
        name = f"stage_loss_curves.{extension}"
        fig.savefig(output_dir / name)
        figures.append({"file": name, "kind": "raw_stage_loss"})
    plt.close(fig)
    return figures


def generate(root: Path, output_dir: Path) -> dict:
    root, output_dir = Path(root), Path(output_dir)
    summary = audit_training(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["figures"] = _plot(root, output_dir, summary)
    with (output_dir / "training_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["runs"][0]))
        writer.writeheader()
        writer.writerows(summary["runs"])
    for name, data in (("training_summary.json", summary), ("legacy_training_lineage.json", summary["legacy_lineage"])):
        (output_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path("results/sts/training"))
    args = parser.parse_args()
    summary = generate(args.root, args.output_dir)
    print(json.dumps({"runs": summary["run_count"], "completed_epochs": summary["total_completed_epochs"],
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
