"""Render recorded learning curves using explicit endpoint and aggregation schemas.

No loss values are smoothed, rescaled, inferred or rewritten. Online training
measurements and epoch-end evaluation are labeled separately. Clinical slice
and subject losses never share a panel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FOLDERS = ("publication/training", "round2_publication/artifacts/stage/training",
           "round2_publication/artifacts/clinical/training")


def _series(field, label):
    return {"field": field, "label": label}


def _panel(title, endpoint, unit, metric, *series):
    return dict(title=title, endpoint=endpoint, unit=unit, metric=metric, series=list(series))


def curve_schema(headers):
    """Fail closed on unknown loss columns instead of overlaying by substring."""
    fields = set(headers)
    if {"train_stage_loss", "val_stage_loss"} <= fields:
        panels = [
            _panel("Four-class stage: image NLL", "stage", "image", "nll",
                   _series("train_stage_loss", "Train: online, augmented CE"),
                   _series("val_stage_loss", "Validation: epoch-end NLL")),
            _panel("Four-class stage: image accuracy", "stage", "image", "accuracy",
                   _series("train_stage_accuracy", "Train: online, augmented"),
                   _series("val_stage_accuracy", "Validation: epoch-end"),
                   _series("val_stage_balanced_accuracy", "Validation: balanced accuracy")),
        ]
        kind = "round1_stage"
        note = "Stage training logs unweighted, unsmoothed online CE with augmentation/dropout; validation uses epoch-end inference."
        if "train_aux_loss" in fields:
            kind = "round1_multitask"
            panels += [
                _panel("AD/control training: sampled-slice CE", "clinical", "sampled_slice", "nll",
                       _series("train_aux_loss", "Train: online, augmented slices")),
                _panel("AD/control training: sampled-slice accuracy", "clinical", "sampled_slice", "accuracy",
                       _series("train_aux_accuracy", "Train: online, augmented slices")),
                _panel("AD/control validation: subject NLL", "clinical", "subject_mean_probability", "nll",
                       _series("val_aux_loss", "Validation: mean slice probabilities")),
                _panel("AD/control validation: subject accuracy", "clinical", "subject_mean_probability", "accuracy",
                       _series("val_aux_accuracy", "Validation: subject accuracy"),
                       _series("val_aux_balanced_accuracy", "Validation: balanced accuracy")),
            ]
            note += "\nClinical train slices and validation subjects use different units; their losses are separated and do not define a matched generalization gap."
        note += ("\nRaw stage/clinical CE excludes weighting, label smoothing and the combined multitask objective."
                 if kind == "round1_multitask" else "\nRaw stage CE excludes the group/class weighting and label smoothing used by the optimization objective.")
    elif {"train_raw_nll", "clean_nll", "stress_nll", "train_weighted_smoothed_objective"} <= fields:
        kind = "round2_stage"
        panels = [
            _panel("Four-class stage: image NLL", "stage", "image", "nll",
                   _series("train_raw_nll", "Train: online, augmented CE"),
                   _series("clean_nll", "Validation: clean image NLL")),
            _panel("Four-class stage: image accuracy", "stage", "image", "accuracy",
                   _series("train_accuracy", "Train: online, augmented"),
                   _series("clean_accuracy", "Validation: clean accuracy"),
                   _series("clean_balanced_accuracy", "Validation: balanced accuracy")),
            _panel("Validation stress: family-mean NLL", "stage", "stress_family_mean", "nll",
                   _series("stress_nll", "Fixed stress: mean across families")),
            _panel("Validation stress: scenario-mean accuracy", "stage", "stress_scenario_mean", "accuracy",
                   _series("stress_accuracy", "Fixed stress: mean across scenarios")),
            _panel("Training optimization objective", "stage", "weighted_image", "objective",
                   _series("train_weighted_smoothed_objective", "Weighted CE with label smoothing")),
            _panel("Validation-only selection score", "stage", "selection_composite", "score",
                   _series("score", "0.5 clean NLL + 0.5 stress-family NLL")),
        ]
        note = ("Raw image NLL, stress aggregates, weighted/smoothed training objective and selection score occupy distinct panels.\n"
                "Epoch 0 is the initialization evaluation; online training metrics begin at epoch 1. No missing epoch-0 training values are invented.")
    elif {"train_aug_loss", "train_clean_loss", "val_loss"} <= fields:
        kind = "round2_clinical"
        panels = [
            _panel("AD/control: matched clean subject NLL", "clinical", "subject_mean_logits", "nll",
                   _series("train_clean_loss", "Train: epoch-end, clean"),
                   _series("val_loss", "Validation: epoch-end, clean")),
            _panel("AD/control: clean subject accuracy", "clinical", "subject_mean_logits", "accuracy",
                   _series("train_clean_accuracy", "Train: epoch-end, clean"),
                   _series("val_accuracy", "Validation: epoch-end, clean"),
                   _series("val_balanced_accuracy", "Validation: balanced accuracy")),
            _panel("AD/control: online augmented subject CE", "clinical", "subject_mean_logits", "nll",
                   _series("train_aug_loss", "Train: online, augmented subject bags")),
            _panel("AD/control: online augmented accuracy", "clinical", "subject_mean_logits", "accuracy",
                   _series("train_aug_accuracy", "Train: online, augmented subject bags")),
        ]
        note = ("All losses use subject bags with mean slice logits. Clean train and validation are evaluated at the same epoch-end weights.\n"
                "Online augmented training measurements are separated. Epoch 0 is initialization; online metrics begin at epoch 1.")
    else:
        raise ValueError("Unsupported training history schema; define endpoint and unit before plotting")
    used = {series["field"] for panel in panels for series in panel["series"]}
    missing = used - fields
    if missing:
        raise ValueError(f"Incomplete {kind} history: {sorted(missing)}")
    metric_fields = {field for field in fields if any(term in field for term in ("loss", "nll", "accuracy", "objective"))}
    if metric_fields - used:
        raise ValueError(f"Unclassified metric fields need explicit units: {sorted(metric_fields - used)}")
    return dict(kind=kind, panels=panels, note=note,
                smoothing=False, missing_values_imputed=False, raw_histories_modified=False)


def validate_history(rows, schema):
    if not rows:
        raise ValueError("History is empty")
    epochs = [int(row["epoch"]) for row in rows]
    start = 1 if schema["kind"].startswith("round1") else 0
    if epochs != list(range(start, start + len(rows))):
        raise ValueError("History epochs must be contiguous and unique from the expected initial epoch")
    for panel in schema["panels"]:
        for series in panel["series"]:
            field = series["field"]
            for row in rows:
                raw = row.get(field)
                if raw in (None, "", "None", "null"):
                    if int(row["epoch"]) == 0 and field in {"train_raw_nll", "train_weighted_smoothed_objective", "train_accuracy", "clean_balanced_accuracy", "train_aug_loss", "train_aug_accuracy"}:
                        continue
                    raise ValueError(f"Missing {field} at epoch {row['epoch']}")
                value = float(raw)
                if not math.isfinite(value) or value < 0 or (panel["metric"] == "accuracy" and value > 1):
                    raise ValueError(f"Invalid {field} at epoch {row['epoch']}")


def render_history(rows, schema, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    validate_history(rows, schema)
    panels = schema["panels"]
    fig, axes = plt.subplots(len(panels)//2, 2, figsize=(13, 3.0 * (len(panels)//2) + 1.7), squeeze=False)
    # Share loss scales within each endpoint, including the separate clinical
    # slice/subject panels, so separate panels do not disguise their magnitude.
    maxima = {}
    for panel in panels:
        group = (panel["endpoint"], panel["metric"])
        values = [float(row[series["field"]]) for series in panel["series"] for row in rows
                  if row.get(series["field"]) not in (None, "", "None", "null")]
        maxima[group] = max(maxima.get(group, 0), max(values))
    for ax, panel in zip(axes.flat, panels):
        for series in panel["series"]:
            values = [(int(row["epoch"]), float(row[series["field"]])) for row in rows
                      if row.get(series["field"]) not in (None, "", "None", "null")]
            field = series["field"]
            color = "#24466B" if field.startswith("train") else "#6A5A83" if "balanced" in field or field.startswith("stress") else "#D49339"
            ax.plot(*zip(*values), label=series["label"], color=color,
                    linewidth=1.6, linestyle="--" if "balanced" in series["field"] else "-")
        ax.set_title(panel["title"], loc="left", fontsize=11)
        ax.set_xlabel("Epoch")
        ylabel = {"accuracy": "Accuracy", "objective": "Objective (natural-log units)", "score": "Score (natural-log units)"}.get(panel["metric"], "Loss (natural-log units)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1 if panel["metric"] == "accuracy" else max(.05, maxima[panel["endpoint"], panel["metric"]] * 1.08))
        ax.set_xlim(int(rows[0]["epoch"]), int(rows[-1]["epoch"]))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(title + "\nRecorded learning curves: separate endpoints and aggregation units", fontsize=13, y=.99)
    note = schema["note"] + "\nEvery recorded value is shown without smoothing; source histories and frozen checkpoints are unchanged."
    fig.text(.055, .012, note, fontsize=8.5, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, .11 if len(panels)>2 else .22, 1, .94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def regenerate(root: Path, output: Path):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index, semantics = [], []
    for folder in FOLDERS:
        for path in sorted((root / "docs/kevin_paper_20261011" / folder).rglob("history.csv")):
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            schema = curve_schema(rows[0])
            name = folder.split("/")[0] + "_" + ("clinical_" if "clinical/" in folder else "") + path.parent.name
            round_label = "Round 1" if folder.startswith("publication/") else "Round 2"
            display = path.parent.name.replace("_seed", " · seed ").replace("_", " ")
            render_history(rows, schema, output / (name + ".png"), f"{round_label} · {display}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise RuntimeError("Source history changed during rendering")
            entry = dict(source=str(path.relative_to(root)), sha256=before, rows=len(rows),
                         figure=name + ".png", schema=schema["kind"], epochs=[int(rows[0]["epoch"]), int(rows[-1]["epoch"])])
            index.append(entry)
            semantics.append(dict(**entry, **{k:v for k,v in schema.items() if k!="kind"}))
    if len(index) != 12:
        raise ValueError(f"Expected 12 retained histories, found {len(index)}")
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (output / "loss_semantics.json").write_text(json.dumps(dict(
        status="historical_curves_replotted_from_unchanged_values", histories=semantics,
        training_performed=False, common_clean_train_validation_available="Round 2 clinical only; unavailable for historical Round 1 epochs",
        repaired_issue="Prior substring plotting mixed stage and clinical losses, mixed clinical slice/subject units, and omitted the weighted/smoothed objective field."), indent=2) + "\n")
    return len(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/current/training"))
    args = parser.parse_args()
    print(json.dumps({"histories_rendered": regenerate(args.root, args.output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
