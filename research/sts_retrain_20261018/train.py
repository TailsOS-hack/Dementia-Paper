"""Retrain the six-run design, logging clean eval-mode train NLL every epoch.

No historical output is changed. Test/external images are never evaluated here.
The original helper is hash-verified, and optimization and checkpoint-selection
rules are retained. Added diagnostics save/restore RNG states explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import importlib
import json
import math
from pathlib import Path
import random
import time

STUDY = "sts_retrain_20261018"
CODE_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATH = CODE_ROOT / "docs/kevin_paper_20261011/training/train_candidate.py"
LEGACY_SHA = "cbf67b7c6e842ce35eeef2e0c0a6a0752b865cec3399d7aa58bd5e8ff0e2a5e2"
STAGE_SHA = "66a0cd9409d61edd32ee137154f769c749664f255ddebe461cf7683c1f4ba7b9"
CLINICAL_SHA = "9dc70cad44f40f06dd028fe5e90790d656cf142ce4fb1d6f2f828646719951c1"
DEFAULT_STAGE = "training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv"
DEFAULT_CLINICAL = "data/external/clinical_t1_preprocessed_v1/clinical_slices.csv"
METRIC_DEFINITIONS = {
    "train_loss": "Unweighted unsmoothed NLL: all training images, clean transform, model.eval(), fixed end-epoch weights",
    "val_loss": "Unweighted unsmoothed NLL: all validation images, same clean transform and model.eval(), same end-epoch weights",
    "train_online_nll": "Unweighted unsmoothed NLL during optimization: augmented images, dropout/BatchNorm train mode, evolving weights",
    "train_stage_objective": "Per-draw mean of group/class-weighted label-smoothed stage cross entropy used for backpropagation",
    "train_total_objective_per_step": "Mean of actual stage-plus-scheduled-auxiliary scalar objectives across optimizer steps; excludes decoupled AdamW decay",
    "train_aux_online_nll": "Unsmoothed NLL on class-balanced sampled augmented clinical slices in train mode",
    "train_aux_objective": "Per-sampled-slice mean of smoothed auxiliary cross entropy before aux_weight",
    "train_aux_eval_nll": "Clean fixed-checkpoint NLL after averaging slice probabilities per training subject",
    "val_aux_nll": "Clean fixed-checkpoint NLL after averaging slice probabilities per validation subject",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_torch_threads(torch, count):
    """Avoid CPU transform oversubscription; this setting is pinned on resume."""
    require(isinstance(count, int) and count > 0, "torch_threads must be a positive integer")
    torch.set_num_threads(count)
    actual = torch.get_num_threads()
    require(actual == count, "PyTorch did not apply the requested CPU thread count")
    return actual


def load_legacy():
    require(hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest() == LEGACY_SHA,
            "Historical helper changed; refuse an unreviewed training implementation")
    # A real importable module name is required when DataLoader uses spawn.
    # An anonymous spec name cannot be imported by worker subprocesses.
    module = importlib.import_module("docs.kevin_paper_20261011.training.train_candidate")
    require(Path(module.__file__).resolve() == LEGACY_PATH.resolve(), "Historical helper resolved outside the reviewed source")
    return module


def validate_output(path, resume=False):
    path = Path(path).resolve()
    require(any(part.startswith(STUDY) for part in path.parts),
            f"Output must be inside a new {STUDY} directory")
    require(not path.exists() or resume, "Output already exists; use a new directory or guarded --resume")
    if resume:
        config_path = path / "config.json"
        require(config_path.is_file(), "Resume requires this study's existing config.json")
        require(json.loads(config_path.read_text()).get("study") == STUDY,
                "Refusing resume from a historical or different experiment")
        require((path / "last.pt").is_file(), "Resume requires last.pt in the same output")
    return path


class SampleMean:
    """Batch means are weighted by their sample count, including short batches."""
    def __init__(self):
        self.total = 0.0
        self.n = 0

    def add(self, value, n):
        require(n > 0 and math.isfinite(float(value)), "Invalid loss accumulation")
        self.total += float(value) * n
        self.n += n

    @property
    def value(self):
        require(self.n > 0, "No samples accumulated")
        return self.total / self.n


def stage_weights(rows):
    group_counts = Counter(r["group_id"] for r in rows)
    group_labels = {}
    for row in rows:
        key, label = row["group_id"], row["label"]
        require(key not in group_labels or group_labels[key] == label, "Mixed-label training group")
        group_labels[key] = label
    class_groups = Counter(group_labels.values())
    weights = [1 / (group_counts[r["group_id"]] * class_groups[r["label"]]) for r in rows]
    scale = len(weights) / sum(weights)
    return [v * scale for v in weights]


@contextmanager
def preserved_rng(torch, np):
    python_state = random.getstate()
    numpy_state = copy.deepcopy(np.random.get_state())
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps = getattr(torch, "mps", None)
    mps_available = getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)()
    mps_state = mps.get_rng_state().clone() if mps_available and hasattr(mps, "get_rng_state") else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            mps.set_rng_state(mps_state)


def select_checkpoint(validation_loss, best, min_delta):
    """Only clean validation NLL can improve the checkpoint selection state."""
    return validation_loss < best - min_delta


def validate_checkpoint_history(checkpoint):
    """Treat last.pt as the epoch commit and verify its selection bookkeeping."""
    history, config = checkpoint["history"], checkpoint["config"]
    epoch = checkpoint["epoch"]
    require(history and [row["epoch"] for row in history] == list(range(1, epoch + 1)),
            "Checkpoint history is incomplete or noncontiguous")
    best, selected, bad = float("inf"), None, 0
    for row in history:
        for field in ("train_loss", "val_loss"):
            require(math.isfinite(row[field]) and row[field] >= 0, "Invalid checkpoint loss history")
        if select_checkpoint(row["val_loss"], best, config["min_delta"]):
            best, selected, bad = row["val_loss"], row["epoch"], 0
        else:
            bad += 1
        require(row["epoch"] == epoch or not (row["epoch"] >= config["min_epochs"] and bad >= config["patience"]),
                "Checkpoint history extends past its stopping epoch")
    require(math.isclose(best, checkpoint["best_val_loss"], rel_tol=0, abs_tol=1e-12), "Checkpoint best NLL disagrees with history")
    require(checkpoint["bad_epochs"] == bad, "Checkpoint patience disagrees with history")
    require(checkpoint["selected_epoch"] == selected, "Checkpoint selected epoch disagrees with history")
    require(math.isclose(checkpoint["stage_validation"]["loss"], history[-1]["val_loss"], rel_tol=0, abs_tol=1e-12),
            "Last validation summary disagrees with history")
    return selected


def atomic_json(c, path, value):
    temporary = path.with_name(path.name + ".tmp")
    c.dump(temporary, value)
    temporary.replace(path)


def synchronize_run_artifacts(c, output, checkpoint, *, render=True):
    """Recover any interrupted post-checkpoint writes from the canonical last.pt.

    If the last epoch improved selection, last.pt contains the complete selected
    model and can reconstruct best.pt. Otherwise it pins the previous best's
    exact bytes, which must still exist before resuming or declaring completion.
    """
    selected = validate_checkpoint_history(checkpoint)
    if selected == checkpoint["epoch"]:
        c.atomic_save(checkpoint, output / "best.pt")
    else:
        expected = checkpoint.get("best_checkpoint_sha256")
        require(expected and (output / "best.pt").is_file() and c.sha(output / "best.pt") == expected,
                "Previously selected checkpoint missing or changed; cannot recover it from a later last.pt")
    atomic_json(c, output / "selection.json", dict(study=STUDY, selected_epoch=selected,
                val_loss=checkpoint["best_val_loss"], reason=checkpoint["config"]["selection"],
                selected_checkpoint_sha256=c.sha(output / "best.pt"), test_used_for_selection=False))
    atomic_json(c, output / "history.json", checkpoint["history"])
    temporary = output / "history.csv.tmp"
    c.csv_write(temporary, checkpoint["history"]); temporary.replace(output / "history.csv")
    atomic_json(c, output / f"epoch_{checkpoint['epoch']:03d}_evaluation.json", dict(
                train_stage=checkpoint["train_clean"], val_stage=checkpoint["stage_validation"],
                train_aux=checkpoint["train_aux_clean"], val_aux=checkpoint["aux_validation"]))
    if render:
        plot_curves(output, checkpoint["history"], checkpoint["config"]["mode"])


def write_completion(c, output, checkpoint):
    artifacts = ("selection.json", "history.json", "history.csv", "learning_curves.png", "learning_curves.pdf")
    atomic_json(c, output / "run_complete.json", dict(study=STUDY, status="completed", epochs=checkpoint["epoch"],
                selected_epoch=checkpoint["selected_epoch"], best_validation_nll=checkpoint["best_val_loss"],
                heldout_inference_performed=False, best_sha256=c.sha(output / "best.pt"), last_sha256=c.sha(output / "last.pt"),
                artifact_sha256={name: c.sha(output / name) for name in artifacts}))


def plot_curves(output, history, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Clean train, eval mode")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="Clean validation, eval mode")
    axes[0].set_title("Comparable stage NLL at each epoch end")
    axes[1].plot(epochs, [r["train_online_nll"] for r in history], color="#777777", label="Online train NLL")
    axes[1].set_title("Augmented training, evolving parameters")
    axes[2].plot(epochs, [r["train_stage_objective"] for r in history], label="Weighted smoothed stage objective")
    if mode == "multitask":
        axes[2].plot(epochs, [r["train_total_objective_per_step"] for r in history], label="Total objective / optimizer step")
    axes[2].set_title("Optimization objective (different definition)")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("New October 18 retraining · raw observations · no held-out evaluation")
    fig.tight_layout(); fig.savefig(output / "learning_curves.png", dpi=180); fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)


def train(args):
    out = validate_output(args.output, resume=args.resume)
    c = load_legacy(); c.imports()
    torch, np, nn = c.torch, c.np, c.nn
    actual_threads = configure_torch_threads(torch, args.torch_threads)
    root = args.data_root.resolve()
    stage_path = c.resolve(root, args.stage_manifest)
    require(c.sha(stage_path) == STAGE_SHA, "Stage manifest differs from the original frozen design")
    if args.mode == "multitask":
        require(args.clinical_manifest, "Multitask training requires clinical data")
        require(len(args.clinical_manifest_sha256) == 64 and
                all(c in "0123456789abcdef" for c in args.clinical_manifest_sha256), "Invalid expected clinical manifest digest")
        require(c.sha(c.resolve(root, args.clinical_manifest)) == args.clinical_manifest_sha256, "Clinical manifest changed")
    stage = c.csv_read(stage_path)
    for row in stage:
        row["label"] = int(row["label"])
    for split in ("train", "val", "test"):
        ids = {r["group_id"] for r in stage if r["split"] == split}
        for other in ("train", "val", "test"):
            require(split == other or not ids & {r["group_id"] for r in stage if r["split"] == other},
                    "Legacy image-similarity group crosses splits")
    tr, va = ([r for r in stage if r["split"] == split] for split in ("train", "val"))
    require(len(tr) == 19023 and len(va) == 2705, "Unexpected frozen training/validation denominators")
    require(set(r["label"] for r in tr) == set(r["label"] for r in va) == set(range(4)), "Missing stage class")
    clinical = c.clinical_rows(root, args.clinical_manifest, args.clinical_image_root) if args.mode == "multitask" else []
    atr, ava = ([r for r in clinical if r["split"] == split and r["label"] >= 0] for split in ("train", "val"))
    if args.mode == "multitask":
        require(atr and ava, "Missing clinical training/validation subjects")
    out.mkdir(parents=True, exist_ok=args.resume)
    device = c.device_for(args.device); c.seed_all(args.seed)
    model, aux = c.network(pretrained=not args.resume, weights_file=None if args.resume else args.imagenet_weights)
    model.to(device); aux.to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + (list(aux.parameters()) if args.mode == "multitask" else []),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * .05)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(study=STUDY, source_helper_sha256=LEGACY_SHA, trainer_sha256=c.sha(Path(__file__)),
                  stage_manifest_sha256=STAGE_SHA, clinical_manifest_sha256=args.clinical_manifest_sha256 if clinical else None,
                  original_clinical_manifest_sha256=CLINICAL_SHA,
                  torch=torch.__version__, device_actual=str(device), torch_threads_actual=actual_threads,
                  selection="minimum clean stage validation NLL with min_delta",
                  train_rows=len(tr), validation_rows=len(va), test_used_for_selection=False, heldout_inference_performed=False,
                  initialization="ImageNet-1K V1, never a previous study checkpoint", metric_definitions=METRIC_DEFINITIONS,
                  training_aux_subjects=len({r["subject_id"] for r in atr}), validation_aux_subjects=len({r["subject_id"] for r in ava}))
    start, best, bad, history, selected_epoch = 0, float("inf"), 0, [], None
    if args.resume:
        checkpoint = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        require(checkpoint.get("study") == STUDY, "Wrong checkpoint study")
        for key in ("mode", "seed", "stage_manifest_sha256", "clinical_manifest_sha256", "image_size", "batch_size",
                    "aux_weight", "aux_every", "learning_rate", "weight_decay", "epochs", "label_smoothing",
                    "min_delta", "min_epochs", "patience", "workers", "torch_threads", "torch_threads_actual",
                    "clinical_image_root", "source_helper_sha256", "trainer_sha256"):
            require(checkpoint["config"].get(key) == config.get(key), f"Resume config mismatch: {key}")
        model.load_state_dict(checkpoint["model_state"]); aux.load_state_dict(checkpoint["aux_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"]); scheduler.load_state_dict(checkpoint["scheduler_state"])
        start, best, bad, history = checkpoint["epoch"], checkpoint["best_val_loss"], checkpoint["bad_epochs"], checkpoint["history"]
        selected_epoch = validate_checkpoint_history(checkpoint)
        synchronize_run_artifacts(c, out, checkpoint)
    atomic_json(c, out / "config.json", config)
    weights = stage_weights(tr)
    aux_by_subject = defaultdict(list)
    for row in atr:
        aux_by_subject[row["subject_id"]].append(row)
    aux_by_label = {label: [sid for sid, rows in aux_by_subject.items() if rows[0]["label"] == label] for label in (0, 1)}
    require(args.mode != "multitask" or all(aux_by_label.values()), "Clinical training requires both classes")
    dataset = c.dataset(tr, root, True, args.image_size)
    if start >= args.epochs or (start >= args.min_epochs and bad >= args.patience):
        write_completion(c, out, checkpoint)
        return
    for epoch in range(start, args.epochs):
        c.seed_all(args.seed + epoch); started = time.time(); model.train(); aux.train()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = c.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", generator=generator)
        online, stage_objective, aux_online, aux_objective = (SampleMean() for _ in range(4))
        total_objective, steps, stage_correct, aux_correct = 0., 0, 0, 0
        for step, (x, y, indices) in enumerate(loader):
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
            logits = c.forward(model, aux, x, "stage")
            ww = torch.tensor([weights[i] for i in indices.tolist()], device=device)
            objective = (nn.functional.cross_entropy(logits, y, label_smoothing=args.label_smoothing, reduction="none") * ww).mean()
            loss = objective
            stage_objective.add(objective.detach().item(), len(y))
            online.add(nn.functional.cross_entropy(logits.detach(), y).item(), len(y))
            stage_correct += int((logits.detach().argmax(1) == y).sum().item())
            if args.mode == "multitask" and step % args.aux_every == 0:
                chosen = []
                for k in range(args.batch_size):
                    sid = random.choice(aux_by_label[k % 2]); chosen.append(random.choice(aux_by_subject[sid]))
                ads = c.dataset(chosen, root, True, args.image_size)
                xx = torch.stack([ads[i][0] for i in range(len(chosen))]).to(device)
                yy = torch.tensor([r["label"] for r in chosen], device=device)
                alog = c.forward(model, aux, xx, "aux")
                aobj = nn.functional.cross_entropy(alog, yy, label_smoothing=args.label_smoothing)
                loss = loss + args.aux_weight * aobj
                aux_objective.add(aobj.detach().item(), len(yy))
                aux_online.add(nn.functional.cross_entropy(alog.detach(), yy).item(), len(yy))
                aux_correct += int((alog.detach().argmax(1) == yy).sum().item())
            total_objective += loss.detach().item(); steps += 1
            loss.backward(); nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux.parameters()), 5.)
            optimizer.step()
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, batch {step+1}/{len(loader)}", flush=True)
        # Original validation happens first. The additional clean training pass is
        # diagnostic only and cannot affect future optimization RNG or selection.
        vm, _ = c.evaluate(model, aux, va, root, device, "stage", args)
        am, _ = c.evaluate(model, aux, ava, root, device, "aux", args) if clinical else ({}, [])
        with preserved_rng(torch, np):
            tm, _ = c.evaluate(model, aux, tr, root, device, "stage", args)
            atm, _ = c.evaluate(model, aux, atr, root, device, "aux", args) if clinical else ({}, [])
        row = dict(epoch=epoch+1, train_loss=tm["loss"], val_loss=vm["loss"], train_online_nll=online.value,
                   train_stage_objective=stage_objective.value, train_total_objective_per_step=total_objective/steps,
                   train_eval_accuracy=tm["accuracy"], val_accuracy=vm["accuracy"], train_online_accuracy=stage_correct/online.n,
                   stage_draws=online.n, aux_draws=aux_online.n, optimizer_steps=steps,
                   learning_rate=optimizer.param_groups[0]["lr"], seconds=time.time()-started)
        if clinical:
            row.update(train_aux_online_nll=aux_online.value, train_aux_objective=aux_objective.value,
                       train_aux_eval_nll=atm["loss"], val_aux_nll=am["loss"],
                       train_aux_online_accuracy=aux_correct/aux_online.n,
                       train_aux_eval_accuracy=atm["accuracy"], val_aux_accuracy=am["accuracy"])
        history.append(row)
        improved = select_checkpoint(vm["loss"], best, args.min_delta)
        if improved:
            best, bad = vm["loss"], 0
            selected_epoch = epoch + 1
        else:
            bad += 1
        scheduler.step()
        checkpoint = dict(study=STUDY, task="dementia", arch="mobilenet_v3_large", class_names=c.STAGES,
                          image_size=args.image_size, model_state=model.state_dict(), aux_state=aux.state_dict(),
                          aux_classes=["CN", "AD"], optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                          epoch=epoch+1, best_val_loss=best, selected_epoch=selected_epoch,
                          best_checkpoint_sha256=None if improved else c.sha(out / "best.pt"),
                          bad_epochs=bad, history=history, config=config,
                          rng_state={"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()},
                          stage_validation=vm, aux_validation=am, train_clean=tm, train_aux_clean=atm, smoke_only=False)
        c.atomic_save(checkpoint, out / "last.pt")
        synchronize_run_artifacts(c, out, checkpoint)
        print(json.dumps(row), flush=True)
        if epoch+1 >= args.min_epochs and bad >= args.patience:
            break
    write_completion(c, out, checkpoint)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage-manifest", default=DEFAULT_STAGE)
    p.add_argument("--clinical-manifest", default=None)
    p.add_argument("--clinical-manifest-sha256", default=CLINICAL_SHA,
                   help="Expected SHA256; a minimized train/val manifest requires an explicit new digest and membership audit")
    p.add_argument("--clinical-image-root", default="data")
    p.add_argument("--mode", choices=("stage_only", "multitask"), required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1,
                   help="Positive CPU intra-op thread count, frozen for the run (default: 1)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.001)
    p.add_argument("--label-smoothing", type=float, default=.05)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--aux-weight", type=float, default=.25)
    p.add_argument("--aux-every", type=int, default=4)
    p.add_argument("--imagenet-weights", type=Path)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
