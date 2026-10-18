"""Launch all six fresh runs sequentially, using only validation for preference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .train import (STUDY, CODE_ROOT, LEGACY_SHA, STAGE_SHA, CLINICAL_SHA, DEFAULT_STAGE, DEFAULT_CLINICAL,
                    load_legacy, require, validate_output)

SEEDS = (20261011, 20261012, 20261013)
MODES = ("stage_only", "multitask")


def build_command(args, mode, seed):
    command = [sys.executable, "-m", "research.sts_retrain_20261018.train", "--data-root", str(args.data_root.resolve()),
               "--output", str(args.output.resolve() / f"{mode}_seed{seed}"), "--stage-manifest", args.stage_manifest,
               "--mode", mode, "--seed", str(seed), "--device", args.device, "--workers", str(args.workers),
               "--torch-threads", str(args.torch_threads),
               "--epochs", "100", "--min-epochs", "20", "--patience", "15", "--min-delta", "0.0001"]
    if mode == "multitask":
        command += ["--clinical-manifest", args.clinical_manifest, "--clinical-image-root", args.clinical_image_root,
                    "--clinical-manifest-sha256", args.clinical_manifest_sha256]
    if args.imagenet_weights:
        command += ["--imagenet-weights", str(args.imagenet_weights.resolve())]
    if args.resume and (args.output / f"{mode}_seed{seed}" / "last.pt").is_file():
        command += ["--resume"]
    return command


def validation_preference(records):
    require(len(records) == 6, "All six runs are required for condition preference")
    means = {mode: sum(records[f"{mode}_seed{seed}"]["val_loss"] for seed in SEEDS)/3 for mode in MODES}
    condition = min(MODES, key=lambda mode: (means[mode], mode))
    representative = min((f"{condition}_seed{seed}" for seed in SEEDS), key=lambda name: records[name]["val_loss"])
    return dict(condition=condition, representative=representative, condition_mean_validation_nll=means,
                selection_basis="Mean selected stage-validation NLL; lowest seed within preferred condition",
                test_used_for_selection=False, all_six_runs_reported=True)


def recover_checkpoint_free_start(out, name, state):
    """Archive a verified first-epoch interruption; never discard saved weights."""
    run = out / name
    if not run.exists() or (run / "last.pt").is_file():
        return None
    require(state.get("current") == name, "Checkpoint-free directory is not the interrupted suite run")
    allowed = {"config.json", "config.json.tmp", "last.tmp"}
    require(all(path.is_file() and path.name in allowed for path in run.iterdir()),
            "Unexpected artifacts in checkpoint-free run; refusing automatic restart")
    config_path = run / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        require(config.get("study") == STUDY and f"{config.get('mode')}_seed{config.get('seed')}" == name,
                "Checkpoint-free directory belongs to another run")
        require(config.get("trainer_sha256") == state["input_identity"]["trainer_sha256"], "Interrupted trainer identity changed")
    else:
        require({path.name for path in run.iterdir()} <= {"config.json.tmp"},
                "Interrupted run contains files without a valid study configuration")
    archive_root = out / "recovery_without_checkpoint"
    archive_root.mkdir(exist_ok=True)
    index = 1
    archive = archive_root / f"{name}_attempt{index}"
    while archive.exists():
        index += 1
        archive = archive_root / f"{name}_attempt{index}"
    run.rename(archive)
    state.setdefault("checkpoint_free_restarts", []).append(str(archive.relative_to(out)))
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage-manifest", default=DEFAULT_STAGE)
    p.add_argument("--clinical-manifest", default=DEFAULT_CLINICAL)
    p.add_argument("--clinical-manifest-sha256", default=CLINICAL_SHA)
    p.add_argument("--clinical-image-root", default="data")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--imagenet-weights", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(); c = load_legacy()
    require(args.torch_threads > 0, "torch_threads must be a positive integer")
    out = args.output.resolve()
    require(c.sha(c.resolve(args.data_root.resolve(), args.stage_manifest)) == STAGE_SHA, "Stage manifest changed")
    require(c.sha(c.resolve(args.data_root.resolve(), args.clinical_manifest)) == args.clinical_manifest_sha256,
            "Clinical manifest changed")
    identity = dict(data_root=str(args.data_root.resolve()), stage_manifest=args.stage_manifest,
                    clinical_manifest=args.clinical_manifest, clinical_manifest_sha256=args.clinical_manifest_sha256,
                    clinical_image_root=args.clinical_image_root,
                    workers=args.workers, torch_threads=args.torch_threads,
                    trainer_sha256=c.sha(Path(__file__).with_name("train.py")))
    require(any(part.startswith(STUDY) for part in out.parts), f"Use a new {STUDY} output directory")
    if args.resume:
        state = json.loads((out / "suite_status.json").read_text())
        require(state["study"] == STUDY and state["source_helper_sha256"] == LEGACY_SHA, "Wrong suite resume")
        require(state["data_root"] == str(args.data_root.resolve()), "Suite data root changed")
        require(state.get("input_identity") == identity, "Suite input identity changed")
    else:
        require(not out.exists(), "Suite output already exists; inspect or use guarded --resume")
        out.mkdir(parents=True)
        state = dict(study=STUDY, source_helper_sha256=LEGACY_SHA, data_root=str(args.data_root.resolve()),
                     started_at=datetime.now(timezone.utc).isoformat(), state="training", completed=[], current=None,
                     seeds=SEEDS, modes=MODES, max_epochs=100, min_epochs=20, patience=15, min_delta=1e-4,
                     heldout_inference_performed=False, input_identity=identity, records={})
    def save():
        temporary = out / "suite_status.tmp"
        temporary.write_text(json.dumps(state, indent=2)+"\n"); temporary.replace(out / "suite_status.json")
    save()
    for seed in SEEDS:
        for mode in MODES:
            name = f"{mode}_seed{seed}"
            if name in state["completed"]:
                require((out / name / "run_complete.json").is_file(), "Completed run record missing")
                record = json.loads((out / name / "run_complete.json").read_text())
                require(record.get("study") == STUDY and record.get("status") == "completed", "Wrong completion record")
                for checkpoint in ("best", "last"):
                    require(c.sha(out / name / f"{checkpoint}.pt") == record[f"{checkpoint}_sha256"],
                            "Completed checkpoint changed")
                require(record.get("artifact_sha256"), "Completed run lacks artifact integrity records")
                for artifact, expected in record["artifact_sha256"].items():
                    require(c.sha(out / name / artifact) == expected, "Completed history, selection or curve changed")
                selection = json.loads((out / name / "selection.json").read_text())
                require(selection["selected_checkpoint_sha256"] == record["best_sha256"], "Selection/checkpoint identity mismatch")
                state["records"][name] = selection
                continue
            if args.resume:
                recover_checkpoint_free_start(out, name, state)
            command = build_command(args, mode, seed)
            state.update(state="training", current=name, command=command); save()
            with (out / f"{name}.log").open("a" if args.resume else "x") as log:
                process = subprocess.run(command, cwd=CODE_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                state.update(state="failed", exit_code=process.returncode); save()
                raise SystemExit(process.returncode)
            require((out / name / "run_complete.json").is_file(), "Run lacks completion record")
            state["records"][name] = json.loads((out / name / "selection.json").read_text())
            state["completed"].append(name); save()
    preference = validation_preference(state["records"])
    c.dump(out / "validation_only_preference.json", preference)
    state.update(state="all_six_completed_no_heldout_evaluation", current=None,
                 finished_at=datetime.now(timezone.utc).isoformat(), validation_only_preference=preference)
    save(); print("ALL_SIX_NEW_TRAINING_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
