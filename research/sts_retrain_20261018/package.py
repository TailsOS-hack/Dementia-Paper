"""Stage verified retraining evidence in the repository; never commit or upload.

Requires a completed real six-run aggregate report and all original checkpoints.
No broad directory copies: only named aggregate training artifacts are eligible.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

STUDY = 'sts_retrain_20261018'
RUN_FILES = ('config.json', 'history.json', 'history.csv', 'selection.json',
             'run_complete.json', 'learning_curves.png', 'learning_curves.pdf')
PRIVATE_KEYS = {'files', 'entries', 'subject_id', 'subject_ids', 'participant_id', 'participant_ids',
                'patient_id', 'patient_ids', 'image_path', 'image_paths', 'api_key', 'token',
                'access_token', 'password', 'secret', 'credentials', 'environment', 'environ'}
RUNTIME_KEYS = {'python', 'platform', 'machine', 'packages', 'cuda_runtime', 'gpu', 'gpu_memory_bytes',
                'driver', 'trainer_torch_threads', 'dataloader_workers', 'trainer_sha256',
                'source_helper_sha256', 'heldout_inference_performed'}
PROTOCOL_KEYS = {'recorded_at_utc', 'stage', 'study', 'trainer_sha256', 'suite_sha256', 'seeds', 'modes',
                 'torch_threads', 'workers', 'epochs_max', 'epochs_min', 'patience', 'min_delta',
                 'checkpoint_selection', 'condition_preference', 'heldout_inference_planned',
                 'prior_attempt', 'clinical_minimized_manifest_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def scan_public_metadata(value, label='metadata'):
    """Fail closed on participant fields, credentials, and personal host paths."""
    if isinstance(value, dict):
        for key, item in value.items():
            require(str(key).lower() not in PRIVATE_KEYS, f'{label}: nonpublic field {key}')
            scan_public_metadata(item, f'{label}.{key}')
    elif isinstance(value, (list, tuple)):
        for item in value:
            scan_public_metadata(item, label)
    elif isinstance(value, str):
        require(not re.search(r'(/Users/|/home/|\\Users\\|\.ssh/|\.aws/|BEGIN [A-Z ]*PRIVATE KEY)', value),
                f'{label}: private host path or key material')
        require(not re.search(r'(?i)(api[_-]?key|access[_-]?token|password)\s*[=:]', value),
                f'{label}: credential-like text')


def safe_source(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative artifact path')
    path = root/relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            f'Missing or symlinked source artifact: {relative}')
    return path


def copy_public(root, relative, stage, expected, records):
    source = safe_source(root, relative)
    require(expected and sha(source) == expected, f'Artifact differs from verified report: {relative}')
    if source.suffix == '.json':
        scan_public_metadata(json.loads(source.read_text()), str(relative))
    elif source.suffix == '.csv':
        with source.open(newline='') as stream:
            scan_public_metadata(list(csv.DictReader(stream)), str(relative))
    target = stage/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha(target) == expected, f'Copy integrity failure: {relative}')
    records.append(dict(path=Path(relative).as_posix(), sha256=expected, bytes=target.stat().st_size,
                        source_sha256=expected, transformation='byte-identical public artifact'))


def tensor_digest(state, torch):
    """Canonical named tensor content digest, independent of torch ZIP encoding."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        require(isinstance(name, str) and isinstance(tensor, torch.Tensor), 'Non-tensor model state')
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode()
        data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()


def export_inference_state(source, target, row, report, constructor, torch):
    """Read only our hash-pinned local training checkpoint; omit resume state."""
    checks = {check['path']: check for check in report['checkpoint_checks']}
    source_digest = checks[f'{row["run"]}/best.pt']['expected_sha256']
    require(sha(source) == source_digest, 'Selected checkpoint changed before derivative export')
    # Our trainer stores NumPy RNG objects, so originals need pickle. The CLI
    # requires explicit acknowledgement; never use this on downloaded unknown models.
    original = torch.load(source, map_location='cpu', weights_only=False)
    require(original['study'] == STUDY and original['smoke_only'] is False, 'Wrong or smoke checkpoint')
    require(original['epoch'] == original['selected_epoch'] == row['selected_epoch'], 'Selected epoch mismatch')
    require(original['config']['mode'] == row['mode'] and original['config']['seed'] == row['seed'], 'Model identity mismatch')
    require(original['config']['trainer_sha256'] == report['trainer_sha256'], 'Checkpoint trainer identity mismatch')
    require(original['class_names'] == constructor.STAGES and original['arch'] == 'mobilenet_v3_large', 'Architecture or labels changed')
    require(original['best_val_loss'] == row['selected_validation_nll'], 'Selected NLL mismatch')
    constructor.imports()  # The frozen helper initializes torch/torchvision lazily.
    model, aux = constructor.network(pretrained=False)
    model.load_state_dict(original['model_state'], strict=True)
    aux.load_state_dict(original['aux_state'], strict=True)
    enabled = row['mode'] == 'multitask'
    derivative = dict(format='sts_inference_state_v1', study=STUDY, task='dementia',
                      arch=original['arch'], class_names=list(original['class_names']), image_size=original['image_size'],
                      mode=row['mode'], seed=row['seed'], epoch=row['selected_epoch'], aux_enabled=enabled,
                      model_state={name: tensor.detach().cpu().clone() for name, tensor in original['model_state'].items()},
                      aux_classes=list(original['aux_classes']) if enabled else [],
                      aux_state={name: tensor.detach().cpu().clone() for name, tensor in original['aux_state'].items()} if enabled else {},
                      source_checkpoint_sha256=source_digest, trainer_sha256=report['trainer_sha256'],
                      source_helper_sha256=report['source_helper_sha256'],
                      selected_stage_validation_nll=row['selected_validation_nll'],
                      normalization_mean=[.485, .456, .406], normalization_std=[.229, .224, .225],
                      evaluation_transform='RGB; resize to image_size square with antialias; tensor; ImageNet normalization',
                      heldout_performance_estimated=False, resumable_training_checkpoint=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derivative, target)
    restored = torch.load(target, map_location='cpu', weights_only=True)
    require(set(restored) == set(derivative), 'Inference-state field mismatch after safe read-back')
    model.load_state_dict(restored['model_state'], strict=True)
    if enabled:
        aux.load_state_dict(restored['aux_state'], strict=True)
    correspondence = {}
    for key in ('model_state', 'aux_state'):
        if key == 'aux_state' and not enabled:
            correspondence[key] = dict(included=False, reason='Untrained auxiliary head omitted from stage-only model')
            continue
        require(set(restored[key]) == set(original[key]), 'Tensor names changed')
        require(all(torch.equal(restored[key][name], original[key][name].cpu()) for name in original[key]), 'Tensor value changed')
        original_digest = tensor_digest(original[key], torch)
        require(original_digest == tensor_digest(restored[key], torch), 'Tensor-byte digest changed')
        correspondence[key] = dict(included=True, entries=len(restored[key]), canonical_tensor_sha256=original_digest,
                                   exact_tensor_bytes_preserved=True)
    require(sha(source) == source_digest, 'Original checkpoint changed during export')
    return dict(run=row['run'], mode=row['mode'], seed=row['seed'], selected_epoch=row['selected_epoch'],
                original_checkpoint=f'{row["run"]}/best.pt', original_checkpoint_sha256=source_digest,
                original_bytes=source.stat().st_size,
                public_file=f'models/{STUDY}/{target.name}', derivative_sha256=sha(target), derivative_bytes=target.stat().st_size,
                safe_weights_only_readback=True, frozen_constructor_strict_load=True,
                tensor_correspondence=correspondence, image_inference_performed=False,
                omitted=['optimizer_state', 'scheduler_state', 'rng_state', 'config', 'history', 'aggregate_evaluation_records'])


def package(args):
    repo, suite, report_path = args.repo.resolve(), args.suite.resolve(), args.report.resolve()
    require(repo.is_dir() and suite.is_dir(), 'Repository or retrieved suite directory is absent')
    raw_target = repo/'results/sts/retraining/raw'
    model_target = repo/'models'/STUDY
    require(not raw_target.exists(), f'Refusing to overwrite existing evidence: {raw_target}')
    if args.export_inference_states:
        require(args.trust_local_checkpoint_pickle, 'Inference export requires --trust-local-checkpoint-pickle')
        require(not model_target.exists(), f'Refusing to overwrite existing models: {model_target}')
    supplied = json.loads(report_path.read_text())
    require(supplied.get('study') == STUDY and supplied.get('status') == 'verified_completed_retraining'
            and supplied.get('synthetic_fixture') is False and supplied.get('completed_runs') == 6,
            'Require an actual verified completed six-run aggregate report; synthetic/incomplete suites are forbidden')
    require(supplied.get('checkpoint_verification') == 'all_12_binaries_verified', 'Aggregate report must verify all original checkpoint binaries')
    require(repo == Path(__file__).resolve().parents[2],
            'Run this module from the target repository; --repo must match its source checkout')
    from . import report as verifier
    verifier.verify_figure_hashes(report_path.parent, supplied.get('figure_sha256'))
    with tempfile.TemporaryDirectory(prefix='sts-public-package-', dir=repo.parent) as temporary:
        temp = Path(temporary)
        fresh = verifier.run(suite, temp/'verification', make_plots=False, require_checkpoints=True,
                             stage_manifest=args.stage_manifest, clinical_manifest=args.clinical_manifest,
                             expected_trainer_sha256=args.expected_trainer_sha256)
        for field in ('study', 'status', 'completed_runs', 'total_epochs', 'runs', 'source_sha256',
                      'trainer_sha256', 'source_helper_sha256', 'hyperparameters', 'denominators',
                      'checkpoint_checks', 'validation_only_preference', 'split_checks', 'runtime',
                      'trainer_verification', 'completion_artifact_checks'):
            require(fresh[field] == supplied[field], f'Provided aggregate report is stale or mismatched: {field}')
        raw_stage, models_stage = temp/'raw', temp/'models'
        raw_stage.mkdir(); records = []; model_records = []
        for relative in ('suite_status.json', 'validation_only_preference.json'):
            copy_public(suite, relative, raw_stage, fresh['source_sha256'].get(relative), records)
        for row in fresh['runs']:
            relative_files = list(RUN_FILES)+[f'epoch_{epoch:03d}_evaluation.json' for epoch in range(1, row['completed_epochs']+1)]
            for filename in relative_files:
                relative = f'{row["run"]}/{filename}'
                copy_public(suite, relative, raw_stage, fresh['source_sha256'].get(relative), records)
        protocol = args.protocol or repo/'research'/STUDY/'protocol_before_final_training.json'
        additional = [('protocol_before_final_training.json', protocol, PROTOCOL_KEYS)]
        if args.runtime:
            additional.append(('runtime.json', args.runtime, RUNTIME_KEYS))
        for filename, source, allowed in additional:
            require(not source.is_symlink(), f'Symlinked provenance source: {filename}')
            source = source.resolve()
            require(source.is_file(), f'Missing public provenance source: {filename}')
            record = json.loads(source.read_text())
            require(set(record) <= allowed, f'Unexpected fields in {filename}; requires explicit public review')
            scan_public_metadata(record, filename)
            require(record.get('trainer_sha256') == fresh['trainer_sha256'], f'{filename}: trainer identity mismatch')
            if filename == 'runtime.json':
                require(record.get('source_helper_sha256') == fresh['source_helper_sha256'] and
                        record.get('heldout_inference_performed') is False, 'Runtime identity/scope mismatch')
                require(record.get('trainer_torch_threads') == fresh['runtime']['torch_threads'],
                        'Runtime thread setting mismatch')
            else:
                require(record.get('study') == STUDY and record.get('heldout_inference_planned') is False,
                        'Protocol identity/scope mismatch')
                require(record.get('suite_sha256') == sha(repo/'research'/STUDY/'suite.py'),
                        'Frozen protocol suite source differs from the current reviewed suite')
            shutil.copyfile(source, raw_stage/filename)
            records.append(dict(path=filename, sha256=sha(source), source_sha256=sha(source), bytes=source.stat().st_size,
                                transformation='byte-identical reviewed public provenance'))
        if args.export_inference_states:
            import torch
            torch.set_num_threads(1)
            from .train import load_legacy
            constructor = load_legacy()  # hash-checks the frozen constructor
            models_stage.mkdir()
            for row in fresh['runs']:
                source = safe_source(suite, f'{row["run"]}/best.pt')
                model_records.append(export_inference_state(source, models_stage/f'{row["run"]}_inference.pt',
                                                            row, fresh, constructor, torch))
            dump(models_stage/'provenance.json', dict(study=STUDY, source_aggregate_report_sha256=sha(report_path),
                                                      derivatives=model_records, heldout_inference_performed=False))
            (models_stage/'README.md').write_text(
                '# October 18 retraining inference states\n\n'
                'Six validation-selected inference-state derivatives. Load with `torch.load(path, map_location="cpu", weights_only=True)`; '
                'construct the hash-pinned frozen MobileNetV3-Large helper with `network(pretrained=False)` and strictly load `model_state`. '
                'Load `aux_state` only when `aux_enabled` is true. Stage-only auxiliary heads were untrained and are omitted.\n\n'
                'Run from the repository root with the research dependencies installed; construction does not download weights:\n\n'
                '```python\n'
                'import torch\n'
                'from research.sts_retrain_20261018.train import load_legacy\n\n'
                'path = "models/sts_retrain_20261018/stage_only_seed20261011_inference.pt"\n'
                'state = torch.load(path, map_location="cpu", weights_only=True)\n'
                'helper = load_legacy()\n'
                'helper.imports()\n'
                'model, aux = helper.network(pretrained=False)\n'
                'model.load_state_dict(state["model_state"], strict=True)\n'
                'model.eval()\n'
                'if state["aux_enabled"]:\n'
                '    aux.load_state_dict(state["aux_state"], strict=True)\n'
                '    aux.eval()\n'
                '```\n\n'
                'The tensor names, shapes, dtypes and bytes are unchanged from the selected local originals. '
                '`provenance.json` records original and derivative file hashes, selected epochs and canonical tensor-content hashes. '
                'Optimizer, scheduler, RNG, config and histories are omitted; these files cannot resume training. '
                'No held-out performance or clinical-readiness claim follows from these weights.\n')
        receipt = dict(study=STUDY, packaging_utc=datetime.now(timezone.utc).isoformat(),
                       aggregate_report_sha256=sha(report_path), packaging_script_sha256=sha(Path(__file__)),
                       aggregate_figure_sha256=supplied['figure_sha256'],
                       completed_runs=6, total_epochs=fresh['total_epochs'], raw_artifacts=records,
                       inference_derivatives=model_records, all_original_checkpoint_binaries_reverified=True,
                       clinical_manifest_membership_verified=fresh['split_checks']['clinical']['status']=='manifest_ids_and_groups_verified',
                       copied_private_images=False, copied_private_manifests=False, copied_logs=False,
                       copied_original_checkpoint_binaries=False, trained=False, image_inference_performed=False,
                       remote_publication_performed=False)
        dump(raw_stage/'package_manifest.json', receipt)
        (raw_stage/'README.md').write_text(
            '# Verified October 18 retraining source evidence\n\n'
            'Byte-identical public metadata, full histories, aggregate per-epoch evaluation JSON and run curves for six completed runs. '
            'The packaging step reran the aggregate verifier against all original checkpoint binaries before copying. '
            'Selected/last checkpoint hashes in the raw metadata identify the original resumable files, which remain local. '
            'Inference-state derivatives, when included, have separate hashes and explicit original-to-derivative mappings in '
            '`models/sts_retrain_20261018/provenance.json`. They must not be substituted for originals during binary verification.\n\n'
            'No training images, participant records, private manifests, input inventory, logs, credentials or resumable binaries are included. '
            'Runtime and protocol records contain only reviewed scientific provenance. This local packaging operation does not publish remotely. '
            'Historical benchmark results are unchanged; this evidence contains no new held-out inference.\n')
        # Nothing is copied into the repository until all evidence and optional
        # model derivatives have passed checks. Existing targets are never replaced.
        moved = []
        try:
            require(not raw_target.exists() and (not args.export_inference_states or not model_target.exists()),
                    'Publication targets appeared during verification; refusing replacement')
            raw_target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(raw_stage, raw_target); moved.append((raw_target, raw_stage))
            if args.export_inference_states:
                model_target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(models_stage, model_target); moved.append((model_target, models_stage))
        except BaseException:
            for destination, original in reversed(moved):
                os.replace(destination, original)
            raise
    return dict(raw_path=str(raw_target), models_path=str(model_target) if model_records else None,
                raw_artifact_count=len(records), model_count=len(model_records), remote_publication_performed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--stage-manifest', type=Path)
    parser.add_argument('--clinical-manifest', type=Path, help='Verify privately; never copied')
    parser.add_argument('--runtime', type=Path, help='Explicit reviewed capture_runtime.py output')
    parser.add_argument('--protocol', type=Path, help='Defaults to the reviewed public frozen protocol in repo')
    parser.add_argument('--expected-trainer-sha256')
    parser.add_argument('--export-inference-states', action='store_true')
    parser.add_argument('--trust-local-checkpoint-pickle', action='store_true',
                        help='Acknowledge these are the locally generated, hash-verified trainer checkpoints, not unknown downloaded pickle files')
    args = parser.parse_args()
    print(json.dumps(package(args), indent=2))


if __name__ == '__main__':
    main()
