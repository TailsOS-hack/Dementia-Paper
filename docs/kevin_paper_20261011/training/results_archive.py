#!/usr/bin/env python3
"""Package completed results and verify every byte before local extraction.

No cloud lifecycle actions. An independently retrieved archive SHA256 is required
for verification; the manifest also checks all individual result files.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

ROOT_NAME = 'kevin_results'
MANIFEST_NAME = ROOT_NAME + '/ARTIFACT_SHA256.json'


def sha_file(path):
    with Path(path).open('rb') as stream:
        return sha_stream(stream)


def sha_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    return ('\x00' not in name and '\\' not in name
            and not path.is_absolute() and '..' not in path.parts
            and len(path.parts) > 1 and path.parts[0] == ROOT_NAME
            and str(path) == name)


def source_records(root):
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink refused: {path}')
        if path.is_file():
            name = ROOT_NAME + '/' + path.relative_to(root).as_posix()
            if not safe_name(name) or name == MANIFEST_NAME:
                raise ValueError(f'Invalid or reserved path: {name}')
            files.append(dict(path=name, size=path.stat().st_size, sha256=sha_file(path)))
        elif not path.is_dir():
            raise ValueError(f'Special file refused: {path}')
    return files


def validate_completion(root):
    status = json.loads((root / 'suite_status.json').read_text())
    complete = json.loads((root / 'evaluation_complete.json').read_text())
    expected = [f'{mode}_seed{seed}' for seed in (20261011, 20261012, 20261013)
                for mode in ('stage_only', 'multitask')]
    if (status['state'] != 'training_complete_awaiting_frozen_evaluation'
            or status['completed'] != expected or complete['checkpoints'] != 7
            or complete.get('candidate_order') != expected):
        raise ValueError('All six runs and the frozen evaluation must be complete')
    if sha_file(root / 'frozen_evaluation_all6/evaluation.json') != complete['evaluation_sha256']:
        raise ValueError('Evaluation completion digest mismatch')
    if sha_file(root / 'validation_selection_before_evaluation.json') != complete['validation_selection_sha256']:
        raise ValueError('Selection completion digest mismatch')
    evaluation = json.loads((root / 'frozen_evaluation_all6/evaluation.json').read_text())
    selection = json.loads((root / 'validation_selection_before_evaluation.json').read_text())
    candidate_names = {f'candidate_{i}' for i in range(1, 7)}
    if (set(evaluation.get('checkpoints', {})) != candidate_names | {'old_baseline'}
            or set(selection.get('candidates', {})) != candidate_names):
        raise ValueError('Actual evaluation/selection must contain all seven/six checkpoints')
    for i, name in enumerate(expected, 1):
        best = root / name / 'best.pt'
        last = root / name / 'last.pt'
        if not best.is_file() or not last.is_file():
            raise ValueError(f'Missing completed checkpoint files: {name}')
        digest = sha_file(best)
        if (evaluation['checkpoints'][f'candidate_{i}']['sha256'] != digest
                or selection['candidates'][f'candidate_{i}']['checkpoint_sha256'] != digest):
            raise ValueError(f'Frozen best checkpoint digest mismatch: {name}')


def validated_payload(tar):
    members = tar.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive entries')
    if any(not member.isfile() or not safe_name(member.name) for member in members):
        raise ValueError('Unsafe archive entry')
    name_set = set(names)
    if any(str(parent) in name_set for name in names for parent in PurePosixPath(name).parents):
        raise ValueError('Archive file/directory prefix collision')
    manifest = json.load(tar.extractfile(MANIFEST_NAME))
    records = manifest['files']
    if manifest.get('version') != 1 or not isinstance(records, list) or not records:
        raise ValueError('Unsupported or empty manifest')
    indexed = {item['path']: item for item in records}
    if (len(indexed) != len(records) or MANIFEST_NAME in indexed
            or name_set != set(indexed) | {MANIFEST_NAME}):
        raise ValueError('Manifest membership mismatch')
    for member in members:
        if member.name == MANIFEST_NAME:
            continue
        record = indexed[member.name]
        if member.size != record['size'] or sha_stream(tar.extractfile(member)) != record['sha256']:
            raise ValueError(f'Payload mismatch: {member.name}')
    return members, records


def pack(root, archive):
    root = root.resolve()
    archive = archive.absolute()
    partial = archive.with_suffix('.partial')
    if archive.exists() or archive.is_symlink() or partial.exists() or partial.is_symlink():
        raise ValueError('Archive already exists; preserve and inspect it')
    if archive.resolve().is_relative_to(root):
        raise ValueError('Archive must be outside the result source tree')
    validate_completion(root)
    files = source_records(root)
    payload = json.dumps(dict(version=1, files=files), indent=2).encode()
    with tarfile.open(partial, 'x:gz', compresslevel=6) as tar:
        for item in files:
            path = root / PurePosixPath(item['path']).relative_to(ROOT_NAME)
            # Emit regular files explicitly, including independently stored copies
            # of source hard links; never let tar create links or follow symlinks.
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as source:
                metadata = os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != item['size']:
                    raise ValueError(f'Source changed during packaging: {path}')
                info = tarfile.TarInfo(item['path'])
                info.size = item['size']
                info.mode = 0o600
                tar.addfile(info, source)
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(partial, 'r:gz') as tar:
        validated_payload(tar)
    if source_records(root) != files:
        raise ValueError('Source files changed during packaging; archive was not published')
    # Publish atomically without replacing a file that appeared during packaging.
    os.link(partial, archive)
    partial.unlink()
    return dict(archive=str(archive), sha256=sha_file(archive),
                file_count=len(files), bytes=archive.stat().st_size,
                archive_payload_verified=True, source_readback_verified=True)


def verify(archive, expected_sha, destination):
    if sha_file(archive) != expected_sha:
        raise ValueError('Archive SHA256 does not match independently retrieved value')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination already exists; never overwrite results')
    with tarfile.open(archive, 'r:gz') as tar:
        members, records = validated_payload(tar)
        destination.mkdir(parents=True)
        # All member paths/types/hashes have been checked before any extraction.
        for member in members:
            relative = PurePosixPath(member.name).relative_to(ROOT_NAME)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(chunk)
        # Read back from the actual destination, not just the archive stream.
        for item in records:
            target = destination / PurePosixPath(item['path']).relative_to(ROOT_NAME)
            if target.stat().st_size != item['size'] or sha_file(target) != item['sha256']:
                raise ValueError(f'Destination readback mismatch: {target}')
        if sha_file(destination / 'ARTIFACT_SHA256.json') != sha_stream(tar.extractfile(MANIFEST_NAME)):
            raise ValueError('Destination manifest readback mismatch')
    if sha_file(archive) != expected_sha:
        raise ValueError('Archive changed during verification; retrieval not verified')
    return dict(archive=str(archive), archive_sha256=expected_sha,
                destination=str(destination.resolve()), verified_payload_files=len(records),
                destination_readback_verified=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    packing = commands.add_parser('pack')
    packing.add_argument('--root', type=Path, required=True)
    packing.add_argument('--archive', type=Path, required=True)
    checking = commands.add_parser('verify')
    checking.add_argument('--archive', type=Path, required=True)
    checking.add_argument('--expected-sha256', required=True)
    checking.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = (pack(args.root, args.archive) if args.command == 'pack'
              else verify(args.archive, args.expected_sha256, args.destination))
    print(json.dumps(result, indent=2))
