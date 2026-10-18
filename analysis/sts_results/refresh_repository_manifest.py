"""Preview or explicitly refresh the public repository inventory.

LFS pointer entries describe the referenced payload, so an LFS-skipped checkout
and a checkout with downloaded weights produce the same payload identity. This
does not download or independently validate the unavailable model bytes.
"""

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath


MANIFEST = "REPOSITORY_MANIFEST.json"
EXCLUSIONS = (
    MANIFEST, ".git/**", ".venv/**", ".pytest_cache/**", "**/__pycache__/**",
    "__pycache__/**", "*.pyc", "*.pyo", ".DS_Store", "**/.DS_Store", ".env*", "**/.env*",
    "work/**", "outputs/**", "data/external/**", "data/alzheimers/**",
    "**/round3_private/**", "round3_private/**", "research/dementia_round3/private/**",
    "*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.partial",
)
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
LFS_PATTERN = re.compile(rb"version https://git-lfs.github.com/spec/v1\n"
                         rb"oid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n?")
HASH_SEMANTICS = (
    "bytes and sha256 identify repository file payloads. For Git LFS pointers, "
    "they are the declared payload size and SHA256 object ID, not the pointer "
    "text hash; unavailable model bytes are not independently rehashed."
)


def excluded(relative):
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in EXCLUSIONS)


def file_record(root, relative):
    """Return the normalized file record and whether an LFS pointer was used."""
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError("Unsafe repository path: " + relative)
    path = root.joinpath(*parts.parts)
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise ValueError("Refusing symlink or outside-root inventory source: " + relative)
    with path.open("rb") as stream:
        prefix = stream.read(1024)
        if prefix.startswith(LFS_HEADER):
            match = LFS_PATTERN.fullmatch(prefix)
            if match is None or stream.read(1):
                raise ValueError("Malformed or unsupported LFS pointer: " + relative)
            return {"path": relative, "bytes": int(match.group(2)), "sha256": match.group(1).decode()}, True
        digest = hashlib.sha256(prefix)
        size = len(prefix)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {"path": relative, "bytes": size, "sha256": digest.hexdigest()}, False


def candidate_paths(root):
    """Include new public code/results as well as tracked files; honor gitignore."""
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    paths = sorted(set(output.decode("utf-8").split("\0")) - {""})
    # A staged deletion remains in the index until commit; inventory the current
    # deliverable tree, not absent historical files.
    return [p for p in paths if not excluded(p) and (root / p).exists()]


def build_manifest(root):
    root = Path(root).resolve()
    top = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip()).resolve()
    if top != root:
        raise ValueError("Supply the repository root, not a subdirectory")
    records = []
    pointer_count = 0
    for relative in candidate_paths(root):
        if not (root / relative).is_file():
            raise ValueError("Non-file repository entry needs explicit handling: " + relative)
        record, pointer = file_record(root, relative)
        records.append(record)
        pointer_count += pointer
    manifest = {"scope": "current dementia research snapshot", "source_history_preserved": True,
                "hash_semantics": HASH_SEMANTICS,
                "discovery": "Git tracked and nonignored untracked files; missing tracked files omitted",
                "exclusions": list(EXCLUSIONS), "files": records}
    return manifest, pointer_count


def compare_manifests(previous, current):
    before = {record["path"]: record for record in previous.get("files", [])}
    after = {record["path"]: record for record in current["files"]}
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(path for path in before.keys() & after.keys() if before[path] != after[path]),
        "metadata_changed": {key: value for key, value in previous.items() if key != "files"}
        != {key: value for key, value in current.items() if key != "files"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="Refresh the manifest after finalizing all public files")
    action.add_argument("--check", action="store_true", help="Fail if the existing inventory is incomplete or stale")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest, pointers = build_manifest(root)
    path = root / MANIFEST
    previous = json.loads(path.read_text()) if path.exists() else {}
    changes = compare_manifests(previous, manifest)
    if args.write:
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"mode": "write" if args.write else "check" if args.check else "preview",
                      "file_count": len(manifest["files"]), "lfs_pointer_payloads": pointers,
                      "manifest_written": args.write, **changes}, indent=2))
    if args.check and any(changes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
