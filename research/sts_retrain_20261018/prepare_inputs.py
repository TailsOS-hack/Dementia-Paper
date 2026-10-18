#!/usr/bin/env python3
"""Prepare and verify a minimal local training bundle. No upload or network use.

Only frozen stage train/validation images, clinical train/validation slices with
minimal pseudonymous grouping labels, two stage split records, and ImageNet
initialization are eligible. Test images and external clinical data are omitted.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image


STAGE_MANIFEST = "training_logs/experiments/kevin_20261011/prepared/stage_manifest.csv"
STAGE_AUDIT = "training_logs/experiments/kevin_20261011/prepared/split_audit.json"
CLINICAL_MANIFEST = "data/external/clinical_t1_preprocessed_v1/clinical_slices.csv"
STAGE_SHA = "66a0cd9409d61edd32ee137154f769c749664f255ddebe461cf7683c1f4ba7b9"
CLINICAL_FIELDS = ("path", "cohort", "subject_id", "group_key", "diagnosis", "binary_label", "split")
INIT_ARCHIVE_PATH = "cache/mobilenet_v3_large-8738ca79.pth"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_source(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe relative source path")
    path = root.joinpath(*name.parts)
    require(not path.is_symlink() and path.is_file(), "Missing or symlink source: " + relative)
    require(root.resolve() in path.resolve().parents, "Source resolves outside source root")
    return path


def csv_data(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def build(source_root, init_path, destination):
    source_root, init_path, destination = map(Path, (source_root, init_path, destination))
    source_root = source_root.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stage_data = safe_source(source_root, STAGE_MANIFEST).read_bytes()
    require(digest(stage_data) == STAGE_SHA, "Stage manifest differs from frozen source")
    stage = csv_data(stage_data)
    audit_data = safe_source(source_root, STAGE_AUDIT).read_bytes()
    audit = json.loads(audit_data)
    require(audit["manifest_sha256"] == STAGE_SHA, "Split audit manifest identity mismatch")
    require(dict(Counter(r["split"] for r in stage)) == {"train": 19023, "val": 2705, "test": 5450}, "Unexpected stage split counts")
    require(len(stage) == len({r["path"] for r in stage}) == 27178, "Stage rows duplicate or incomplete")
    stage_selected = [r for r in stage if r["split"] in ("train", "val")]
    require(len(stage_selected) == 21728, "Unexpected stage training input count")
    group_splits = defaultdict(set)
    for row in stage:
        group_splits[row["group_id"]].add(row["split"])
    require(all(len(s) == 1 for s in group_splits.values()), "A stage similarity group crosses splits")

    clinical_data = safe_source(source_root, CLINICAL_MANIFEST).read_bytes()
    clinical_source = csv_data(clinical_data)
    clinical = [{field: row[field] for field in CLINICAL_FIELDS} for row in clinical_source if row["split"] in ("train", "val")]
    require(len(clinical) == 387 and len({r["path"] for r in clinical}) == 387, "Unexpected or duplicate clinical slice count")
    require(dict(Counter(r["split"] for r in clinical)) == {"train": 288, "val": 99}, "Unexpected clinical split counts")
    subjects = defaultdict(set)
    labels = defaultdict(set)
    for row in clinical:
        require(row["group_key"] and row["subject_id"], "Clinical grouping identifier missing")
        require("openneuro" not in row["cohort"].lower(), "External source entered train/validation inputs")
        require({"AD": "1", "CN": "0"}.get(row["diagnosis"]) == row["binary_label"], "Clinical diagnosis/label mismatch")
        subjects[row["group_key"]].add(row["split"])
        labels[row["group_key"]].add(row["binary_label"])
    require(len(subjects) == 43 and all(len(s) == 1 for s in subjects.values()), "Clinical subject count or split crossing mismatch")
    require(all(len(s) == 1 for s in labels.values()), "Clinical subject has inconsistent labels")
    clinical_stream = io.StringIO(newline="")
    writer = csv.DictWriter(clinical_stream, fieldnames=CLINICAL_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(clinical)
    reduced_clinical_data = clinical_stream.getvalue().encode("utf-8")

    entries = []
    sources = {}
    generated = {CLINICAL_MANIFEST: reduced_clinical_data}
    metadata_counts = Counter()

    def add_file(name, source, kind, split=None, expected_pixel=None):
        require(name not in sources and name not in {e["archive_path"] for e in entries}, "Duplicate archive member")
        payload = source.read_bytes()
        entry = {"archive_path": name, "kind": kind, "bytes": len(payload), "sha256": digest(payload)}
        if split:
            entry["split"] = split
        if kind in ("stage_image", "clinical_slice"):
            with Image.open(io.BytesIO(payload)) as im:
                require(getattr(im, "n_frames", 1) == 1, "Unexpected multi-frame image")
                require(len(im.getexif()) == 0, "Image contains EXIF metadata; review before transfer")
                im.load()
                require(not getattr(im, "text", {}), "PNG contains textual metadata; review before transfer")
                metadata_counts["images_checked_no_exif_or_png_text"] += 1
                if kind == "stage_image":
                    require(im.format == "JPEG", "Stage input is not JPEG")
                    rgb = im.convert("RGB")
                    pixel_sha = digest(str(rgb.size).encode() + rgb.tobytes())
                    require(pixel_sha == expected_pixel, "Decoded image differs from frozen pixel identity: " + name)
                    entry["frozen_pixel_sha256_verified"] = pixel_sha
                else:
                    require(im.format == "PNG", "Clinical slice is not PNG")
        sources[name] = source
        entries.append(entry)

    for index, row in enumerate(stage_selected, 1):
        require(row["path"].startswith("data/alzheimers/") and Path(row["path"]).suffix.lower() in (".jpg", ".jpeg"), "Stage input outside allowed JPEG tree")
        add_file(row["path"], safe_source(source_root, row["path"]), "stage_image", row["split"], row["pixel_sha256"])
        if index % 5000 == 0:
            print("Verified %d / 21728 stage images" % index, flush=True)
    for row in clinical:
        name = "data/" + row["path"]
        require(name.startswith("data/external/clinical_t1_preprocessed_v1/images/") and name.endswith(".png"), "Clinical slice outside allowed preprocessed tree")
        add_file(name, safe_source(source_root, name), "clinical_slice", row["split"])
    for name in (STAGE_MANIFEST, STAGE_AUDIT):
        add_file(name, safe_source(source_root, name), "stage_split_record")
    entries.append({"archive_path": CLINICAL_MANIFEST, "kind": "reduced_clinical_manifest", "bytes": len(reduced_clinical_data), "sha256": digest(reduced_clinical_data)})
    require(init_path.is_file() and not init_path.is_symlink(), "ImageNet initialization file missing or symlinked")
    add_file(INIT_ARCHIVE_PATH, init_path, "imagenet_initialization")
    require(entries[-1]["sha256"].startswith("8738ca79"), "Unexpected official ImageNet checkpoint identity")
    require(len(entries) == 22119, "Unexpected total archive member count")

    archive = destination / "sts_training_inputs.tar.gz"
    temporary = destination / "sts_training_inputs.tar.gz.partial"
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for entry in sorted(entries, key=lambda e: e["archive_path"]):
                    name = entry["archive_path"]
                    payload = generated[name] if name in generated else sources[name].read_bytes()
                    require(digest(payload) == entry["sha256"], "Source changed after verification")
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(payload))
    by_name = {e["archive_path"]: e for e in entries}
    with tarfile.open(temporary, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == len(by_name) and {m.name for m in members} == set(by_name), "Archive membership differs from allowlist")
        for member in members:
            require(member.isfile() and not member.issym() and not member.islnk(), "Archive contains nonregular member")
            payload = tar.extractfile(member).read()
            require(member.size == by_name[member.name]["bytes"] and digest(payload) == by_name[member.name]["sha256"], "Archive readback hash mismatch")
        actual_clinical = csv_data(tar.extractfile(CLINICAL_MANIFEST).read())
        require(actual_clinical == clinical and tuple(actual_clinical[0]) == CLINICAL_FIELDS, "Reduced clinical manifest readback mismatch")
    temporary.replace(archive)

    counts = dict(Counter(e["kind"] for e in entries))
    sizes = {kind: sum(e["bytes"] for e in entries if e["kind"] == kind) for kind in counts}
    inventory = {
        "status": "prepared_locally_and_verified_not_uploaded",
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": file_digest(archive), "regular_file_count": len(entries)},
        "counts_by_kind": counts, "uncompressed_bytes_by_kind": sizes,
        "stage": {"train_images": 19023, "validation_images": 2705, "included_images": 21728, "excluded_test_images": 5450, "frozen_manifest_sha256": STAGE_SHA,
                  "full_manifest_rows": 27178, "note": "The unchanged full split manifest includes 5,450 test metadata rows solely to preserve frozen split identity; no test pixels are included.", "all_included_decoded_pixel_hashes_match": True},
        "clinical": {"train_slices": 288, "validation_slices": 99, "total_slices": 387, "train_subjects": 32, "validation_subjects": 11, "total_subjects": 43,
                     "manifest_fields": list(CLINICAL_FIELDS), "original_full_manifest_sha256": digest(clinical_data), "reduced_manifest_sha256": digest(reduced_clinical_data),
                     "pseudonymous_grouping_identifiers_retained": True, "age_and_sex_removed": True,
                     "note": "Diagnosis labels and pseudonymous subject/group identifiers remain in the reduced training manifest and corresponding filenames; they are needed for supervised training and subject grouping."},
        "verification": {"all_source_byte_hashes_recorded": True, "all_archive_members_readback_hash_checked": True, "exact_allowlist_membership_verified": True, **metadata_counts},
        "source_references": {"stage_manifest": STAGE_MANIFEST, "stage_audit": STAGE_AUDIT, "clinical_manifest": CLINICAL_MANIFEST,
                              "initialization": "official torchvision MobileNetV3-Large ImageNet V1 mobilenet_v3_large-8738ca79.pth"},
        "omissions": ["All stage test image pixels", "All clinical test and external slices", "Original clinical NIfTI volumes", "subjects.csv and clinical demographics (age/sex)",
                      "Clinical QA artifacts", "Reader identities, score sheets and results", "Original100 benchmark image files and key", "Historical or new fitted disease-model checkpoints", "Source code (root will transfer separately)", "Unrelated diseases or datasets"],
        "files": sorted(entries, key=lambda e: e["archive_path"]),
    }
    inventory_path = destination / "sts_training_inputs.inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    summary = ["Minimal training transfer bundle — prepared locally; not uploaded", "", "Archive: " + archive.name,
               "SHA256: " + inventory["archive"]["sha256"], "Compressed size: {:,} bytes ({:.2f} MiB)".format(archive.stat().st_size, archive.stat().st_size / 2**20),
               "Files: {:,}".format(len(entries)), "", "Included:",
               "- 21,728 legacy stage JPEGs: 19,023 training + 2,705 validation.",
               "- Original stage_manifest.csv (all 27,178 metadata rows) and split_audit.json, unchanged. The 5,450 test rows remain as split metadata, but no test image pixels are included.",
               "- 387 preprocessed clinical PNG slices from 43 pseudonymous subjects: 288 slices / 32 training subjects and 99 slices / 11 validation subjects.",
               "- Reduced clinical manifest with only path, cohort, subject_id, group_key, diagnosis, binary_label and split. Diagnosis labels and pseudonymous grouping identifiers remain; age and sex are removed.",
               "- Official MobileNetV3-Large ImageNet initialization under cache/.", "", "Omitted:"]
    summary.extend("- " + item for item in inventory["omissions"])
    summary.extend(["", "Verification:", "- Every included stage image's decoded RGB pixel hash matches the frozen manifest.",
                    "- All 22,115 included images were decoded and checked for EXIF or PNG text metadata; none was present.",
                    "- Every archive member's byte hash was verified by rereading the finished archive; exact member allowlist verified.",
                    "- Full per-file paths, byte sizes and hashes are in sts_training_inputs.inventory.json.", "", "No network calls, upload or cloud training were performed by this preparation script.", ""])
    summary_path = destination / "sts_training_inputs.review.txt"
    summary_path.write_text("\n".join(summary))
    print(json.dumps({"archive": str(archive), **inventory["archive"], "inventory": str(inventory_path), "review_summary": str(summary_path)}, indent=2), flush=True)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--imagenet-initialization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    require(args.output_dir.resolve() != repo_root and repo_root not in args.output_dir.resolve().parents,
            "Keep private input bundles outside the public repository")
    build(args.source_root, args.imagenet_initialization, args.output_dir)


if __name__ == "__main__":
    main()
