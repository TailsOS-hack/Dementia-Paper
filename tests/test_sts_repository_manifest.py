import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from analysis.sts_results.refresh_repository_manifest import build_manifest, compare_manifests, file_record


class RepositoryManifestTests(unittest.TestCase):
    def test_pointer_and_downloaded_payload_have_identical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"a representative downloaded model payload\x00\x01"
            digest = hashlib.sha256(payload).hexdigest()
            path = root / "weights.pt"
            path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:%s\nsize %d\n" % (digest, len(payload)))
            pointed, is_pointer = file_record(root, "weights.pt")
            self.assertTrue(is_pointer)
            path.write_bytes(payload)
            downloaded, is_pointer = file_record(root, "weights.pt")
            self.assertFalse(is_pointer)
            self.assertEqual(pointed, downloaded)

    def test_new_public_outputs_covered_private_and_self_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("ignored/\n")
            paths = ["analysis/new.py", "results/sts/new.json", "work/runpod/sts_training_inputs.tar.gz",
                     "data/external/clinical.csv", "data/alzheimers/raw.jpg", "private_training.zip",
                     "ignored/file.txt", "analysis/__pycache__/code.pyc", "REPOSITORY_MANIFEST.json"]
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test\n")
            manifest, count = build_manifest(root)
            self.assertEqual(count, 0)
            self.assertEqual([row["path"] for row in manifest["files"]], [".gitignore", "analysis/new.py", "results/sts/new.json"])
            self.assertEqual((root / "REPOSITORY_MANIFEST.json").read_text(), "test\n")

    def test_invalid_pointer_and_symlink_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad.pt").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:bad\nsize 1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                file_record(root, "bad.pt")
            (root / "linked").symlink_to(root / "bad.pt")
            with self.assertRaisesRegex(ValueError, "symlink"):
                file_record(root, "linked")

    def test_diff_identifies_missing_changed_and_removed_payloads(self):
        old = {"files": [{"path": "a", "bytes": 1, "sha256": "one"}, {"path": "b", "bytes": 2, "sha256": "two"}]}
        new = {"files": [{"path": "a", "bytes": 3, "sha256": "three"}, {"path": "c", "bytes": 4, "sha256": "four"}]}
        self.assertEqual(compare_manifests(old, new), {"added": ["c"], "removed": ["b"], "changed": ["a"], "metadata_changed": False})


if __name__ == "__main__":
    unittest.main()
