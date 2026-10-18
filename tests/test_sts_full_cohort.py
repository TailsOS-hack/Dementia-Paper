"""Check denominator, class semantics, tamper detection and archived result recovery."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

from analysis.sts_results.full_cohort import CLASSES, audit_predictions, check_digest, generate, sha256


class FullCohortTests(unittest.TestCase):
    def setUp(self):
        # Off-diagonal mistakes have unequal class supports. Accuracy, balanced
        # accuracy and macro-F1 must therefore remain distinct estimands.
        assignments = [(0, 0), (0, 1), (1, 1), (2, 2), (3, 0), (3, 3)]
        self.rows = []
        self.expected = {}
        for i, (truth, predicted) in enumerate(assignments):
            path = "case_%d" % i
            self.expected[path] = CLASSES[truth]
            row = {"path": path, "TrueCategory": CLASSES[truth], "ModelPrediction": CLASSES[predicted]}
            row.update({"P_" + label: str(0.7 if j == predicted else 0.1) for j, label in enumerate(CLASSES)})
            self.rows.append(row)

    def test_nontrivial_metrics_and_confusion_axis(self):
        result = audit_predictions(self.rows, self.expected)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
        self.assertEqual(result["correct"], 4)
        self.assertEqual(result["errors"], 2)
        self.assertAlmostEqual(result["accuracy"], 4 / 6)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (0.5 + 2 / 3 + 1 + 2 / 3) / 4)
        self.assertAlmostEqual(result["negative_log_likelihood"], -(4 * math.log(0.7) + 2 * math.log(0.1)) / 6)
        self.assertEqual(result["per_class"][0]["support"], 2)
        self.assertAlmostEqual(result["per_class"][0]["specificity"], 3 / 4)
        self.assertEqual(result, audit_predictions(list(reversed(self.rows)), self.expected))

    def test_rejects_duplicate_or_changed_denominator(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit_predictions(self.rows + [self.rows[0]], self.expected)
        with self.assertRaisesRegex(ValueError, "membership"):
            audit_predictions(self.rows[:-1], self.expected)
        altered = copy.deepcopy(self.rows)
        altered[0]["path"] = "unexpected_case"
        with self.assertRaisesRegex(ValueError, "membership"):
            audit_predictions(altered, self.expected)

    def test_rejects_wrong_truth_or_decision(self):
        for key, value, message in [("TrueCategory", CLASSES[1], "truth"), ("ModelPrediction", CLASSES[1], "argmax")]:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaisesRegex(ValueError, message):
                audit_predictions(rows, self.expected)

    def test_rejects_invalid_probabilities(self):
        for invalid in ("nan", "inf", "-0.1", "1.1", "0.4"):
            rows = copy.deepcopy(self.rows)
            rows[0]["P_" + CLASSES[0]] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "[Pp]robability"):
                audit_predictions(rows, self.expected)

    def test_rejects_source_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.csv"
            path.write_text("original\n")
            frozen_hash = sha256(path)
            self.assertEqual(check_digest(path, frozen_hash), frozen_hash)
            path.write_text("modified\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                check_digest(path, frozen_hash)

    def test_archived_six_model_results_and_provenance(self):
        root = Path(__file__).resolve().parents[1]
        report = generate(root)
        self.assertEqual(report["n_images"], 5450)
        self.assertEqual(report["n_prepared_similarity_groups"], 4273)
        self.assertEqual([r["errors"] for r in report["models"]], [12, 37, 7, 15, 6, 22])
        self.assertEqual(sum(r["n"] for r in report["models"]), 32700)
        self.assertEqual(report["historical_baseline_split_counts_within_full_cohort"], {"train": 3860, "val": 532, "test": 1058})
        self.assertEqual(report["analysis_role"], "post_hoc_descriptive_overlapping_test_split")
        self.assertEqual(report["validation_exploratory_representative"], "candidate_3")
        self.assertTrue(report["candidate_hash_groups_disjoint_across_splits"])
        for model in report["models"]:
            self.assertEqual(sum(row["support"] for row in model["per_class"]), 5450)
            self.assertIn(model["prediction_file"], report["source_sha256"])


if __name__ == "__main__":
    unittest.main()
