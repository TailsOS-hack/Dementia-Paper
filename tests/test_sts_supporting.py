"""Protect image identity, label alignment and the frozen supporting results."""
import copy
import tempfile
import unittest
from pathlib import Path

from analysis.sts_results.supporting import (CLASSES, PRIMARY, SENSITIVITY, align, ensemble,
                                             index_predictions, paired_counts, run)


def raw(key, label=0):
    return {"path": key, "TrueCategory": CLASSES[label], "ModelPrediction": CLASSES[label],
            **{f"P_{name}": 0.97 if i == label else 0.01 for i, name in enumerate(CLASSES)}}


class SupportingIdentityTests(unittest.TestCase):
    def test_duplicate_ids_and_conflicting_schema_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_predictions([raw("a"), raw("a")])
        row = raw("a"); row["case_id"] = "b"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            index_predictions([row])

    def test_missing_case_and_reference_label_change_rejected(self):
        a = index_predictions([raw("a"), raw("b", 1)])
        with self.assertRaisesRegex(ValueError, "Case ID mismatch"):
            align(a, index_predictions([raw("a")]))
        with self.assertRaisesRegex(ValueError, "Reference-label mismatch"):
            align(a, index_predictions([raw("a"), raw("b", 2)]))

    def test_invalid_probabilities_and_stale_hard_labels_rejected(self):
        for value in (float("nan"), -0.1, 1.1):
            row = raw("a"); row["P_MildDemented"] = value
            with self.assertRaises(ValueError):
                index_predictions([row])
        row = raw("a"); row["ModelPrediction"] = CLASSES[1]
        with self.assertRaisesRegex(ValueError, "Argmax"):
            index_predictions([row])

    def test_ensemble_is_key_aligned_and_uses_probabilities_not_hard_vote(self):
        a = index_predictions([raw("a"), raw("b", 1)])
        b = index_predictions([raw("b", 1), raw("a")])
        c = copy.deepcopy(a)
        for member in (a, b):
            member["a"].update(probabilities=(0.51, 0.47, 0.01, 0.01), prediction=CLASSES[0])
        c["a"].update(probabilities=(0.01, 0.97, 0.01, 0.01), prediction=CLASSES[1])
        averaged = ensemble([a, b, c])
        self.assertEqual(averaged["a"]["prediction"], CLASSES[1])
        self.assertEqual(averaged["b"]["prediction"], CLASSES[1])

    def test_original100_normalizes_only_known_filename_schema(self):
        row = raw("data/evaluation/images/image_001.jpg")
        row["FileName"] = "image_001.jpg"
        self.assertEqual(list(index_predictions([row], original100=True)), ["image_001.jpg"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_predictions([row, {**row, "path": "elsewhere/image_001.jpg"}], original100=True)

    def test_paired_counts_preserve_direction(self):
        reference = index_predictions([raw("a"), raw("b", 1)])
        candidate = copy.deepcopy(reference)
        candidate["a"]["prediction"] = CLASSES[1]
        counts = paired_counts(candidate, reference)
        self.assertEqual(counts["candidate_wrong_reference_correct"], 1)
        self.assertEqual(counts["candidate_correct_reference_wrong"], 0)


class SupportingFrozenEvidenceTests(unittest.TestCase):
    def test_public_predictions_reproduce_all_frozen_points_and_contrasts(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            audit = run(root, Path(temp), make_plot=False)
        self.assertEqual(audit["status"], "verified")
        self.assertEqual(audit["cohort_model_rows"], 27)
        self.assertEqual(audit["paired_contrasts_verified"], 36)
        self.assertEqual(audit["frozen_point_records_verified"], 30)
        self.assertEqual(audit["cohorts"][PRIMARY]["models"]["ensemble_stage_only"]["correct"], 1058)
        self.assertEqual(audit["cohorts"][SENSITIVITY]["models"]["ensemble_stage_only"]["correct"], 756)
        self.assertEqual(audit["cohorts"]["original100"]["models"]["old_baseline"]["correct"], 100)


if __name__ == "__main__":
    unittest.main()
