"""Check scientific bookkeeping and selection edge cases, without inference."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sts_training", ROOT / "analysis/sts_results/training.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class TrainingSelectionTests(unittest.TestCase):
    def test_sub_delta_improvement_does_not_change_selected_checkpoint(self):
        rows = [{"epoch": 1, "val_stage_loss": .1},
                {"epoch": 2, "val_stage_loss": .09995},
                {"epoch": 3, "val_stage_loss": .09994}]
        result = training.replay_selection(rows, minimum_epochs=1, patience=2)
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(result["mathematical_minimum_epoch"], 3)
        self.assertEqual(result["first_stopping_epoch"], 3)

    def test_missing_epoch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            training.replay_selection([{"epoch": 1, "val_stage_loss": .2},
                                       {"epoch": 3, "val_stage_loss": .1}])

    def test_post_stop_history_is_rejected(self):
        rows = [{"epoch": i, "val_stage_loss": .1} for i in range(1, 5)]
        with self.assertRaisesRegex(ValueError, "beyond"):
            training.replay_selection(rows, minimum_epochs=1, patience=2)

    def test_nonfinite_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            training.replay_selection([{"epoch": 1, "val_stage_loss": float("nan")}])


class PublishedTrainingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = training.audit_training(ROOT)

    def test_six_runs_are_two_conditions_with_three_seeds(self):
        self.assertEqual(self.result["run_count"], 6)
        self.assertEqual(self.result["conditions"], {"stage_only": 3, "multitask": 3})
        self.assertEqual(self.result["distinct_seed_count"], 3)
        self.assertEqual(self.result["total_completed_epochs"], 305)

    def test_all_frozen_epochs_match_history_replay(self):
        expected = [(38, 23), (43, 28), (52, 37), (74, 59), (54, 39), (44, 29)]
        actual = [(row["completed_epochs"], row["selected_epoch"]) for row in self.result["runs"]]
        self.assertEqual(actual, expected)
        self.assertTrue(all(row["epochs_since_selected"] == 15 for row in self.result["runs"]))
        self.assertEqual(self.result["exploratory_representative"], "candidate_3")

    def test_same_named_legacy_histories_are_not_conflated(self):
        lineage = self.result["legacy_lineage"]
        self.assertEqual([row["completed_epochs"] for row in lineage["histories"]], [32, 5, 20, 16, 17])
        self.assertFalse(lineage["same_filename_comparison"]["identical"])
        self.assertTrue(all(row["selected_epoch"] is None for row in lineage["histories"]))


if __name__ == "__main__":
    unittest.main()
