"""Prevent incompatible scientific quantities being plotted as matched losses."""
import csv
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("curve_schema", ROOT / "analysis_refresh/training_curves.py")
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)


class CurveSchemaTests(unittest.TestCase):
    def load(self, relative):
        with (ROOT / "docs/kevin_paper_20261011" / relative).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return rows, curves.curve_schema(rows[0])

    def test_clinical_slice_and_subject_losses_do_not_share_panel(self):
        rows, schema = self.load("publication/training/multitask_seed20261011/history.csv")
        curves.validate_history(rows, schema)
        fields = [{s["field"] for s in p["series"]} for p in schema["panels"]]
        self.assertFalse(any({"train_aux_loss", "val_aux_loss"} <= keys for keys in fields))
        self.assertFalse(any("train_stage_loss" in keys and "train_aux_loss" in keys for keys in fields))

    def test_raw_nll_and_optimization_objective_do_not_share_panel(self):
        rows, schema = self.load("round2_publication/artifacts/stage/training/stage_only_seed20261011/history.csv")
        curves.validate_history(rows, schema)
        objective = next(p for p in schema["panels"] if p["metric"] == "objective")
        self.assertEqual([s["field"] for s in objective["series"]], ["train_weighted_smoothed_objective"])
        self.assertEqual(rows[0]["epoch"], "0")
        self.assertEqual(rows[0]["train_raw_nll"], "")

    def test_round2_clean_losses_have_same_subject_aggregation(self):
        rows, schema = self.load("round2_publication/artifacts/clinical/training/clinical_seed20261021/history.csv")
        curves.validate_history(rows, schema)
        panel = schema["panels"][0]
        self.assertEqual(panel["unit"], "subject_mean_logits")
        self.assertEqual({s["field"] for s in panel["series"]}, {"train_clean_loss", "val_loss"})

    def test_unknown_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            curves.curve_schema(["epoch", "train_loss", "val_loss"])

    def test_new_metric_cannot_be_silently_dropped(self):
        rows, _ = self.load("publication/training/stage_only_seed20261011/history.csv")
        with self.assertRaisesRegex(ValueError, "Unclassified"):
            curves.curve_schema(list(rows[0]) + ["new_subject_loss"])

    def test_missing_post_initialization_measurement_is_rejected(self):
        rows, schema = self.load("round2_publication/artifacts/clinical/training/clinical_seed20261021/history.csv")
        rows[1]["train_aug_loss"] = ""
        with self.assertRaisesRegex(ValueError, "Missing"):
            curves.validate_history(rows, schema)

    def test_all_retained_histories_validate(self):
        count = 0
        for folder in curves.FOLDERS:
            for path in (ROOT / "docs/kevin_paper_20261011" / folder).rglob("history.csv"):
                with path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                curves.validate_history(rows, curves.curve_schema(rows[0]))
                count += 1
        self.assertEqual(count, 12)


if __name__ == "__main__":
    unittest.main()
