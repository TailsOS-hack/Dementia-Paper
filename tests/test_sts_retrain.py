import copy
import importlib.util
import json
import math
import pickle
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from research.sts_retrain_20261018.train import (
    LEGACY_SHA, STUDY, SampleMean, load_legacy, preserved_rng, select_checkpoint,
    stage_weights, validate_output, synchronize_run_artifacts, validate_checkpoint_history, configure_torch_threads, parser,
)
from research.sts_retrain_20261018.suite import SEEDS, MODES, build_command, validation_preference, recover_checkpoint_free_start


class FakeState(list):
    def clone(self):
        return FakeState(self)


class FakeTorch:
    def __init__(self):
        self.cpu = FakeState([1, 2])
        self.gpu = [FakeState([3, 4])]
        self.metal = FakeState([5, 6])
        self.cuda = SimpleNamespace(is_available=lambda: True,
                                    get_rng_state_all=lambda: copy.deepcopy(self.gpu),
                                    set_rng_state_all=lambda state: setattr(self, "gpu", state))
        self.mps = SimpleNamespace(get_rng_state=lambda: self.metal.clone(),
                                   set_rng_state=lambda state: setattr(self, "metal", state))
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))

    def get_rng_state(self):
        return self.cpu.clone()

    def set_rng_state(self, value):
        self.cpu = value


class RetrainSemanticsTests(unittest.TestCase):
    def test_partial_batch_is_sample_weighted(self):
        loss = SampleMean(); loss.add(1.0, 32); loss.add(4.0, 3)
        self.assertAlmostEqual(loss.value, 44/35)
        self.assertNotAlmostEqual(loss.value, 2.5)

    def test_group_weights_balance_classes_and_groups_within_class(self):
        rows = ([dict(group_id="a", label=0)] * 5 + [dict(group_id="b", label=0)] +
                [dict(group_id="c", label=1)] * 3)
        weights = stage_weights(rows)
        self.assertAlmostEqual(sum(weights), len(rows))
        self.assertAlmostEqual(sum(weights[:6]), sum(weights[6:]))
        self.assertAlmostEqual(sum(weights[:5]), weights[5])
        with self.assertRaisesRegex(ValueError, "Mixed-label"):
            stage_weights([dict(group_id="a", label=0), dict(group_id="a", label=1)])

    def test_added_diagnostics_restore_all_rngs_even_on_error(self):
        torch = FakeTorch()
        state = {"numpy": [7, 8]}
        np = SimpleNamespace(random=SimpleNamespace(get_state=lambda: state["numpy"],
                              set_state=lambda value: state.update(numpy=value)))
        original_python = random.getstate()
        with self.assertRaises(RuntimeError):
            with preserved_rng(torch, np):
                random.random(); state["numpy"][0] = 99
                torch.cpu[0] = 99; torch.gpu[0][0] = 99; torch.metal[0] = 99
                raise RuntimeError("diagnostic failure")
        self.assertEqual(random.getstate(), original_python)
        self.assertEqual(state["numpy"], [7, 8])
        self.assertEqual(torch.cpu, [1, 2]); self.assertEqual(torch.gpu, [[3, 4]])
        self.assertEqual(torch.metal, [5, 6])

    def test_checkpoint_min_delta_and_validation_only_condition_preference(self):
        self.assertTrue(select_checkpoint(.0998, .1, .0001))
        self.assertFalse(select_checkpoint(.09995, .1, .0001))
        rows = {f"{mode}_seed{seed}": {"val_loss": .03 if mode == "stage_only" else .04,
                                     "train_loss": 100 if mode == "stage_only" else 0,
                                     "test_accuracy": 0 if mode == "stage_only" else 1}
                for seed in SEEDS for mode in MODES}
        result = validation_preference(rows)
        self.assertEqual(result["condition"], "stage_only")
        self.assertFalse(result["test_used_for_selection"])
        with self.assertRaises(ValueError):
            validation_preference(dict(list(rows.items())[:5]))

    def test_historical_overwrite_and_resume_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "inside a new"):
                validate_output(root / "historical")
            target = root / STUDY / "new_run"
            self.assertEqual(validate_output(target), target.resolve())
            target.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                validate_output(target)
            (target / "config.json").write_text(json.dumps({"study": "historical"}))
            with self.assertRaisesRegex(ValueError, "historical or different"):
                validate_output(target, resume=True)

    def test_original_helper_hash_and_fixed_suite_design(self):
        load_legacy()
        args = SimpleNamespace(data_root=Path("/data"), output=Path("/workspace/sts_retrain_20261018"),
                               stage_manifest="stage.csv", clinical_manifest="clinical.csv", clinical_image_root="data",
                               clinical_manifest_sha256="a"*64,
                               device="cuda", workers=4, torch_threads=1, imagenet_weights=None, resume=False)
        commands = [build_command(args, mode, seed) for seed in SEEDS for mode in MODES]
        self.assertEqual(len(commands), 6)
        for command in commands:
            self.assertEqual(command[command.index("--epochs")+1], "100")
            self.assertEqual(command[command.index("--patience")+1], "15")
            self.assertEqual(command[command.index("--min-epochs")+1], "20")
            self.assertEqual(command[command.index("--torch-threads")+1], "1")
            self.assertNotIn("--resume", command)
            self.assertNotIn("--test", command)
        self.assertNotIn("--clinical-manifest", commands[0])
        self.assertIn("--clinical-manifest", commands[1])

    def test_cpu_thread_default_and_invalid_counts(self):
        args = parser().parse_args(["--data-root", "/data", "--output", "/tmp/sts_retrain_20261018/run", "--mode", "stage_only"])
        self.assertEqual(args.torch_threads, 1)
        for invalid in (0, -1, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                configure_torch_threads(None, invalid)

    def test_helper_dataset_is_importable_for_spawn_workers(self):
        c = load_legacy()
        restored = pickle.loads(pickle.dumps(c.ImageRows([], None, None)))
        self.assertEqual(len(restored), 0)
        self.assertIs(type(restored), c.ImageRows)

    def test_condition_tie_uses_original_lexicographic_rule(self):
        rows = {f"{mode}_seed{seed}": {"val_loss": .04} for seed in SEEDS for mode in MODES}
        result = validation_preference(rows)
        self.assertEqual(result["condition"], "multitask")
        self.assertEqual(result["representative"], "multitask_seed20261011")


class CheckpointRecoveryTests(unittest.TestCase):
    def checkpoint(self):
        return dict(study=STUDY, epoch=2, selected_epoch=2, best_val_loss=.2, bad_epochs=0,
                    best_checkpoint_sha256=None, model_state={"synthetic_weight": 2},
                    config=dict(min_delta=.0001, min_epochs=20, patience=15, selection="validation NLL", mode="stage_only"),
                    history=[dict(epoch=1, train_loss=.4, val_loss=.3), dict(epoch=2, train_loss=.3, val_loss=.2)],
                    stage_validation={"loss": .2}, train_clean={"loss": .3}, aux_validation={}, train_aux_clean={})

    def storage(self):
        c = load_legacy()
        def atomic_save(value, path):
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value)); temporary.replace(path)
        return SimpleNamespace(sha=c.sha, dump=c.dump, csv_write=c.csv_write, atomic_save=atomic_save)

    def test_last_selected_epoch_recovers_best_and_missing_history_files(self):
        with tempfile.TemporaryDirectory() as temp:
            out, ck, c = Path(temp), self.checkpoint(), self.storage()
            c.atomic_save(ck, out / "last.pt")
            # Simulate a kill immediately after the canonical last checkpoint.
            (out / "best.pt").write_text("stale previous model")
            synchronize_run_artifacts(c, out, ck, render=False)
            self.assertEqual(json.loads((out / "best.pt").read_text())["model_state"], ck["model_state"])
            selection = json.loads((out / "selection.json").read_text())
            self.assertEqual(selection["selected_epoch"], 2)
            self.assertEqual(selection["selected_checkpoint_sha256"], c.sha(out / "best.pt"))
            self.assertEqual(json.loads((out / "history.json").read_text()), ck["history"])
            self.assertEqual(len(load_legacy().csv_read(out / "history.csv")), 2)

    def test_later_epoch_requires_exact_previous_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            out, ck, c = Path(temp), self.checkpoint(), self.storage()
            synchronize_run_artifacts(c, out, ck, render=False)
            digest = c.sha(out / "best.pt")
            later = copy.deepcopy(ck)
            later.update(epoch=3, bad_epochs=1, best_checkpoint_sha256=digest, stage_validation={"loss": .22})
            later["history"].append(dict(epoch=3, train_loss=.25, val_loss=.22))
            synchronize_run_artifacts(c, out, later, render=False)
            self.assertEqual(c.sha(out / "best.pt"), digest)
            (out / "best.pt").write_text("corrupt or different checkpoint")
            with self.assertRaisesRegex(ValueError, "selected checkpoint missing or changed"):
                synchronize_run_artifacts(c, out, later, render=False)

    def test_bad_checkpoint_selection_bookkeeping_is_rejected(self):
        ck = self.checkpoint(); ck["selected_epoch"] = 1
        with self.assertRaisesRegex(ValueError, "selected epoch disagrees"):
            validate_checkpoint_history(ck)

    def test_first_epoch_interruption_is_archived_before_fresh_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); name = "stage_only_seed20261011"; run = out / name; run.mkdir()
            state = dict(current=name, input_identity={"trainer_sha256": "reviewed"})
            (run / "config.json").write_text(json.dumps(dict(study=STUDY, mode="stage_only", seed=20261011, trainer_sha256="reviewed")))
            (run / "last.tmp").write_bytes(b"incomplete checkpoint")
            archived = recover_checkpoint_free_start(out, name, state)
            self.assertFalse(run.exists())
            self.assertEqual((archived / "last.tmp").read_bytes(), b"incomplete checkpoint")
            self.assertEqual(len(state["checkpoint_free_restarts"]), 1)

    def test_checkpoint_free_restart_refuses_saved_best_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); name = "stage_only_seed20261011"; run = out / name; run.mkdir()
            (run / "best.pt").write_bytes(b"must not discard")
            with self.assertRaisesRegex(ValueError, "Unexpected artifacts"):
                recover_checkpoint_free_start(out, name, dict(current=name))
            self.assertEqual((run / "best.pt").read_bytes(), b"must not discard")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"),
                     "PyTorch/torchvision runtime required for real tensor smoke checks")
class RealTorchInstrumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_legacy(); cls.c.imports()
        cls.c.torch.set_num_threads(1)

    def tiny_network(self):
        nn = self.c.nn
        model = nn.Module()
        model.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Dropout(.2), nn.Linear(6, 4))
        return model, nn.Linear(6, 2)

    def test_actual_cpu_thread_setting_is_applied_and_verified(self):
        torch = self.c.torch; before = torch.get_num_threads()
        try:
            self.assertEqual(configure_torch_threads(torch, 2), 2)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertEqual(configure_torch_threads(torch, 1), 1)
        finally:
            torch.set_num_threads(before)

    def synthetic_images(self, root):
        from PIL import Image
        rows = []
        for i in range(8):
            path = f"synthetic_{i}.png"
            Image.new("RGB", (12, 12), color=(20*i, 255-20*i, 100)).save(root / path)
            rows.append(dict(path=path, label=i%4, cohort="synthetic_smoke", subject_id=f"s{i//2}"))
        return rows

    def test_clean_eval_preserves_parameters_batchnorm_and_all_rng_streams(self):
        c = self.c; c.seed_all(42); model, aux = self.tiny_network(); model.eval(); aux.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        args = SimpleNamespace(image_size=12, batch_size=3, workers=0)
        py_state, np_state, torch_state = random.getstate(), copy.deepcopy(c.np.random.get_state()), c.torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = self.synthetic_images(root)
            with preserved_rng(c.torch, c.np):
                metrics, records = c.evaluate(model, aux, rows, root, c.torch.device("cpu"), "stage", args)
        self.assertEqual(metrics["n"], 8)
        for key, value in model.state_dict().items():
            self.assertTrue(c.torch.equal(value, before[key]), key)
        self.assertEqual(random.getstate(), py_state)
        self.assertTrue(c.torch.equal(c.torch.get_rng_state(), torch_state))
        for old, new in zip(np_state, c.np.random.get_state()):
            self.assertTrue(c.np.array_equal(old, new))
        expected = sum(-math.log(r["probabilities"][r["label"]]/sum(r["probabilities"])) for r in records)/8
        self.assertAlmostEqual(metrics["loss"], expected, places=12)

    def test_auxiliary_clean_loss_uses_subject_mean_probabilities(self):
        c = self.c; c.seed_all(72); model, aux = self.tiny_network()
        args = SimpleNamespace(image_size=12, batch_size=3, workers=0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); rows = self.synthetic_images(root)
            for i, row in enumerate(rows):
                row["label"] = (i//2)%2
            metrics, records = c.evaluate(model, aux, rows, root, c.torch.device("cpu"), "aux", args)
            model.eval(); aux.eval()
            dataset = c.dataset(rows, root, False, 12)
            with c.torch.inference_mode():
                probs = c.forward(model, aux, c.torch.stack([dataset[i][0] for i in range(8)]), "aux").softmax(1).numpy()
        expected = sum(-math.log(probs[i:i+2].mean(axis=0)[rows[i]["label"]]) for i in range(0, 8, 2))/4
        self.assertEqual(metrics["n"], 4)
        self.assertEqual(metrics["unit"], "subject")
        self.assertEqual([r["n_slices"] for r in records], [2, 2, 2, 2])
        self.assertAlmostEqual(metrics["loss"], expected, places=6)

    def test_actual_objective_and_online_nll_have_correct_distinct_denominators(self):
        c = self.c; c.seed_all(91)
        logits = c.torch.randn(7, 4); labels = c.torch.tensor([0, 1, 2, 3, 0, 1, 2])
        weights = c.torch.tensor([.5, .5, 1., 1., 1., 1., 2.])
        objective, online = SampleMean(), SampleMean()
        for start, stop in ((0, 4), (4, 7)):
            batch = c.nn.functional.cross_entropy(logits[start:stop], labels[start:stop], reduction="none", label_smoothing=.05)
            objective.add((batch*weights[start:stop]).mean().item(), stop-start)
            online.add(c.nn.functional.cross_entropy(logits[start:stop], labels[start:stop]).item(), stop-start)
        expected_obj = (c.nn.functional.cross_entropy(logits, labels, reduction="none", label_smoothing=.05)*weights).mean().item()
        expected_online = c.nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(objective.value, expected_obj, places=6)
        self.assertAlmostEqual(online.value, expected_online, places=6)
        self.assertNotAlmostEqual(objective.value, online.value, places=3)


if __name__ == "__main__":
    unittest.main()
