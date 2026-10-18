"""Tiny explicitly synthetic fixtures; never substitute these for trained results."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.sts_retrain_20261018.report import COMPLETION_ARTIFACTS, EXPECTED_CONFIG, EXPECTED_NAMES, run, sha, verify_manifests
from research.sts_retrain_20261018.train import CODE_ROOT, LEGACY_SHA, STAGE_SHA, STUDY, METRIC_DEFINITIONS
from research.sts_retrain_20261018.suite import validation_preference


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+"\n")


def metric(n, loss, clinical=False):
    classes = 2 if clinical else 4
    counts = [n//classes]*classes; counts[-1] += n-sum(counts)
    return dict(n=n, loss=loss, accuracy=1.0, unit="subject" if clinical else "legacy_image",
                confusion_matrix=[[counts[i] if i==j else 0 for j in range(classes)] for i in range(classes)])


def fixture(root, binaries=False):
    """Create only test data, with prominent suite marker and fake binary bytes."""
    selections = {}
    identity = dict(trainer_sha256=sha(CODE_ROOT/"research/sts_retrain_20261018/train.py"), clinical_manifest_sha256="b"*64,
                    torch_threads=1)
    for name in EXPECTED_NAMES:
        mode, seed = name.rsplit("_seed", 1); seed=int(seed)
        config=dict(EXPECTED_CONFIG, study=STUDY, mode=mode, seed=seed,
                    test_used_for_selection=False, heldout_inference_performed=False,
                    source_helper_sha256=LEGACY_SHA, stage_manifest_sha256=STAGE_SHA,
                    trainer_sha256=identity["trainer_sha256"], train_rows=19023, validation_rows=2705,
                    metric_definitions=METRIC_DEFINITIONS, torch_threads=1, torch_threads_actual=1,
                    clinical_manifest_sha256=identity["clinical_manifest_sha256"] if mode=="multitask" else None,
                    training_aux_subjects=32 if mode=="multitask" else 0, validation_aux_subjects=11 if mode=="multitask" else 0)
        history=[]
        for epoch in range(1, 21):
            # Epoch 3 is the mathematical minimum; delta rule preserves epoch 2.
            val_loss=.7 if epoch==1 else .5 if epoch==2 else .49995
            row=dict(epoch=epoch, train_loss=.4, val_loss=val_loss, train_online_nll=.45,
                     train_stage_objective=.6, train_total_objective_per_step=.65,
                     train_eval_accuracy=1., val_accuracy=1., train_online_accuracy=.95,
                     stage_draws=19023, aux_draws=4768 if mode=="multitask" else 0, optimizer_steps=595,
                     learning_rate=.0001, seconds=.01)
            evaluation=dict(train_stage=metric(19023,.4), val_stage=metric(2705,val_loss), train_aux={}, val_aux={})
            if mode=="multitask":
                row.update(train_aux_online_nll=.6,train_aux_objective=.65, train_aux_eval_nll=.4,val_aux_nll=.5,
                           train_aux_online_accuracy=.9,train_aux_eval_accuracy=1.,val_aux_accuracy=1.)
                evaluation.update(train_aux=metric(32,.4,True),val_aux=metric(11,.5,True))
            history.append(row)
            dump(root/name/f"epoch_{epoch:03d}_evaluation.json",evaluation)
        blob=b"SYNTHETIC TEST FIXTURE: NOT A MODEL CHECKPOINT"
        digest=hashlib.sha256(blob).hexdigest()
        selection=dict(study=STUDY,selected_epoch=2,val_loss=.5,reason="synthetic fixture validation",test_used_for_selection=False,
                       selected_checkpoint_sha256=digest)
        selections[name]=selection
        complete=dict(study=STUDY,status="completed",epochs=20,best_validation_nll=.5,heldout_inference_performed=False,
                      selected_epoch=2,best_sha256=digest,last_sha256=digest)
        for filename, value in (("config.json",config),("selection.json",selection),("history.json",history)):
            dump(root/name/filename,value)
        with (root/name/"history.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        complete["artifact_sha256"]={filename: sha(root/name/filename) if (root/name/filename).exists() else digest
                                     for filename in COMPLETION_ARTIFACTS}
        dump(root/name/"run_complete.json",complete)
        if binaries:
            (root/name/"best.pt").write_bytes(blob);(root/name/"last.pt").write_bytes(blob)
    preference=validation_preference(selections)
    dump(root/"validation_only_preference.json",preference)
    dump(root/"suite_status.json",dict(study=STUDY,synthetic_fixture=True,state="all_six_completed_no_heldout_evaluation",
         heldout_inference_performed=False,source_helper_sha256=LEGACY_SHA,completed=list(EXPECTED_NAMES),
         input_identity=identity,records=selections,validation_only_preference=preference))


class RetrainReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.root=self.base/"synthetic_fixture";self.output=self.base/"synthetic_report"
        fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def report(self, **kwargs):
        return run(self.root,self.output,make_plots=False,allow_synthetic=True,**kwargs)

    def test_synthetic_opt_in_required_and_metadata_only_status_clear(self):
        with self.assertRaisesRegex(ValueError,"explicit allow_synthetic"):
            run(self.root,self.output,make_plots=False)
        result=self.report()
        self.assertEqual(result["status"],"verified_synthetic_fixture")
        self.assertEqual(result["total_epochs"],120)
        self.assertEqual(result["checkpoint_verification"],"metadata_only")
        self.assertEqual(result["runs"][0]["selected_epoch"],2)
        self.assertEqual(result["runs"][0]["mathematical_minimum_epoch"],3)
        self.assertEqual(result["hyperparameters"],EXPECTED_CONFIG)
        self.assertEqual(result["denominators"]["clinical_validation_slices"],99)
        self.assertEqual(result["trainer_verification"]["reference"],"local_train.py")
        self.assertIn("SYNTHETIC TEST FIXTURE",(self.output/"README.md").read_text())

    def test_binary_hash_verification_and_required_missing_files(self):
        with self.assertRaisesRegex(ValueError,"Missing required checkpoint"):
            self.report(require_checkpoints=True)
        fixture(self.root,binaries=True)
        result=self.report(require_checkpoints=True)
        self.assertEqual(result["checkpoint_binaries_verified"],12)
        (self.root/EXPECTED_NAMES[0]/"best.pt").write_bytes(b"CHANGED SYNTHETIC BYTES")
        with self.assertRaisesRegex(ValueError,"Checkpoint hash mismatch"):
            self.report()

    def test_per_epoch_denominator_mismatch_rejected(self):
        path=self.root/EXPECTED_NAMES[0]/"epoch_003_evaluation.json"
        item=json.loads(path.read_text());item["train_stage"]["n"]=19022;dump(path,item)
        with self.assertRaisesRegex(ValueError,"incorrect clean evaluation denominator"):
            self.report()

    def test_wrong_selection_and_missing_epoch_rejected(self):
        path=self.root/EXPECTED_NAMES[0]/"selection.json"
        item=json.loads(path.read_text());item["selected_epoch"]=3;dump(path,item)
        with self.assertRaisesRegex(ValueError,"checkpoint epoch disagrees"):
            self.report()
        fixture(self.root)
        (self.root/EXPECTED_NAMES[0]/"epoch_003_evaluation.json").unlink()
        with self.assertRaisesRegex(ValueError,"Missing source file"):
            self.report()

    def test_changed_split_identity_and_incomplete_suite_rejected(self):
        path=self.root/EXPECTED_NAMES[0]/"config.json"
        item=json.loads(path.read_text());item["stage_manifest_sha256"]="f"*64;dump(path,item)
        with self.assertRaisesRegex(ValueError,"split identity mismatch"):
            self.report()
        fixture(self.root)
        path=self.root/"suite_status.json";item=json.loads(path.read_text());item["completed"].pop();dump(path,item)
        with self.assertRaisesRegex(ValueError,"all six runs"):
            self.report()

    def test_json_csv_disagreement_and_input_output_overlap_rejected(self):
        path=self.root/EXPECTED_NAMES[0]/"history.csv"
        path.write_text(path.read_text().replace("0.45","0.46",1))
        with self.assertRaisesRegex(ValueError,"CSV/JSON train_online_nll"):
            self.report()
        with self.assertRaisesRegex(ValueError,"outside the immutable input"):
            run(self.root,self.root/"report",make_plots=False,allow_synthetic=True)

    def test_clinical_manifest_subject_overlap_is_detected(self):
        rows=[]
        for subject in range(43):
            for image in range(9):
                rows.append(dict(path=f"synthetic/s{subject}_{image}.png",cohort="synthetic_smoke",
                                 group_key=f"synthetic_s{subject}",split="train" if subject<32 else "val",
                                 diagnosis="AD" if subject%2 else "CN",binary_label=str(subject%2)))
        path=self.base/"synthetic_clinical.csv"
        def write():
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            return hashlib.sha256(path.read_bytes()).hexdigest()
        digest=write()
        result=verify_manifests(clinical_manifest=path,clinical_digest=digest)
        self.assertEqual(result["clinical"]["subject_counts"],{"train":32,"val":11})
        rows[288]["group_key"]=rows[0]["group_key"]
        with self.assertRaisesRegex(ValueError,"crosses split or label"):
            verify_manifests(clinical_manifest=path,clinical_digest=write())

    def test_completion_artifact_digest_is_required_and_checked(self):
        path=self.root/EXPECTED_NAMES[0]/"run_complete.json"
        item=json.loads(path.read_text());item["artifact_sha256"]["history.json"]="0"*64;dump(path,item)
        with self.assertRaisesRegex(ValueError,"Completion artifact hash mismatch"):
            self.report()
        item.pop("artifact_sha256");dump(path,item)
        with self.assertRaisesRegex(ValueError,"must pin all five"):
            self.report()

    def test_completion_selection_and_checkpoint_identity_are_required(self):
        path=self.root/EXPECTED_NAMES[0]/"run_complete.json"
        item=json.loads(path.read_text());item.pop("selected_epoch");dump(path,item)
        with self.assertRaisesRegex(ValueError,"completion selection"):
            self.report()
        fixture(self.root)
        path=self.root/EXPECTED_NAMES[0]/"selection.json"
        item=json.loads(path.read_text());item.pop("selected_checkpoint_sha256");dump(path,item)
        with self.assertRaisesRegex(ValueError,"selected checkpoint hash"):
            self.report()

    def test_internal_agreement_cannot_substitute_for_expected_trainer_source(self):
        path=self.root/"suite_status.json"
        item=json.loads(path.read_text());item["input_identity"]["trainer_sha256"]="c"*64;dump(path,item)
        for name in EXPECTED_NAMES:
            path=self.root/name/"config.json"
            item=json.loads(path.read_text());item["trainer_sha256"]="c"*64;dump(path,item)
        with self.assertRaisesRegex(ValueError,"expected reviewed source"):
            self.report()
        result=self.report(expected_trainer_sha256="c"*64)
        self.assertEqual(result["trainer_verification"]["reference"],"explicit_archived_source_digest")

    def test_unapplied_or_different_cpu_thread_configuration_is_rejected(self):
        path=self.root/EXPECTED_NAMES[0]/"config.json"
        item=json.loads(path.read_text());item["torch_threads_actual"]=64;dump(path,item)
        with self.assertRaisesRegex(ValueError,"CPU thread setting"):
            self.report()


if __name__=="__main__":
    unittest.main()
