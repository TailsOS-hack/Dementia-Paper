"""Packaging helper tests; all created checkpoint bytes are temporary synthetic fixtures."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.sts_retrain_20261018 import package as pack


class PublicPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-packaging-test-')
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_private_inventory_participant_fields_and_host_paths_rejected(self):
        for item in ({'files': []}, {'nested': {'subject_id': 'synthetic'}}, {'path': '/Users/test/private.csv'},
                     {'value': 'api_key=synthetic-secret'}):
            with self.assertRaises(ValueError):
                pack.scan_public_metadata(item)
        pack.scan_public_metadata({'clinical_manifest_sha256': 'a'*64, 'training_aux_subjects': 32,
                                   'data_root': '/tmp/sts_input'})

    def test_copy_is_hash_pinned_and_symlinks_or_traversal_rejected(self):
        source = self.root/'source'; source.mkdir()
        target = self.root/'target'; target.mkdir()
        path = source/'config.json'; path.write_text('{"synthetic_fixture": true}')
        with self.assertRaisesRegex(ValueError, 'differs from verified'):
            pack.copy_public(source, 'config.json', target, '0'*64, [])
        records = []
        pack.copy_public(source, 'config.json', target, pack.sha(path), records)
        self.assertEqual((target/'config.json').read_bytes(), path.read_bytes())
        self.assertEqual(records[0]['sha256'], pack.sha(path))
        (source/'alias.json').symlink_to(path)
        for relative in ('alias.json', '../outside.json'):
            with self.assertRaises(ValueError):
                pack.safe_source(source, relative)

    def test_incomplete_or_synthetic_report_rejected_before_repository_mutation(self):
        repo = self.root/'repo'; repo.mkdir()
        suite = self.root/'suite'; suite.mkdir()
        report = self.root/'report.json'
        args = argparse.Namespace(repo=repo, suite=suite, report=report, export_inference_states=False)
        for record in ({'status': 'running'}, {'study': pack.STUDY, 'status': 'verified_synthetic_fixture',
                                             'synthetic_fixture': True, 'completed_runs': 6}):
            report.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'actual verified completed'):
                pack.package(args)
            self.assertEqual(list(repo.iterdir()), [])

    def test_stale_or_tampered_aggregate_plots_rejected_before_packaging(self):
        from research.sts_retrain_20261018.report import FIGURE_FILES, verify_figure_hashes
        repo=self.root/'repo';repo.mkdir()
        suite=self.root/'suite';suite.mkdir()
        report=self.root/'report.json'
        for name in FIGURE_FILES:
            (self.root/name).write_bytes(b'SYNTHETIC PLOT BYTES FOR HASH GUARD')
        hashes={name:pack.sha(self.root/name) for name in FIGURE_FILES}
        verify_figure_hashes(self.root,hashes)
        record=dict(study=pack.STUDY,status='verified_completed_retraining',synthetic_fixture=False,
                    completed_runs=6,checkpoint_verification='all_12_binaries_verified',figure_sha256={})
        # Deliberately insufficient metadata must fail at the figure boundary,
        # before the fresh verifier or any filesystem publication can occur.
        args=argparse.Namespace(repo=repo,suite=suite,report=report,export_inference_states=False)
        with patch.object(pack,'__file__',str(repo/'research'/pack.STUDY/'package.py')):
            report.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'all six generated figures'):
                pack.package(args)
            record['figure_sha256']=hashes;report.write_text(json.dumps(record))
            (self.root/'matched_clean_nll.png').write_bytes(b'TAMPERED SYNTHETIC PLOT')
            with self.assertRaisesRegex(ValueError,'figure hash mismatch: matched_clean_nll.png'):
                pack.package(args)
        self.assertEqual(list(repo.iterdir()),[])

    def test_tiny_synthetic_derivative_omits_resume_state_and_preserves_tensor_bytes(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Real tensor checks require torch')
        class SyntheticConstructor:
            STAGES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
            @staticmethod
            def imports():
                pass
            @staticmethod
            def network(pretrained=False):
                return torch.nn.Linear(2, 4), torch.nn.Linear(2, 2)
        for mode in ('stage_only', 'multitask'):
            model, aux = SyntheticConstructor.network()
            source = self.root/f'{mode}_SYNTHETIC_ORIGINAL.pt'
            original = dict(study=pack.STUDY, smoke_only=False, epoch=2, selected_epoch=2,
                            config=dict(mode=mode, seed=20261011, trainer_sha256='a'*64),
                            class_names=SyntheticConstructor.STAGES, arch='mobilenet_v3_large', image_size=224,
                            best_val_loss=.5, model_state=model.state_dict(), aux_state=aux.state_dict(),
                            aux_classes=['CN', 'AD'], optimizer_state={'SYNTHETIC': True}, rng_state={'SYNTHETIC': True})
            torch.save(original, source)
            row=dict(run=f'{mode}_seed20261011', mode=mode, seed=20261011, selected_epoch=2, selected_validation_nll=.5)
            report=dict(checkpoint_checks=[dict(path=f'{row["run"]}/best.pt', expected_sha256=pack.sha(source))],
                        trainer_sha256='a'*64, source_helper_sha256='b'*64)
            target=self.root/f'{mode}_SYNTHETIC_DERIVATIVE.pt'
            record=pack.export_inference_state(source, target, row, report, SyntheticConstructor, torch)
            restored=torch.load(target, weights_only=True)
            self.assertNotIn('optimizer_state', restored); self.assertNotIn('rng_state', restored)
            self.assertNotIn('config', restored)
            self.assertTrue(record['safe_weights_only_readback'])
            self.assertTrue(record['tensor_correspondence']['model_state']['exact_tensor_bytes_preserved'])
            self.assertEqual(bool(restored['aux_state']), mode == 'multitask')
            self.assertEqual(record['original_checkpoint_sha256'], pack.sha(source))

    def test_real_frozen_constructor_is_initialized_before_derivative_validation(self):
        """Actual architecture, random temporary weights, no downloads or images."""
        try:
            import torch
            import torchvision
        except ImportError:
            self.skipTest('Frozen architecture regression requires torch and torchvision')
        from research.sts_retrain_20261018.train import load_legacy
        helper = load_legacy()
        helper.imports()
        model, aux = helper.network(pretrained=False)
        source = self.root/'SYNTHETIC_FULL_ARCHITECTURE.pt'
        torch.save(dict(study=pack.STUDY, smoke_only=False, epoch=2, selected_epoch=2,
                        config=dict(mode='multitask', seed=20261011, trainer_sha256='a'*64),
                        class_names=helper.STAGES, arch='mobilenet_v3_large', image_size=224,
                        best_val_loss=.5, model_state=model.state_dict(), aux_state=aux.state_dict(),
                        aux_classes=['CN', 'AD']), source)
        row=dict(run='multitask_seed20261011', mode='multitask', seed=20261011,
                 selected_epoch=2, selected_validation_nll=.5)
        report=dict(checkpoint_checks=[dict(path=f'{row["run"]}/best.pt', expected_sha256=pack.sha(source))],
                    trainer_sha256='a'*64, source_helper_sha256='b'*64)
        target = self.root/'SYNTHETIC_FULL_ARCHITECTURE_DERIVATIVE.pt'
        # Recreate the fresh-process boundary; imports() must restore both names.
        with patch.dict(helper.__dict__):
            del helper.models
            del helper.nn
            with patch.object(helper, 'imports', wraps=helper.imports) as initialize:
                record = pack.export_inference_state(source, target, row, report, helper, torch)
                initialize.assert_called_once_with()
        self.assertTrue(record['frozen_constructor_strict_load'])
        self.assertTrue(record['safe_weights_only_readback'])
        self.assertEqual(record['tensor_correspondence']['model_state']['entries'], len(model.state_dict()))
        self.assertTrue(record['tensor_correspondence']['aux_state']['exact_tensor_bytes_preserved'])


if __name__ == '__main__':
    unittest.main()
