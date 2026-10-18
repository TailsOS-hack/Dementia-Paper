"""Appendix guards use explicitly synthetic metadata, never research estimates."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.sts_results import build_manuscript as manuscript


def synthetic_report(folder):
    rows = []
    for seed in (20261011, 20261012, 20261013):
        for mode in ('stage_only', 'multitask'):
            rows.append(dict(run=f'{mode}_seed{seed}', mode=mode, seed=seed,
                             completed_epochs=20, selected_epoch=2, selected_train_nll=.4,
                             selected_validation_nll=.5, train_images=19023, validation_images=2705))
    report = dict(study='sts_retrain_20261018', status='verified_synthetic_fixture', synthetic_fixture=True,
                  completed_runs=6, total_epochs=120, heldout_performance_estimated=False, runs=rows,
                  denominators=dict(stage_train_images=19023, stage_validation_images=2705,
                                    clinical_train_subjects=32, clinical_validation_subjects=11),
                  hyperparameters=dict(epochs=100, min_epochs=20, patience=15, min_delta=.0001,
                                       image_size=224, batch_size=32, learning_rate=.0001, weight_decay=.001,
                                       label_smoothing=.05, aux_weight=.25, aux_every=4),
                  validation_only_preference=dict(condition='multitask', representative='multitask_seed20261011',
                                                  test_used_for_selection=False),
                  checkpoint_binaries_verified=0,
                  checkpoint_checks=[dict(path=f'{r["run"]}/{name}.pt', status='metadata_only_binary_absent')
                                     for r in rows for name in ('best', 'last')],
                  split_checks={})
    path = folder/'report.json'
    path.write_text(json.dumps(report))
    with (folder/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Image existence checks only; the layout test separately uses labelled plots.
    for name in ('matched_clean_nll.png', 'clinical_clean_nll.png'):
        (folder/name).write_bytes(b'SYNTHETIC TEST PLACEHOLDER - NOT IMAGE DATA')
    return path, report


class RetrainingAppendixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name).resolve()
        self.path, self.report = synthetic_report(self.folder)

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.report))

    def appendix(self):
        return manuscript.make_retraining_appendix(self.path, allow_synthetic=True)

    def test_synthetic_rejected_without_internal_test_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'Synthetic results cannot'):
            manuscript.make_retraining_appendix(self.path)
        self.report['status'] = 'training_in_progress'; self.write()
        with self.assertRaisesRegex(ValueError, 'validated completed'):
            self.appendix()

    def test_two_pages_one_table_two_figures_from_report_directory(self):
        blocks = self.appendix()
        self.assertEqual(sum(b['kind'] == 'break' for b in blocks), 2)
        tables = [b for b in blocks if b['kind'] == 'table']
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]['rows']), 6)
        self.assertEqual(tables[0]['rows'][0][-2:], ['0.400000', '0.500000'])
        images = [b for b in blocks if b['kind'] == 'image']
        self.assertEqual(len(images), 2)
        self.assertTrue(all(Path(b['path']).parent == self.folder for b in images))
        text = '\n'.join(b.get('text', '') for b in blocks)
        self.assertIn('SYNTHETIC LOCAL TEST FIXTURE ONLY', text)
        self.assertIn('120 epochs', text)
        self.assertIn('checkpoint binaries were not supplied', text)
        self.assertIn('metadata-only verification', text)

    def test_summary_disagreement_and_wrong_denominators_rejected(self):
        self.report['runs'][0]['selected_train_nll'] = .3; self.write()
        with self.assertRaisesRegex(ValueError, 'report/summary mismatch'):
            self.appendix()
        self.path, self.report = synthetic_report(self.folder)
        self.report['denominators']['stage_train_images'] = 19022; self.write()
        with self.assertRaisesRegex(ValueError, 'denominators differ'):
            self.appendix()

    def test_run_identity_and_validation_preference_rejected(self):
        self.report['runs'][0]['mode'] = 'multitask'; self.write()
        with self.assertRaisesRegex(ValueError, 'condition/seed'):
            self.appendix()
        self.path, self.report = synthetic_report(self.folder)
        self.report['validation_only_preference']['condition'] = 'stage_only'; self.write()
        with self.assertRaisesRegex(ValueError, 'preferred condition'):
            self.appendix()

    def test_incomplete_or_inconsistent_checkpoint_evidence_rejected(self):
        self.report['checkpoint_checks'].pop(); self.write()
        with self.assertRaisesRegex(ValueError, 'all twelve checkpoint'):
            self.appendix()
        self.path, self.report = synthetic_report(self.folder)
        self.report['checkpoint_binaries_verified'] = 12; self.write()
        with self.assertRaisesRegex(ValueError, 'verification count'):
            self.appendix()

    def test_cli_without_flag_preserves_original_content_and_filename(self):
        historical = [dict(kind='title', text='Original historical content')]
        with patch.object(manuscript, 'make_content', return_value=(historical, {}, [])), \
             patch.object(manuscript, 'make_retraining_appendix') as appendix, \
             patch.object(manuscript, 'render_docx') as render, \
             patch.object(manuscript, 'markdown') as markdown, \
             patch('sys.argv', ['build_manuscript', '--source-commit', 'source', '--output-dir', str(self.folder)]):
            manuscript.main()
        appendix.assert_not_called()
        self.assertEqual(render.call_args.args[0], historical)
        self.assertEqual(render.call_args.args[2], self.folder/'STS_Expanded_Results.docx')
        self.assertEqual(markdown.call_args.args[0], historical)


if __name__ == '__main__':
    unittest.main()
