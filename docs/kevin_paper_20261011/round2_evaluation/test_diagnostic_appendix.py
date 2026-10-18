"""Synthetic saved-artifact checks; no models, clinical records, or inference."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pypdf import PdfReader

spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).with_name('prepare_attribution_diagnostic_appendix.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


class DiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir();(self.source/'cases').mkdir()
        for i in range(115):Image.new('RGB',(1200,240),(i,20,30)).save(self.source/'cases'/f'case_{i:03}.png')
        fig,ax=plt.subplots(figsize=(10,4));ax.plot([0,1],[0,1]);ax.set_title('Synthetic source')
        for stem in d.STEMS:fig.savefig(self.source/(stem+'.png'));fig.savefig(self.source/(stem+'.pdf'))
        plt.close(fig)
        (self.source/'gallery.html').write_text('synthetic');(self.source/'native.npy').write_bytes(b'unchanged synthetic bytes')
        d.dump(self.source/'verification.json',dict(case_count=115,valid_maps=115,randomization_records=72,cloud_probability_tolerance=1e-4,cloud_probability_or_argmax_failures=[dict(case_id=k,argmax_agreement=True,max_probability_difference=v) for k,v in d.EXPECTED_FAILURES.items()]))
        d.dump(self.source/'recovery_receipt.json',dict(no_inference=True));self.seal()
    def tearDown(self):self.tmp.cleanup()
    def seal(self):
        for name in ['SHA256SUMS.json','attribution_complete.json']:(self.source/name).unlink(missing_ok=True)
        inv=d.inventory(self.source);d.dump(self.source/'SHA256SUMS.json',inv);d.dump(self.source/'attribution_complete.json',dict(status='blocked_cloud_probability_replay_review',payloads=len(inv),manifest_sha256=d.sha(self.source/'SHA256SUMS.json')))
    def test_derivatives_and_atlas_preserve_block_and_source(self):
        before=d.inventory(self.source);out=self.root/'diagnostic';done=d.appendix(self.source,out)
        self.assertEqual(done['panel_count'],115);self.assertEqual(before,d.inventory(self.source));self.assertEqual((out/'native.npy').read_bytes(),b'unchanged synthetic bytes')
        prov=d.read(out/'diagnostic_provenance.json');self.assertEqual(len(prov['warning_visuals_sha256']),119);self.assertEqual(prov['original_marker_sha256'],d.sha(self.source/'attribution_complete.json'))
        self.assertEqual(d.read(out/'source_attribution_complete.json')['status'],'blocked_cloud_probability_replay_review')
        report=self.root/'report';report.mkdir()
        for i in range(28):(report/f'figure_{i:02}.pdf').write_bytes((self.source/(d.STEMS[0]+'.pdf')).read_bytes())
        inv=d.inventory(report);d.dump(report/'SHA256SUMS.json',inv);d.dump(report/'report_complete.json',dict(status='complete_numerically_verified_visual_review_required',payloads=len(inv),manifest_sha256=d.sha(report/'SHA256SUMS.json')))
        atlas=d.atlas(report,out,self.root/'atlas');self.assertEqual(atlas['pages'],30)
        pdf=PdfReader(self.root/'atlas'/'Round2_Diagnostic_Graphs_Atlas.pdf');self.assertEqual(len(pdf.pages),30);self.assertIn('DIAGNOSTIC ONLY',pdf.pages[29].extract_text())
    def test_changed_source_rejected_before_output(self):
        (self.source/'native.npy').write_bytes(b'tampered');out=self.root/'out'
        with self.assertRaises(ValueError):d.appendix(self.source,out)
        self.assertFalse(out.exists())
    def test_wrong_failure_semantics_rejected(self):
        v=d.read(self.source/'verification.json');v['cloud_probability_or_argmax_failures'][0]['argmax_agreement']=False;d.dump(self.source/'verification.json',v);self.seal();out=self.root/'out'
        with self.assertRaises(ValueError):d.appendix(self.source,out)
        self.assertFalse(out.exists())
    def test_existing_destination_rejected(self):
        out=self.root/'out';out.mkdir()
        with self.assertRaises(ValueError):d.appendix(self.source,out)
    def test_uncovered_payload_rejected(self):
        (self.source/'uncovered.txt').write_text('extra')
        with self.assertRaises(ValueError):d.appendix(self.source,self.root/'out')


if __name__=='__main__':unittest.main()
