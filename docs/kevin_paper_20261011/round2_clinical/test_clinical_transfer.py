"""Stdlib private archive roundtrip and adversarial installer checks.

Tests operate only inside temporary directories. The real bundle is read-only.
"""
import argparse
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile

spec=importlib.util.spec_from_file_location('clinical_transfer',Path(__file__).with_name('clinical_transfer.py'))
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)


def rejected(fn, phrase=None):
    try:fn()
    except (ValueError,KeyError,tarfile.TarError) as exc:
        if phrase is not None and phrase not in str(exc):raise AssertionError(str(exc))
        return
    raise AssertionError('Expected rejection')


def bad_tar(path,name,content=b'x',kind=None):
    with tarfile.open(path,'w:gz') as tar:
        info=tarfile.TarInfo(name);info.size=len(content)
        if kind is not None:info.type=kind;info.linkname='/etc/passwd';info.size=0
        tar.addfile(info,io.BytesIO(content) if kind is None else None)


def run(archive):
    checked=t.verify(archive,t.sha(archive));cases=[]
    with tempfile.TemporaryDirectory(prefix='clinical-transfer-test-') as d:
        p=Path(d);dst=p/'repo';dst.mkdir();(dst/'bundle_manifest.json').write_text('original stage manifest');(dst/'SHA256SUMS').write_text('original stage hashes')
        before={n:t.sha(dst/n) for n in ['bundle_manifest.json','SHA256SUMS']}
        result=t.install(archive,t.sha(archive),dst,p/'evidence')
        assert result['created_files']==396 and result['already_identical_files']==0
        assert {n:t.sha(dst/n) for n in before}==before
        cases.append('397-member archive install:396payloads including387developmentPNGs; old metadata preserved')
        again=t.install(archive,t.sha(archive),dst,p/'evidence')
        assert again['created_files']==0 and again['already_identical_files']==396
        cases.append('Idempotent byte-identical merge and full destination readback')
        conflict=p/'conflict';conflict.mkdir();last=checked['manifest']['files'][-1]['path'];target=conflict/last;target.parent.mkdir(parents=True);target.write_bytes(b'preexisting-different')
        rejected(lambda:t.install(archive,t.sha(archive),conflict,p/'evidence_conflict'),'overwrite')
        assert [str(x.relative_to(conflict)) for x in conflict.rglob('*') if x.is_file()]==[last]
        assert target.read_bytes()==b'preexisting-different'
        cases.append('Conflicting destination aborts whole preflight before any payload write')
        linked=p/'linked';linked.mkdir();outside=p/'outside';outside.mkdir();(linked/'data').symlink_to(outside,target_is_directory=True)
        rejected(lambda:t.install(archive,t.sha(archive),linked,p/'evidence_link'))
        assert list(outside.iterdir())==[]
        cases.append('Destination symlink parent rejected without outside writes')
        rejected(lambda:t.verify(archive,'0'*64),'SHA256')
        cases.append('Wrong archive SHA rejected')
        bad=p/'bad.tar.gz'
        bad_tar(bad,t.TOP+'../escape');rejected(lambda:t.verify(bad),'Unsafe')
        cases.append('Path traversal rejected')
        bad_tar(bad,'/absolute');rejected(lambda:t.verify(bad),'Unsafe')
        cases.append('Absolute member rejected')
        bad_tar(bad,t.TOP+'link',kind=tarfile.SYMTYPE);rejected(lambda:t.verify(bad),'regular')
        cases.append('Archive symlink rejected')
        bad_tar(bad,t.TOP+'hardlink',kind=tarfile.LNKTYPE);rejected(lambda:t.verify(bad),'regular')
        cases.append('Archive hardlink rejected')
        with tarfile.open(bad,'w:gz') as tar:
            t.tar_entry(tar,t.TOP+'repeated',b'a');t.tar_entry(tar,t.TOP+'repeated',b'a')
        rejected(lambda:t.verify(bad),'Duplicate');cases.append('Duplicate archive member rejected')
        # Preserve all legitimate members but corrupt the first actual PNG.
        with tarfile.open(archive,'r:gz') as src,tarfile.open(bad,'w:gz') as out:
            done=False
            for member in src:
                b=src.extractfile(member).read()
                if not done and member.name.endswith('.png'):
                    b=b[:-1]+bytes([b[-1]^1]);done=True
                t.tar_entry(out,member.name,b)
        rejected(lambda:t.verify(bad),'checksum');cases.append('Changed image bytes rejected by per-file SHA')
        with tarfile.open(archive,'r:gz') as src,tarfile.open(bad,'w:gz') as out:
            for member in src:t.tar_entry(out,member.name,src.extractfile(member).read())
            t.tar_entry(out,t.TOP+'data/heldout.jpg',b'undeclared')
        rejected(lambda:t.verify(bad),'membership');cases.append('Undeclared/heldout payload rejected')
    return dict(status='passed',tests_passed=len(cases),archive_sha256=t.sha(archive),cases=cases,
                source_sha256=t.sha(Path(t.__file__)),test_source_sha256=t.sha(Path(__file__)),cloud_actions=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    result=run(a.archive);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_bytes(t.json_bytes(result));print(json.dumps(result,indent=2))
