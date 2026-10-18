#!/usr/bin/env python3
"""Stdlib-only private clinical development archive builder/verifier/installer.

Only the frozen43development patients (387PNGs) and explicit support files may
enter. This never extracts arbitrary TAR paths, overwrites existing files, or
installs metadata at the old bundle_manifest.json/SHA256SUMS locations.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile

_PARENTS=Path(__file__).resolve().parents
ROOT=_PARENTS[3] if len(_PARENTS)>3 else Path.cwd()
BASE='docs/kevin_paper_20261011/round2_clinical/'
TOP='clinical_development_v1/'
MANIFEST=TOP+'clinical_input_manifest.json'
PATIENTS=BASE+'prepared/patients.json'
PROTOCOL=BASE+'prepared/protocol.json'
SUPPORT={
 BASE+'clinical_bag.py':'frozen_trainer',
 BASE+'test_clinical_bag.py':'synthetic_tests',
 BASE+'README.md':'frozen_usage',
 PROTOCOL:'frozen_protocol',PATIENTS:'private_development_patient_manifest',
 'models/kevin_20261011/imagenet_mobilenet_v3_large_8738ca79.pth':'official_imagenet_initialization',
 'docs/kevin_paper_20261011/preprocess_clinical_t1.py':'unchanged_preprocessor_reference',
 BASE+'clinical_transfer.py':'stdlib_private_installer',
 BASE+'smoke_clinical_cuda.py':'discarded_training_only_smoke',
}
PINS={
 BASE+'clinical_bag.py':'96c4d3a53b717d4e0bfcee25221c8c31ac6b8c84e9ea03652adfa9c45cd88156',
 PROTOCOL:'58061826e2c1b351845bf3a0a6022552a4f1e60ce637c6d267524f62b8bfe2cf',
 PATIENTS:'ecac589248aa48dfc117a94b2650d78ea4df945b507d51cf3da0a6b6713289e1',
 'models/kevin_20261011/imagenet_mobilenet_v3_large_8738ca79.pth':'8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04',
 'docs/kevin_paper_20261011/preprocess_clinical_t1.py':'bc3e1c7012cc54e9453af38d554073b3666cbc115fbb2a2072ed10a822f49475',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def json_bytes(obj):return (json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def safe_name(name):
    if not isinstance(name,str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('Invalid relative path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p)!=name or any(x.startswith('.') for x in p.parts):
        raise ValueError('Unsafe or noncanonical relative path')
    return p


def patient_images(patients):
    if len(patients)!=43 or len({r['group_key'] for r in patients})!=43:
        raise ValueError('Expected43unique development patients')
    counts=Counter((r['split'],r['label']) for r in patients)
    if counts!={('train',0):11,('train',1):21,('val',0):4,('val',1):7}:
        raise ValueError('Unexpected development labels/splits')
    images={}
    for r in patients:
        if r['cohort']!='besga_zenodo_3935636' or r['diagnosis']!=('AD' if r['label']==1 else 'CN'):
            raise ValueError('Wrong clinical development cohort/diagnosis')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['subject_id']) or r['group_key']!='besga_zenodo_3935636__'+r['subject_id']:
            raise ValueError('Patient identity mismatch')
        expected=[f"data/external/clinical_t1_preprocessed_v1/images/besga_zenodo_3935636/{r['subject_id']}/axial_{i:02d}.png" for i in range(9)]
        if r['paths']!=expected or len(r['pixel_sha256'])!=9 or r['n_slices']!=9 or r['slice_fractions']!=[.45,.5,.55,.6,.65,.7,.75,.8,.85]:
            raise ValueError('Changed frozen patient bag')
        if any(not re.fullmatch('[0-9a-f]{64}',h) for h in r['pixel_sha256']):
            raise ValueError('Invalid pixel hash')
        for p in r['paths']:
            safe_name(p)
            if p in images:raise ValueError('Repeated development image')
            images[p]='train_image' if r['split']=='train' else 'validation_image'
    if len(images)!=387:raise ValueError('Expected387distinct development PNGs')
    return images


def tar_entry(tar,name,b):
    info=tarfile.TarInfo(name);info.size=len(b);info.mtime=0;info.mode=0o644
    info.uid=info.gid=0;info.uname=info.gname=''
    tar.addfile(info,io.BytesIO(b))


def verify(archive,expected_sha=None):
    archive=Path(archive)
    archive_sha=sha(archive)
    if expected_sha is not None and (not re.fullmatch('[0-9a-f]{64}',expected_sha) or archive_sha!=expected_sha):
        raise ValueError('Archive SHA256 mismatch')
    payload={};records={};total=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            safe_name(member.name)
            if not member.name.startswith(TOP) or not member.isfile() or member.issparse():
                raise ValueError('Only regular files under the clinical archive prefix are allowed')
            if member.name in records:raise ValueError('Duplicate archive member')
            if member.size<0 or member.size>100*1024*1024:raise ValueError('Oversized archive member')
            total+=member.size
            if total>512*1024*1024:raise ValueError('Excess archive payload')
            content=tar.extractfile(member).read()
            if len(content)!=member.size:raise ValueError('Truncated archive payload')
            records[member.name]=dict(bytes=len(content),sha256=sha_bytes(content))
            if member.name in {MANIFEST,TOP+PATIENTS,TOP+PROTOCOL}:
                payload[member.name]=content
            if member.name.endswith('.png'):
                if content[:8]!=b'\x89PNG\r\n\x1a\n' or content[12:16]!=b'IHDR' or int.from_bytes(content[16:20],'big')!=224 or int.from_bytes(content[20:24],'big')!=224:
                    raise ValueError('Wrong clinical PNG header/dimensions')
    if set(payload)!={MANIFEST,TOP+PATIENTS,TOP+PROTOCOL}:raise ValueError('Missing archive controls')
    manifest=json.loads(payload[MANIFEST]);patients=json.loads(payload[TOP+PATIENTS]);protocol=json.loads(payload[TOP+PROTOCOL])
    if manifest.get('schema')!='private_clinical_development_v1' or manifest.get('heldout_images')!=0:
        raise ValueError('Invalid private clinical manifest')
    paths=patient_images(patients)|SUPPORT
    declared=manifest['files']
    if len(declared)!=len(paths) or len({r['path'] for r in declared})!=len(paths):raise ValueError('Repeated/missing declared payload')
    if {r['path'] for r in declared}!=set(paths) or set(records)!={TOP+p for p in paths}|{MANIFEST}:
        raise ValueError('Archive membership is not exactly387development images plus explicit support')
    for record in declared:
        p=record['path'];safe_name(p)
        if record['category']!=paths[p] or records[TOP+p]!={k:record[k] for k in ['bytes','sha256']}:
            raise ValueError('Payload category/checksum/size mismatch: '+p)
    for p,h in PINS.items():
        if records[TOP+p]['sha256']!=h:raise ValueError('Frozen source changed: '+p)
    if protocol['trainer_sha256']!=PINS[BASE+'clinical_bag.py'] or protocol['patients_manifest_sha256']!=PINS[PATIENTS] or protocol['initialization']['sha256']!=PINS[protocol['initialization']['path']]:
        raise ValueError('Prepared protocol pins disagree')
    if manifest.get('training_images')!=288 or manifest.get('validation_images')!=99 or manifest.get('patients')!=43:
        raise ValueError('Manifest development counts changed')
    return dict(status='verified_private_development_archive',archive_sha256=archive_sha,
                archive_bytes=archive.stat().st_size,manifest_sha256=records[MANIFEST]['sha256'],
                payload_files=len(paths),training_images=288,validation_images=99,heldout_images=0,
                patients=43,manifest=manifest)


def build(root,output):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite an existing archive')
    patients=json.loads((root/PATIENTS).read_text());paths=patient_images(patients)|SUPPORT
    records=[]
    for name,category in sorted(paths.items()):
        source=root/name
        if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError('Source must be an explicit regular local file')
        for parent in source.parents:
            if parent==root:break
            if parent.is_symlink():raise ValueError('Symlink source parent forbidden')
        if not stat.S_ISREG(source.stat().st_mode):raise ValueError('Nonregular source')
        digest=sha(source)
        if name in PINS and digest!=PINS[name]:raise ValueError('Frozen source pin changed')
        records.append(dict(path=name,category=category,bytes=source.stat().st_size,sha256=digest))
    manifest=dict(schema='private_clinical_development_v1',purpose='Only frozen Besga development fitting; no held-out outcome data',
                  patients=43,training_images=288,validation_images=99,heldout_images=0,
                  source_pins=PINS,files=records)
    output.parent.mkdir(parents=True,exist_ok=True);partial=output.with_suffix(output.suffix+'.partial')
    with partial.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        tar_entry(tar,MANIFEST,json_bytes(manifest))
        for record in records:
            content=(root/record['path']).read_bytes()
            if sha_bytes(content)!=record['sha256']:raise ValueError('Source changed during packaging')
            tar_entry(tar,TOP+record['path'],content)
    result=verify(partial)
    for record in records:
        if sha(root/record['path'])!=record['sha256']:raise ValueError('Source changed before publication')
    os.link(partial,output);partial.unlink()
    if sha(output)!=result['archive_sha256']:raise ValueError('Published archive readback failed')
    return result


def destination_path(destination,name):
    safe_name(name);target=destination/name
    if not target.resolve().is_relative_to(destination):raise ValueError('Destination escape')
    for p in [target,*target.parents]:
        if p==destination:break
        if p.is_symlink():raise ValueError('Symlink destination/parent forbidden')
    if target.exists() and not target.is_file():raise ValueError('Destination is not a regular file')
    return target


def install(archive,expected_sha,destination,evidence):
    result=verify(archive,expected_sha);manifest=result['manifest']
    original_destination=Path(destination)
    if original_destination.is_symlink():raise ValueError('Destination root must not be a symlink')
    destination=original_destination.resolve();evidence=Path(evidence).resolve()
    if evidence==destination or evidence.is_relative_to(destination):raise ValueError('Evidence must remain outside the repository destination')
    # Complete conflict/symlink preflight before any destination mutation.
    existing=[];missing=[]
    for record in manifest['files']:
        target=destination_path(destination,record['path'])
        if target.exists():
            if sha(target)!=record['sha256']:raise ValueError('Refusing to overwrite differing file: '+record['path'])
            existing.append(record['path'])
        else:missing.append(record['path'])
    created=[]
    records={r['path']:r for r in manifest['files']}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.name==MANIFEST:continue
            name=member.name[len(TOP):];record=records[name];target=destination_path(destination,name)
            content=tar.extractfile(member).read()
            if sha_bytes(content)!=record['sha256']:raise ValueError('Archive payload changed after verification')
            if target.exists():
                if sha(target)!=record['sha256']:raise ValueError('Destination changed after preflight')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            destination_path(destination,name)
            fd,tmp=tempfile.mkstemp(prefix='.clinical-install-',dir=target.parent)
            try:
                with os.fdopen(fd,'wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
                os.chmod(tmp,0o644)
                try:os.link(tmp,target);created.append(name)
                except FileExistsError:
                    if target.is_symlink() or not target.is_file() or sha(target)!=record['sha256']:raise ValueError('Concurrent destination conflict')
            finally:os.unlink(tmp)
    if sha(archive)!=expected_sha:raise ValueError('Archive changed during installation')
    readback=[]
    for record in manifest['files']:
        target=destination_path(destination,record['path']);digest=sha(target)
        if digest!=record['sha256'] or target.stat().st_size!=record['bytes']:raise ValueError('Destination readback mismatch')
        readback.append(dict(path=record['path'],sha256=digest,bytes=target.stat().st_size))
    evidence.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report={k:v for k,v in result.items() if k!='manifest'}|dict(status='installed_and_all_destination_files_reverified',
        destination=str(destination),created_files=len(created),already_identical_files=len(existing),
        completed_utc=stamp,all_readback=readback,old_bundle_metadata_written=False)
    for filename,obj in [(f'manifest_{expected_sha}.json',manifest),(f'installation_{stamp}.json',report)]:
        path=evidence/filename;content=json_bytes(obj)
        if path.exists():
            if path.read_bytes()!=content:raise ValueError('Evidence filename conflict')
        else:path.write_bytes(content)
    return report


def compact(result):return {k:v for k,v in result.items() if k not in {'manifest','all_readback'}}


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('build');a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('verify');a.add_argument('--archive',type=Path,required=True);a.add_argument('--sha256',required=True)
    a=s.add_parser('install');a.add_argument('--archive',type=Path,required=True);a.add_argument('--sha256',required=True)
    a.add_argument('--destination',type=Path,default=Path('/tmp/Neuro-Radiology-Database'));a.add_argument('--evidence',type=Path,default=Path('/workspace/clinical_input_verification'))
    a=p.parse_args()
    if a.command=='build':result=build(a.root,a.output)
    elif a.command=='verify':result=verify(a.archive,a.sha256)
    else:result=install(a.archive,a.sha256,a.destination,a.evidence)
    print(json.dumps(compact(result),indent=2))


if __name__=='__main__':main()
