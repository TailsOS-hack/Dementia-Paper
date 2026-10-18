"""Validate an approved, locally acquired OASIS manifest without inventing diagnoses."""
import argparse,csv,json
from pathlib import Path
from collections import Counter
from datetime import date
import nibabel as nib

def validate(rows,release,data_root):
 if release not in {'OASIS-1','OASIS-2','OASIS-3','OASIS-4'}:raise ValueError('Select one official release')
 required={'patient_id','image_path','reference_label','diagnosis_source','scan_date','diagnosis_date','site_id','release'}
 if not rows or not required.issubset(rows[0]):raise ValueError('Missing subject, diagnosis, timing, site or release metadata')
 seen=set()
 for row in rows:
  if row['release']!=release:raise ValueError('Do not silently pool different OASIS releases')
  if not all(row[k].strip() for k in required):raise ValueError('Empty required metadata')
  if row['reference_label'] not in {'AD','CN'}:raise ValueError('Explicit clinical AD/CN diagnoses required; CDR/stage labels are not automatically mapped')
  p=Path(row['image_path']);p=p if p.is_absolute() else data_root/p
  if not p.is_file() or not str(p).lower().endswith(('.nii','.nii.gz')):raise ValueError('Expected an acquired NIfTI image')
  img=nib.load(str(p))
  if len(img.shape)!=3 or min(img.shape)<2:raise ValueError('Expected a three-dimensional MRI volume')
  if abs((date.fromisoformat(row['scan_date'])-date.fromisoformat(row['diagnosis_date'])).days)>180:raise ValueError('Diagnosis is outside the 180-day intake window')
  key=(row['patient_id'],row['scan_date'])
  if key in seen:raise ValueError('Duplicate patient visit')
  seen.add(key)
 return {'release':release,'image_visits':len(rows),'unique_patients':len({r['patient_id'] for r in rows}),'labels':dict(Counter(r['reference_label'] for r in rows)),'eligible_for_training':False,'remaining_checks':['source-overlap audit','patient/site splitting','diagnosis timing window','modality and preprocessing QC','frozen development/evaluation protocol']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release',required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--approval-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 receipt=json.loads(a.approval_receipt.read_text())
 if receipt.get('access_approved') is not True or a.release not in receipt.get('approved_releases',[]):raise ValueError('Verified access for this release is required')
 result=validate(list(csv.DictReader(a.manifest.open())),a.release,a.data_root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
