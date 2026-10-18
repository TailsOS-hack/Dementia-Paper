from pathlib import Path
from PIL import Image
from collections import Counter,defaultdict
import hashlib,csv,json
import argparse
p=argparse.ArgumentParser();p.add_argument('--source-project',type=Path,required=True);a=p.parse_args()
root=a.source_project.resolve();out=Path(__file__).resolve().parent
known=defaultdict(set); dh=defaultdict(set)
for name in ['stage_manifest.csv','excluded_legacy.csv']:
 for r in csv.DictReader((root/'training_logs/experiments/kevin_20261011/prepared'/name).open()):
  if r.get('pixel_sha256'):known[r['pixel_sha256']].add(r.get('baseline_strict_split') or r.get('prior_split') or r.get('split','unknown'))
  if r.get('dhash'):dh[r['dhash']].add(r.get('baseline_strict_split') or r.get('prior_split') or r.get('split','unknown'))
rows=[]
base=root/'data/external/mendeley_xx9zzz6t54_v1/extracted'
for p in sorted(base.rglob('*.jpg')):
 with Image.open(p) as im:
  rgb=im.convert('RGB');ph=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest();pix=list(rgb.convert('L').resize((9,8)).getdata());d=0
  for y in range(8):
   for x in range(8):d=(d<<1)|(pix[y*9+x]>pix[y*9+x+1])
  d=f'{d:016x}'
 rows.append(dict(path=str(p.relative_to(base)),split=p.parent.parent.name,label=p.parent.name,width=rgb.width,height=rgb.height,pixel_sha256=ph,dhash=d,exact_legacy_overlap=bool(known[ph]),dhash_legacy_overlap=bool(dh[d]),legacy_splits=';'.join(sorted(known[ph]))))
with (out/'image_inventory.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
summary={'doi':'10.17632/xx9zzz6t54.1','source_url':'https://data.mendeley.com/datasets/xx9zzz6t54/1','archive_sha256':hashlib.sha256((root/'data/external/mendeley_xx9zzz6t54_v1/source.zip').read_bytes()).hexdigest(),'images':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),'class_counts':dict(Counter(r['label'] for r in rows)),'dimensions':dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),'metadata_csv_files':len(list(base.rglob('*.csv'))),'exact_legacy_overlap':sum(r['exact_legacy_overlap'] for r in rows),'dhash_legacy_overlap':sum(r['dhash_legacy_overlap'] for r in rows),'exact_overlap_by_split':{s:sum(r['exact_legacy_overlap'] for r in rows if r['split']==s) for s in ['train','test']},'patient_independence':'not established; no participant metadata in downloaded payload'}
(out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
