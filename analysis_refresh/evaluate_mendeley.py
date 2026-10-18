"""Frozen stage evaluation of an overlapping image collection, not external validation."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import models,transforms
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,log_loss,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
CLASSES=['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'results/current/mendeley');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 paths=sorted(x for x in a.data.rglob('*.jpg') if x.parent.parent.name=='test');assert len(paths)==1279
 y=np.array([CLASSES.index(x.parent.name) for x in paths]);tf=transforms.Compose([transforms.Resize((224,224),antialias=True),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 torch.set_num_threads(4);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');torch.manual_seed(0)
 checkpoints=[ROOT/'models/alzheimers_classifier.pt']+sorted((ROOT/'models/kevin_20261011').glob('*.pt'))+sorted((ROOT/'models/kevin_round2_20261011').glob('stage_*.pt'))
 assert len(checkpoints)==10
 allmetrics=[];preds=[]
 for cp in checkpoints:
  start=time.monotonic();ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['class_names']==CLASSES and ck.get('image_size',224)==224 and not ck.get('smoke_only',False)
  model=models.mobilenet_v3_large(weights=None);model.classifier[3]=torch.nn.Linear(1280,4);model.load_state_dict(ck['model_state'],strict=True);model.to(device).eval();probs=[]
  with torch.inference_mode():
   for starti in range(0,len(paths),32):
    batch=[]
    for f in paths[starti:starti+32]:
     with Image.open(f) as im:batch.append(tf(im.convert('RGB')))
    probs.extend(model(torch.stack(batch).to(device)).softmax(1).cpu().numpy().tolist())
  prob=np.array(probs);prob/=prob.sum(1,keepdims=True);guess=prob.argmax(1);name=str(cp.relative_to(ROOT/'models')).replace('/','__').removesuffix('.pt')
  metric=dict(model=name,n=len(y),correct=int((y==guess).sum()),accuracy=float(accuracy_score(y,guess)),balanced_accuracy=float(balanced_accuracy_score(y,guess)),macro_f1=float(f1_score(y,guess,average='macro')),nll=float(log_loss(y,prob,labels=list(range(4)))),macro_ovr_auc=float(roc_auc_score(y,prob,multi_class='ovr',average='macro')),confusion_matrix=confusion_matrix(y,guess,labels=list(range(4))).tolist(),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),seconds=time.monotonic()-start)
  allmetrics.append(metric)
  for path,label,pr,pp in zip(paths,y,guess,prob):preds.append(dict(model=name,path=str(path.relative_to(a.data)),label=int(label),prediction=int(pr),**{f'p_{c}':float(v) for c,v in zip(CLASSES,pp)}))
  print(name,metric['correct'],len(y),f"{metric['seconds']:.1f}s",flush=True)
  del model,ck
  if device.type=='mps':torch.mps.empty_cache()
 with (a.output/'predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
 (a.output/'metrics.json').write_text(json.dumps(dict(scope='reused image diagnostic only; all 6400 source images overlap existing data',device=str(device),torch=torch.__version__,training_performed=False,model_selection_performed=False,classes=CLASSES,metrics=allmetrics),indent=2)+'\n')
if __name__=='__main__':main()
