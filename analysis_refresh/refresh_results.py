"""Recompute retained clinical statistics and render current plots from saved predictions."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,log_loss,roc_auc_score,roc_curve,precision_recall_curve
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/kevin_paper_20261011/round2_clinical'))
import clinical_statistics as stats
import clinical_figures as figures
COHORTS={'besga_reused_test':(7,4),'openneuro_reused_external':(1,14)}
SCENARIOS=['clean','intensity_0.9','intensity_1.1','gaussian_noise_0.01','gaussian_noise_0.02','blur_0.5','blur_1.0']
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def clinical(source,out):
 rows=[r for r in json.loads(source.read_text()) if r['cohort'] in COHORTS and r['binary_evaluable']]
 assert len(rows)==26*8*7
 indexed={(r['cohort'],r['group_key'],r['model'],r['scenario']):r for r in rows};assert len(indexed)==len(rows)
 data={};metrics=[]
 for cohort,(nad,ncn) in COHORTS.items():
  keys=sorted({r['group_key'] for r in rows if r['cohort']==cohort});assert len(keys)==nad+ncn
  y=np.array([indexed[cohort,k,figures.c.ORDER[0],'clean']['label'] for k in keys]);assert (y==1).sum()==nad
  d={'keys':keys,'y':y,'models':{}}
  for model in figures.c.ORDER:
   d['models'][model]={}
   clean=np.array([indexed[cohort,k,model,'clean']['probabilities'] for k in keys])
   for scenario in SCENARIOS:
    rr=[indexed[cohort,k,model,scenario] for k in keys]
    assert all(r['label']==int(label) and r['n_slices']==9 for r,label in zip(rr,y))
    pp=np.array([r['probabilities'] for r in rr]);stats.validate(y,pp)
    if model in ['round1_clinical_ensemble','new_clinical_ensemble']:
     members=figures.c.OLD if model.startswith('round1') else figures.c.NEW
     expected=np.mean([np.array([indexed[cohort,k,m,scenario]['probabilities'] for k in keys]) for m in members],axis=0)
     assert np.allclose(expected,pp,atol=1e-7,rtol=0)
    m=stats.metrics(y,pp);m.update(mean_probability_total_variation=float(np.abs(pp-clean).sum(1).mean()/2),label_flip_fraction=float(np.mean((pp[:,1]>=.5)!=(clean[:,1]>=.5))))
    d['models'][model][scenario]={'prob':pp,'metrics':m};metrics.append(dict(model=model,scenario=scenario,cohort=cohort,**m))
  data[cohort]=d
 assert len(metrics)==112
 out.mkdir(parents=True,exist_ok=True)
 prior=ROOT/'docs/kevin_paper_20261011/round2_publication/artifacts/clinical/report/all_model_scenario_metrics.csv'
 old={(r['cohort'],r['model'],r['scenario']):r for r in csv.DictReader(prior.open())}
 for m in metrics:
  for key in ['accuracy','balanced_accuracy','sensitivity_ad','specificity_cn','nll','binary_brier','roc_auc']:
   assert abs(m[key]-float(old[m['cohort'],m['model'],m['scenario']][key]))<1e-10,(key,m['cohort'],m['model'])
 analyses=figures.analyze(data,draws=10000)
 for v in analyses.values():v.pop('patient_ids',None)
 dump(out/'analysis.json',dict(scope='two reused clinical cohorts',cohorts=analyses,fresh_external_binary_n=0,n_patients=26,all_model_scenario_count=112,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),probability_metrics_recomputed=True,patient_rows_public=False,training_performed=False))
 table(out/'all_model_scenario_metrics.csv',[{k:v for k,v in m.items() if not isinstance(v,(dict,list))} for m in metrics])
 for cohort,d in data.items():figures.cohort_figures(cohort,d,out,False);figures.stress_figures(cohort,d,out,False)
 return {'metrics':112,'model_cohort_pairs':16,'saved_prediction_rows_checked':len(rows),'numeric_match_to_retained_prior_metrics':True}
def stage(out):
 p=ROOT/'results/current/mendeley/predictions.csv';rows=list(csv.DictReader(p.open()));names=sorted({r['model'] for r in rows});summary=[]
 classes=['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
 for name in names:
  rr=[r for r in rows if r['model']==name];assert len(rr)==1279
  y=np.array([int(r['label']) for r in rr]);pp=np.array([[float(r['p_'+c]) for c in classes] for r in rr]);assert np.allclose(pp.sum(1),1,atol=1e-6);pp/=pp.sum(1,keepdims=True);pred=pp.argmax(1);cm=confusion_matrix(y,pred,labels=range(4))
  metric=dict(model=name,n=1279,correct=int((y==pred).sum()),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),nll=float(log_loss(y,pp,labels=range(4))),macro_ovr_auc=float(roc_auc_score(y,pp,multi_class='ovr',average='macro')));summary.append(metric)
  fig,axs=plt.subplots(1,3,figsize=(16,4.7));axs[0].imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues')
  for i in range(4):
   for j in range(4):axs[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='black')
  axs[0].set(xticks=range(4),yticks=range(4),xticklabels=['Mild','Moderate','None','Very mild'],yticklabels=['Mild','Moderate','None','Very mild'],xlabel='Predicted',ylabel='Provided class',title=f"Correct {metric['correct']}/1279")
  for i,c in enumerate(classes):
   fpr,tpr,_=roc_curve(y==i,pp[:,i]);axs[1].plot(fpr,tpr,label=c);precision,recall,_=precision_recall_curve(y==i,pp[:,i]);axs[2].step(recall,precision,where='post',label=c)
  axs[1].set(xlabel='False-positive rate',ylabel='Sensitivity',title='One-vs-rest ROC');axs[2].set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision–recall');axs[2].legend(fontsize=7)
  fig.suptitle(name+' — reused Mendeley image diagnostic',fontsize=11);fig.text(.02,.01,'All 1,279 test images overlap the existing collection. Image-level results; not independent patient validation.',fontsize=9);fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
 table(out/'recomputed_metrics.csv',summary)
 fig,axs=plt.subplots(1,2,figsize=(13,6));labels=[m['model'].replace('kevin_20261011__','R1 ').replace('kevin_round2_20261011__','R2 ') for m in summary]
 for ax,key in zip(axs,['accuracy','balanced_accuracy']):ax.barh(labels,[m[key]*100 for m in summary]);ax.set(xlim=(0,100),xlabel=key+' (%)');ax.grid(axis='x',alpha=.2)
 fig.suptitle('Frozen stage models — all source images already present in legacy data');fig.tight_layout();fig.savefig(out/'all_stage_models.png',dpi=180);plt.close(fig)
 return {'stage_models':len(summary),'test_images_per_model':1279,'independent_external':False}
def training(out):
 out.mkdir(parents=True,exist_ok=True);folders=['publication/training','round2_publication/artifacts/stage/training','round2_publication/artifacts/clinical/training'];index=[]
 for folder in folders:
  for p in sorted((ROOT/'docs/kevin_paper_20261011'/folder).rglob('history.csv')):
   rows=list(csv.DictReader(p.open()));fig,axs=plt.subplots(1,2,figsize=(13,4.6))
   for ax,kind in zip(axs,['loss','accuracy']):
    for key in rows[0]:
     if ('loss' in key or 'nll' in key) if kind=='loss' else ('accuracy' in key):
      vals=[(float(r['epoch']),float(r[key])) for r in rows if r.get(key) not in [None,'','None','null']]
      if vals:ax.plot(*zip(*vals),label=key)
    ax.set(xlabel='Epoch',ylabel=kind);ax.legend(fontsize=6);ax.grid(alpha=.2)
   name=folder.split('/')[0]+'_'+('clinical_' if 'clinical/' in folder else '')+p.parent.name
   fig.suptitle(name+' — stored unsmoothed history, no retraining');fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
   index.append(dict(source=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(rows),figure=name+'.png'))
 assert len(index)==12;dump(out/'index.json',index);return len(index)
def main():
 p=argparse.ArgumentParser();p.add_argument('--clinical-predictions',type=Path,required=True);a=p.parse_args();out=ROOT/'results/current';out.mkdir(parents=True,exist_ok=True)
 report={'clinical':clinical(a.clinical_predictions,out/'clinical'),'stage':stage(out/'mendeley'),'training_history_figures':training(out/'training')}
 report['figures']=len(list(out.rglob('*.png')));dump(out/'verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
