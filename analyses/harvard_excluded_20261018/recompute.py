"""Post-hoc subset analysis from saved aggregate confusion counts; no inference."""
from pathlib import Path
import json, hashlib, csv
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'docs/kevin_paper_20261011/round2_publication/artifacts/clinical/report/analysis.json'
KEPT = ('besga_reused_test', 'openneuro_reused_external')

def main():
    source = json.loads(SOURCE.read_text())
    assert set(source['cohorts']) == set(KEPT) | {'harvard_fresh_external'}
    rows=[]
    for cohort in KEPT:
        for model, saved in source['cohorts'][cohort]['models'].items():
            (tn, fp), (fn, tp) = saved['confusion_cn_ad']
            n = tn+fp+fn+tp
            assert n == saved['n_patients']
            assert tp+tn == saved['correct']
            sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
            row=dict(cohort=cohort,model=model,n_patients=n,n_ad=tp+fn,n_cn=tn+fp,
                     tn=tn,fp=fp,fn=fn,tp=tp,correct=tn+tp,accuracy=(tn+tp)/n,
                     sensitivity_ad=sensitivity,specificity_cn=specificity,
                     balanced_accuracy=(sensitivity+specificity)/2)
            for key in ('accuracy','sensitivity_ad','specificity_cn','balanced_accuracy'):
                assert abs(row[key]-saved[key])<1e-12,(cohort,model,key)
            rows.append(row)
    assert len(rows)==16
    assert len({r['model'] for r in rows})==8
    with (OUT/'recomputed_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    record=dict(analysis_type='post_hoc_cohort_exclusion',excluded_cohort='harvard_fresh_external',
                exclusion_reason='User requested exclusion after observing poor performance; no independent data-quality exclusion established.',
                retained_cohorts=list(KEPT),retained_unique_patients=26,fresh_external_patients_remaining=0,
                source_path=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                metrics_recomputed_from='saved aggregate confusion matrices',
                all_16_model_cohort_count_metrics_match_original=True,
                inference_rerun=False,training_rerun=False,original_results_preserved=True,
                pooled_accuracy_reported=False,
                limitation='Descriptive selected subset of reused cohorts, not new external validation or evidence of clinical superiority.')
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
