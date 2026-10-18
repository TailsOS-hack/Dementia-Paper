"""Recompute and audit the historical STS manuscript tables without inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import full_cohort, supporting, training


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir',type=Path,default=Path('results/sts'))
    args=parser.parse_args()
    root=args.repo_root.resolve();out=args.output_dir.resolve()
    a=training.generate(root,out/'training')
    b=full_cohort.generate(root,out/'full_cohort')
    c=supporting.run(root,out/'supporting')
    report={'status':'verified','training_runs':a['run_count'],
        'completed_epochs':a['total_completed_epochs'],'full_test_images':b['n_images'],
        'full_test_models':len(b['models']),'supporting_cohort_model_rows':c['cohort_model_rows'],
        'paired_contrasts_verified':c['paired_contrasts_verified'],
        'training_or_inference_performed':False,
        'uncertainty':'Historical paired intervals preserved; point metrics and discordances independently recomputed.',
        'historical_primary':'Matched 1058 images','full_test_role':'Post-hoc descriptive supplement containing primary1058'}
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
