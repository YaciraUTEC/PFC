"""Close the continuous-governor alternative before opening final test."""
import json,time
import numpy as np
from .evaluate import OUT,evaluate,close_pool
from .inventory import core,TEST_LEVELS
from .select_core import summarize
from ..run import save

def main():
    start=time.time();selected=json.loads((OUT/'core_selection.json').read_text());cfg=core();original=dict(estimator='original',readiness=False,emergency='legacy',reserve=.005)
    collections={}
    for label,parameters,source in [('original',original,'original'),('selected',cfg,selected['label']),('balanced',dict(cfg,emergency='balanced'),None)]:
        combined=[]
        for condition in ['exact','estimated']:
            target=OUT/'structural_challenger'/f'{label}_{condition}.json'
            if target.exists():rows=json.loads(target.read_text())
            else:
                old=OUT/'core_validation_shared_sensors'/f'{source}_{condition}.json' if source else None
                rows=json.loads(old.read_text()) if old and old.exists() else []
                missing=[l for l in TEST_LEVELS if l not in {r['level'] for r in rows}]
                if missing:rows.extend(evaluate(condition=condition,seeds=range(528000,528032),levels=missing,config=parameters))
                save(target,rows)
            combined.extend([dict(r,condition=condition) for r in rows])
        collections[label]=combined
    before=summarize(collections['selected'],collections['original']);after=summarize(collections['balanced'],collections['original'])
    eligible=after['risk_losses']==0 and after['severe_success']>=before['severe_success']
    result=dict(config=dict(cfg,emergency='balanced'),selected_summary=before,challenger_summary=after,production_eligible=bool(eligible),
        selection_rule='No protected validation regression and severe success at least equal to selected core;otherwise retain only as final ablation.',seconds=time.time()-start,final_test_unopened=True)
    save(OUT/'structural_challenger.json',result);close_pool();print('STRUCTURAL_CHALLENGER',json.dumps(result),flush=True)
if __name__=='__main__':main()
