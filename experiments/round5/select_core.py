"""Core selection entirely on predeclared validation route IDs."""
import json,time
import numpy as np
from .evaluate import OUT,evaluate,close_pool
from ..run import save
LEVELS=[0.,.02,.05,.10,.20,.25,.28,.29,.295]

def run(label,cfg,seeds,directory,levels):
    combined=[]
    for condition in ['exact','estimated']:
        target=OUT/directory/f'{label}_{condition}.json'
        if target.exists():rows=json.loads(target.read_text())
        else:rows=evaluate(condition=condition,seeds=seeds,levels=levels,config=cfg);save(target,rows)
        combined.extend([dict(r,condition=condition) for r in rows])
    return combined

def summarize(rows,base):
    ref={(r['condition'],r['seed'],r['level']):r for r in base}
    risk=[r for r in rows if r['condition']=='exact' or r['level']<=.25]
    losses=sum(ref[(r['condition'],r['seed'],r['level'])]['stable_landing'] and not r['stable_landing'] for r in risk)
    severe=[r for r in rows if r['condition']=='estimated' and r['level']>=.28]
    return dict(risk_losses=int(losses),severe_success=float(np.mean([r['stable_landing'] for r in severe])),
        overall_success=float(np.mean([r['stable_landing'] for r in rows])),mean_steps=float(np.mean([r['episode_steps'] for r in rows])),
        by_condition_level={f'{c}/{l}':float(np.mean([r['stable_landing'] for r in rows if r['condition']==c and r['level']==l])) for c in ['exact','estimated'] for l in sorted({r['level'] for r in rows})})

def main():
    start=time.time();basecfg=dict(estimator='original',readiness=False,emergency='legacy',reserve=.005)
    variants=[('original',basecfg),('ready',dict(basecfg,readiness=True)),
        ('hybrid',dict(basecfg,estimator='hybrid',readiness=True)),
        ('hybrid_no_ready',dict(basecfg,estimator='hybrid',readiness=False)),
        ('adaptive',dict(basecfg,estimator='adaptive',readiness=True)),
        ('adaptive_no_ready',dict(basecfg,estimator='adaptive')),
        ('damped',dict(basecfg,estimator='adaptive',readiness=True,emergency='damped')),
        ('reserve003',dict(basecfg,estimator='adaptive',readiness=True,emergency='damped',reserve=.003)),
        ('reserve001',dict(basecfg,estimator='adaptive',readiness=True,emergency='damped',reserve=.001)),
        ('reserve000',dict(basecfg,estimator='adaptive',readiness=True,emergency='damped',reserve=0.))]
    candidates=[];stage1base=None
    try:
        for label,cfg in variants:
            rows=run(label,cfg,range(527100,527116),'core_hpo_shared_sensors',[0.,.02,.25,.28,.29,.295])
            if label=='original':stage1base=rows
            summary=summarize(rows,stage1base);candidates.append(dict(label=label,config=cfg,**summary))
            print('CORE_HPO',label,summary['risk_losses'],summary['severe_success'],'elapsed',round(time.time()-start),flush=True)
        ordered=sorted(candidates,key=lambda x:(-x['risk_losses'],x['severe_success'],x['overall_success'],-x['mean_steps']),reverse=True)
        finalists=[('original',basecfg)]+[(r['label'],r['config']) for r in ordered if r['label']!='original']
        final=[];base=None
        for label,cfg in finalists:
            rows=run(label,cfg,range(528000,528032),'core_validation_shared_sensors',LEVELS)
            if label=='original':base=rows
            result=dict(label=label,config=cfg,**summarize(rows,base));final.append(result)
            print('CORE_VALIDATION',label,result['risk_losses'],result['severe_success'],'elapsed',round(time.time()-start),flush=True)
        eligible=[r for r in final if r['risk_losses']==0]
        selected=max(eligible,key=lambda x:(x['severe_success'],x['overall_success'],-x['mean_steps']))
        record=dict(config=selected['config'],label=selected['label'],rule='No paired regression on exact or estimated0..25percent validation cases; highest severe estimated success,overall success,then duration.',
            hpo=candidates,validation=final,seconds=time.time()-start,final_test_unopened=True,engineering_version=4,contact_hz=960)
        save(OUT/'core_selection.json',record);print('CORE_SELECTED',json.dumps(record['config']),flush=True)
    finally:close_pool()
if __name__=='__main__':main()
