"""Runtime choices and production candidates fixed on validation, before test."""
import json,time
import numpy as np
from .inventory import groups,core,config_for,cached_evaluation
from .evaluate import OUT,close_pool
from ..run import save
CALIBRATION_LEVELS=[0.,.02,.25,.28,.29,.295]
STRATEGIES=[dict(id='learned',threshold=0.,judge=False,off=False),dict(id='strict',threshold=1.5,judge=False,off=False),dict(id='conservative',threshold=0.,judge=True,off=False),dict(id='judge_only',threshold=0.,judge=True,off=False,shadow=True),dict(id='off',threshold=0.,judge=False,off=True)]

def keyed(rows):return {(r['condition'],r['level'],r['seed']):r for r in rows}
def score(rows,base):
    reference=keyed(base);severe=[r for r in rows if r['condition']=='estimated' and r['level']>=.28]
    risk=[r for r in rows if r['condition']=='exact' or r['level']<=.25]
    losses=sum(reference[(r['condition'],r['level'],r['seed'])]['stable_landing'] and not r['stable_landing'] for r in risk)
    gains=np.mean([int(r['stable_landing'])-int(reference[(r['condition'],r['level'],r['seed'])]['stable_landing']) for r in severe])
    return dict(risk_losses=int(losses),primary_gain=float(gains),success=float(np.mean([r['stable_landing'] for r in severe])),
        effort=float(np.mean([r.get('q_delta_energy',0.) for r in rows])),mean_steps=float(np.mean([r['episode_steps'] for r in rows])))

def main():
    from .verification import run_tests
    run_tests()
    if not (OUT/'observer_fallback_selection.json').exists():
        from .observer_fallback import main as reserve_observer
        reserve_observer()
    start=time.time();cache={};base=[];result=[]
    for condition in ['exact','estimated']:
        rr=cached_evaluation(None,OUT/'runtime_selection'/f'base_{condition}.json',condition,range(528000,528008),CALIBRATION_LEVELS,core(),cache=cache)
        base.extend([dict(r,condition=condition) for r in rr])
    for group in groups():
        target=OUT/'runtime_selection'/f"{group['id']}.json"
        if target.exists():result.append(json.loads(target.read_text()));continue
        options=[]
        for strategy in STRATEGIES:
            rows=[];by_seed=[]
            for entry in group['entries']:
                selected=[]
                for condition in ['exact','estimated']:
                    path=OUT/'runtime_selection'/'cases'/f"{group['id']}_{entry['seed']}_{strategy['id']}_{condition}.json"
                    rr=cached_evaluation(entry['path'],path,condition,range(528000,528008),CALIBRATION_LEVELS,config_for(group,strategy),threshold=strategy['threshold'],cache=cache,proposal_only=strategy.get('shadow',False))
                    selected.extend([dict(r,condition=condition,training_seed=entry['seed']) for r in rr])
                rows.extend(selected);by_seed.append(dict(seed=entry['seed'],**score(selected,base)))
            options.append(dict(strategy=strategy,**score(rows,base),by_seed=by_seed))
        # Prefer complete abstention to an equally successful active strategy.
        eligible=[r for r in options if r['risk_losses']==0]
        chosen=max(eligible,key=lambda r:(r['primary_gain'],int(r['strategy']['off']),-r['effort'],-r['mean_steps']))
        strategy=chosen['strategy'];screen=[];screenbase=[]
        risk_levels=[0.,.02,.05,.10,.15,.20,.25,.28,.29,.295]
        for condition in ['exact','estimated']:
            levels=risk_levels if condition=='exact' else risk_levels[:7]
            bb=cached_evaluation(None,OUT/'runtime_selection'/f'full_base_{condition}.json',condition,range(528000,528032),levels,core(),cache=cache)
            screenbase.extend([dict(r,condition=condition) for r in bb])
        if not strategy['off']:
            for entry in group['entries']:
                for condition in ['exact','estimated']:
                    levels=risk_levels if condition=='exact' else risk_levels[:7]
                    path=OUT/'runtime_selection'/'screen'/f"{group['id']}_{entry['seed']}_{condition}.json"
                    rr=cached_evaluation(entry['path'],path,condition,range(528000,528032),levels,config_for(group,strategy),threshold=strategy['threshold'],cache=cache,proposal_only=strategy.get('shadow',False))
                    screen.extend([dict(r,condition=condition) for r in rr])
            ref=keyed(screenbase);losses=sum(ref[(r['condition'],r['level'],r['seed'])]['stable_landing'] and not r['stable_landing'] for r in screen)
            if losses:strategy=next(s for s in STRATEGIES if s['off'])
        else:losses=0
        record=dict(group=group,options=options,chosen_strategy=strategy,pre_screen_choice=chosen['strategy'],full_screen_losses=int(losses),
            rule='Highest severe gain with zero paired exact/mild validation losses; prefer abstention on tie; any full32route risk regression forces abstention.')
        save(target,record);result.append(record);print('RUNTIME_SELECTED',group['id'],strategy['id'],'gain',chosen['primary_gain'],'elapsed',round(time.time()-start),flush=True)
    candidates=[]
    for kind in ['residual','observer','reward']:
        eligible=[r for r in result if r['group']['kind']==kind and not r['chosen_strategy']['off']]
        if not eligible:continue
        def gain(r):return next(o['primary_gain'] for o in r['options'] if o['strategy']['id']==r['chosen_strategy']['id'])
        best=max(eligible,key=gain)
        if gain(best)<=0:continue
        option=next(o for o in best['options'] if o['strategy']['id']==best['chosen_strategy']['id'])
        seed=max(option['by_seed'],key=lambda x:(x['primary_gain'],-x['risk_losses'],-x['effort']))['seed']
        entry=next(e for e in best['group']['entries'] if e['seed']==seed)
        candidates.append(dict(id=best['group']['id'],kind=kind,path=entry['path'],seed=seed,strategy=best['chosen_strategy'],config=config_for(best['group'],best['chosen_strategy'])))
    save(OUT/'runtime_selection_complete.json',dict(seconds=time.time()-start,groups=result,production_candidates=candidates,final_test_unopened=True))
    close_pool()
    from .oracle_proposals import main as audit_proposals
    audit_proposals()
if __name__=='__main__':main()
