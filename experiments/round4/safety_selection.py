"""Prospective mild-fault validation safeguard,run before freezing/test access.
Known earlier issue:good10percent did not imply good2percent. Check0/2/5 directly.
"""
import json,time,shutil,os,datetime
from concurrent.futures import ProcessPoolExecutor,as_completed
from .models import ARCHS,Agent
from .inference import load_agent
from .train import SEEDS
from .evaluate import OUT,evaluate
from .campaign import actor_hash
from ..run import save

def initialize():
    import torch
    os.environ['NICOL_EVAL_WORKERS']='8';torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def mild_regressions(baseline,selected):
    key=lambda r:(r['condition'],r['seed'],r['level'])
    b={key(r):r for r in baseline};s={key(r):r for r in selected}
    if set(b)!=set(s):raise ValueError('Mild validation cases must be paired')
    return [dict(condition=k[0],seed=k[1],level=k[2]) for k in b if b[k]['stable_landing'] and not s[k]['stable_landing']]

def job(item):
    arch,seed=item;directory=OUT/'models'/arch/f'seed_{seed}'
    target=OUT/'selection_safety'/f'{arch}_{seed}.json'
    if target.exists():return json.loads(target.read_text())
    backup=directory/'before_mild_safety_best.pt'
    base=load_agent(directory/'bc.pt');best=load_agent(backup if backup.exists() else directory/'best.pt')
    identical=actor_hash(base)==actor_hash(best)
    baseline=[];selected=[]
    if not identical:
        for condition in ['exact','estimated']:
            rr=evaluate(base,seeds=range(430000,430008),levels=[0.,.02,.05],condition=condition)
            baseline.extend([dict(r,condition=condition) for r in rr])
            rr=evaluate(best,seeds=range(430000,430008),levels=[0.,.02,.05],condition=condition)
            selected.extend([dict(r,condition=condition) for r in rr])
    regressions=mild_regressions(baseline,selected)
    rolled_back=bool(regressions)
    complete=json.loads((directory/'complete.json').read_text())
    if rolled_back:
        if not backup.exists():shutil.copy2(directory/'best.pt',backup)
        temp=directory/'best.safety_tmp.pt'
        shutil.copy2(directory/'bc.pt',temp);os.replace(temp,directory/'best.pt')
        complete.setdefault('before_safety_selected_steps',complete['selected_steps'])
        complete.setdefault('before_safety_best_score',complete['best_score'])
        complete.update(selected_steps=0,best_score=complete['base_score'],selection_safety_override=True)
    complete['selection_safety_checked']=True
    save(directory/'complete.json',complete)
    result=dict(architecture=arch,seed=seed,identical_actor=identical,regressions=regressions,
        rolled_back_to_bc=rolled_back,baseline=baseline,selected_before_check=selected,
        rule='Any routewise loss at0/2/5percent RPM in exact/estimated validation falls back to own BC',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(target,result);return result

def ensure_selection():
    if (OUT/'selection_safety_complete.json').exists():return
    assert not (OUT/'frozen_before_test.json').exists()
    assert all((OUT/'models'/a/f'seed_{s}'/'complete.json').exists() for a in ARCHS for s in SEEDS)
    start=time.time();jobs=[(a,s) for s in SEEDS for a in ARCHS];results=[]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        fs=[pool.submit(job,item) for item in jobs]
        for f in as_completed(fs):
            result=f.result();results.append(result)
            save(OUT/'selection_safety_progress.json',dict(completed=len(results),total=len(jobs),seconds=time.time()-start))
            print('MILD_SELECTION_CHECK',result['architecture'],result['seed'],'rollback',result['rolled_back_to_bc'],flush=True)
    save(OUT/'selection_safety_complete.json',dict(seconds=time.time()-start,models=len(jobs),
        rollbacks=sum(r['rolled_back_to_bc'] for r in results),identical_skipped=sum(r['identical_actor'] for r in results),
        simulated_cases=sum(len(r['baseline'])+len(r['selected_before_check']) for r in results)))
if __name__=='__main__':ensure_selection()
