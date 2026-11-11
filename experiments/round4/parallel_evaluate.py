"""Two bounded evaluation workers;freeze precedes every final test access."""
import json,time,hashlib,datetime,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
from stable_baselines3 import PPO
from .models import ARCHS,Agent
from .inference import load_agent,load_historical
from .train import SEEDS
from .env import LEVELS,STRESS
from .evaluate import OUT,evaluate,close_pool
from .campaign import freeze,cached_eval,ROOT
from ..run import save
_CACHE={}
def initialize():
    os.environ['NICOL_EVAL_WORKERS']='8'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def job(item):
    kind,key=item
    if kind=='baseline':
        controller,condition=key;p=OUT/'test'/f'baseline_{controller}_{condition}.json'
        if not p.exists():
            ts=time.time();rr=evaluate(controller=controller,condition=condition,levels=LEVELS+STRESS)
            save(p,rr);save(p.with_suffix('.timing.json'),dict(seconds=time.time()-ts,episodes=len(rr)))
    elif kind=='original_profile':
        condition=key;p=OUT/'test'/f'metric_only_original_profile_{condition}.json'
        if not p.exists():save(p,evaluate(controller='updated_round3',condition=condition,profile='original',levels=[0.,.25,.29]))
    elif kind=='new':
        arch,seed=key
        for mode in ['bc','best','final']:
            m=load_agent(OUT/'models'/arch/f'seed_{seed}'/f'{mode}.pt')
            for condition in ['exact','estimated']:
                cached_eval(m,OUT/'test'/f'{arch}_{seed}_{mode}_{condition}.json',_CACHE,condition=condition,levels=LEVELS+STRESS)
            del m
    elif kind=='guard':
        arch=key;m=load_agent(OUT/'models'/arch/'seed_71'/'best.pt')
        for condition in ['exact','estimated']:
            cached_eval(m,OUT/'test'/f'{arch}_71_no_guard_{condition}.json',_CACHE,condition=condition,guard=False,levels=LEVELS)
        del m
    elif kind=='historical':
        seed=key
        for name in ['bc','manual','csil','meirl','rime']:
            for mode in (['bc'] if name=='bc' else ['best','final']):
                base=ROOT/'runs'/'irl_round3'/f'seed_{seed}'
                path=base/'bc.zip' if name=='bc' else base/name/f'{mode}.zip'
                label=name if mode!='final' else name+'_final'
                m=load_historical(path,device='cpu')
                for condition in ['exact','estimated']:
                    cached_eval(m,OUT/'historical'/f'round3_{seed}_{label}_{condition}.json',_CACHE,
                        controller='updated_round3',condition=condition,guard=False,levels=LEVELS+STRESS)
                del m
    elif kind=='v2':
        arch=key
        for mode in ['best','final']:
            m=load_historical(ROOT/'runs'/'dslpid_comparison_v2'/arch/f'{mode}.zip',device='cpu')
            label=arch if mode=='best' else arch+'_final'
            for condition in ['exact','estimated']:
                cached_eval(m,OUT/'historical'/f'v2_{label}_{condition}.json',_CACHE,controller='historical_rpm',
                    condition=condition,guard=False,levels=LEVELS+STRESS)
            del m
    return dict(kind=kind,key=key)
def main():
    frozen=freeze();start=time.time();(OUT/'test').mkdir(exist_ok=True);(OUT/'historical').mkdir(exist_ok=True)
    for r in frozen['checkpoints']:assert hashlib.sha256((OUT/r['path']).read_bytes()).hexdigest()==r['sha256']
    jobs=[('baseline',(c,k)) for c in ['updated','updated_round3','robust','legacy'] for k in ['exact','estimated','biased']]
    jobs += [('original_profile',c) for c in ['exact','estimated']]
    jobs += [('new',(a,s)) for s in SEEDS for a in ARCHS]
    jobs += [('guard',a) for a in ARCHS]
    jobs += [('historical',s) for s in [11,22,33,44,55]]
    jobs += [('v2',a) for a in ARCHS]
    done=[]
    with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        fs=[pool.submit(job,j) for j in jobs]
        for f in as_completed(fs):
            result=f.result();done.append(result)
            save(OUT/'evaluation_campaign_progress.json',dict(completed=len(done),total=len(jobs),jobs=done,seconds=time.time()-start))
            print('EVALUATION_COMPLETE',result,'total',len(done),'/',len(jobs),flush=True)
    for r in frozen['checkpoints']:assert hashlib.sha256((OUT/r['path']).read_bytes()).hexdigest()==r['sha256']
    record=dict(seconds=time.time()-start,models_unchanged=True,workers=2,joint_evaluation=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(OUT/'test_complete.json',record);save(OUT/'historical_complete.json',record)
if __name__=='__main__':main()
