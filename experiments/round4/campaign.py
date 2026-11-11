"""Freeze all30 selected agents,then open paired final tests once."""
import argparse,datetime,json,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from .evaluate import OUT,evaluate,close_pool
from .models import ARCHS,Agent
from .inference import load_agent,load_historical
from .train import SEEDS
from .env import LEVELS,STRESS
from ..run import save

ROOT=Path(__file__).resolve().parents[2]
def actor_hash(model):
    if hasattr(model,'encoder'):
        d={k:v for k,v in model.state_dict().items() if not k.startswith('critic.') and k!='log_std'}
    else:
        # Exclude only critic branches;attention value projections belong to actor.
        critic_prefixes=('value_net.','mlp_extractor.value_net.','vf_features_extractor.')
        d={k:v for k,v in model.policy.state_dict().items() if not k.startswith(critic_prefixes) and k!='log_std'}
    h=hashlib.sha256()
    config=model.cfg if hasattr(model,'cfg') else dict(policy_kwargs=model.policy_kwargs,observation_shape=model.observation_space.shape)
    h.update(json.dumps(config,sort_keys=True,default=str).encode())
    for k in sorted(d):h.update(k.encode());h.update(d[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()
def freeze():
    target=OUT/'frozen_before_test.json'
    if target.exists():return json.loads(target.read_text())
    assert all((OUT/'models'/a/f'seed_{s}'/'complete.json').exists() for a in ARCHS for s in SEEDS),'Finish all model selections first'
    from .safety_selection import ensure_selection
    ensure_selection()
    records=[]
    for a in ARCHS:
        for seed in SEEDS:
            d=OUT/'models'/a/f'seed_{seed}'
            for mode in ['bc','best','final']:
                p=d/f'{mode}.pt'
                records.append(dict(arch=a,seed=seed,mode=mode,path=str(p.relative_to(OUT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    historical=[]
    for sid in [11,22,33,44,55]:
        directory=ROOT/'runs'/'irl_round3'/f'seed_{sid}'
        historical.append((directory/'bc.zip',f'round3_bc_{sid}'))
        for method in ['manual','csil','meirl','rime']:
            for mode in ['best','final']:historical.append((directory/method/f'{mode}.zip',f'round3_{method}_{mode}_{sid}'))
    for arch in ARCHS:
        for mode in ['best','final']:historical.append((ROOT/'runs'/'dslpid_comparison_v2'/arch/f'{mode}.zip',f'v2_{arch}_{mode}'))
    for path,label in historical:
        records.append(dict(arch=label,seed=None,mode='historical',
            path=Path(os.path.relpath(path,OUT)).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    sources=sorted(set((ROOT/'experiments').rglob('*.py')) | set((ROOT/'gym_pybullet_drones').rglob('*.py')))
    source_hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    record=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoints=records,source_hashes_before_test=source_hashes,
        evaluation_reference=dict(device='cpu',network_dtype='float64',checkpoint_dtype='original float32',action_dtype='float32',workers_per_process=8))
    save(target,record);return record
def cached_eval(model,target,cache,**kw):
    if target.exists():return
    parameter=next(model.parameters() if hasattr(model,'encoder') else model.policy.parameters())
    key=actor_hash(model)+json.dumps(dict(kw,device=str(parameter.device),dtype=str(parameter.dtype)),sort_keys=True,default=list)
    if key in cache:
        source=cache[key];save(target,json.loads(source.read_text()))
        save(target.with_suffix('.reuse.json'),dict(source=str(source),reason='identical deterministic actor/config and paired cases'))
    else:
        ts=time.time();rr=evaluate(model,**kw);save(target,rr);cache[key]=target
        save(target.with_suffix('.timing.json'),dict(seconds=time.time()-ts,episodes=len(rr)))
def verify_frozen(frozen):
    for item in frozen['checkpoints']:
        assert hashlib.sha256((OUT/item['path']).read_bytes()).hexdigest()==item['sha256']
def test_new():
    frozen=freeze();verify_frozen(frozen);start=time.time()
    from .parallel_evaluate import job,initialize
    initialize()
    for controller in ['legacy','robust','updated_round3','updated']:
        for condition in ['exact','estimated','biased']:job(('baseline',(controller,condition)))
    for condition in ['exact','estimated']:job(('original_profile',condition))
    for seed in SEEDS:
        for arch in ARCHS:job(('new',(arch,seed)))
    for arch in ARCHS:job(('guard',arch))
    close_pool();verify_frozen(frozen)
    save(OUT/'test_complete.json',dict(seconds=time.time()-start,models_unchanged=True,joint_evaluation=False))
def test_historical():
    frozen=freeze();verify_frozen(frozen);start=time.time()
    from .parallel_evaluate import job,initialize
    initialize()
    for seed in [11,22,33,44,55]:job(('historical',seed))
    for arch in ARCHS:job(('v2',arch))
    close_pool();verify_frozen(frozen)
    save(OUT/'historical_complete.json',dict(seconds=time.time()-start,models_unchanged=True,joint_evaluation=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all','new','historical','freeze'],default='all');args=p.parse_args()
    if args.phase=='freeze':freeze()
    else:
        freeze()
        if args.phase in ['all','new']:test_new()
        if args.phase in ['all','historical']:test_historical()
