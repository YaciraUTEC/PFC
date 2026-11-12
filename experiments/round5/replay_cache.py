"""Exact zero-action trajectory replay. Never approximate learned control.
A causal deterministic policy whose complete action stream is canonicalZERO on
base observations generates the same states,sensor RNG and terminal outcome.
If any action differs,the entire learned-policy case is simulated normally.
Reference inference is CPUfloat64,one tick at a time,batch8 with independent rows.
"""
import json,os,hashlib,uuid
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .evaluate import OUT,evaluate,resolved_config
from .env import ZERO

ROOT=Path(__file__).resolve().parents[2]
BANK=OUT/'zero_replay_bank'
_MEMORY=OrderedDict()
MAX_MEMORY_CASES=4096
def memo(key,value):
    _MEMORY[key]=value;_MEMORY.move_to_end(key)
    while len(_MEMORY)>MAX_MEMORY_CASES:_MEMORY.popitem(last=False)

def source_digest():
    paths=['experiments/round5/replay_cache.py','experiments/round5/evaluate.py','experiments/round5/env.py','experiments/round5/diagnostics.py','experiments/round5/estimator.py','experiments/round5/supervisor.py','experiments/round4/landing.py','experiments/round3/env.py','gym_pybullet_drones/control/DSLPIDControl.py']
    h=hashlib.sha256()
    paths+=['experiments/round4/estimator.py']
    paths+=sorted(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'gym_pybullet_drones').rglob('*') if p.suffix.lower() in ['.py','.urdf','.obj','.stl','.dae'])
    for p in sorted(set(paths)):h.update(p.encode());h.update((ROOT/p).read_bytes())
    return h.hexdigest()

def canonical(config):
    cfg=resolved_config(config);cfg.update(intervention='residual',gate_mode='learned',judge_path=None)
    if cfg['gate_threshold']<-1:raise ValueError('CanonicalZERO must actually be inactive')
    return cfg

def identity(condition,config):
    payload=dict(schema=1,source_sha256=source_digest(),condition=condition,config=canonical(config))
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

def filename(key,level,seed):return Path(BANK)/key/f'{float(level):.8f}_{int(seed)}.npz'
def frame_digest(frames):return hashlib.sha256(np.asarray(frames,dtype=np.float32).tobytes()).hexdigest()

def store_case(key,payload,row,frames):
    path=filename(key,row['level'],row['seed']);path.parent.mkdir(parents=True,exist_ok=True)
    frames=np.asarray(frames,np.float32)
    record=dict(identity=payload,row=row,frames_sha256=frame_digest(frames))
    temporary=path.with_name(path.stem+f'.tmp_{os.getpid()}_{uuid.uuid4().hex}.npz')
    np.savez_compressed(temporary,frames=frames,record=np.asarray(json.dumps(record)))
    try:os.replace(temporary,path)
    except PermissionError:
        # Another process may already have atomically written the same case.
        if not path.exists():raise
        temporary.unlink(missing_ok=True)
    frames.setflags(write=False);memo((str(Path(BANK)),key,row['level'],row['seed']),(row,frames))

def load_case(key,payload,level,seed):
    memkey=(str(Path(BANK)),key,float(level),int(seed))
    if memkey in _MEMORY:
        _MEMORY.move_to_end(memkey);return _MEMORY[memkey]
    path=filename(key,level,seed)
    if not path.exists():return None
    with np.load(path,allow_pickle=False) as z:frames=z['frames'];record=json.loads(str(z['record']))
    assert record['identity']==payload and record['frames_sha256']==frame_digest(frames),'Invalid zero-replay provenance'
    row=record['row'];assert row['seed']==seed and row['level']==level
    assert frames.dtype==np.float32 and frames.shape==(row['episode_steps'],80) and np.isfinite(frames).all()
    frames.setflags(write=False);memo(memkey,(row,frames));return _MEMORY[memkey]

def ensure(condition,seeds,levels,config):
    seeds=list(seeds);levels=list(levels);key,payload=identity(condition,config);missing=[]
    for level in levels:
        for seed in seeds:
            if load_case(key,payload,level,seed) is None:missing.append((seed,level))
    if missing:
        rows,frames=evaluate(condition=condition,config=canonical(config),case_pairs=missing,capture=True,workers=8,fixed_workers=True)
        for row,x in zip(rows,frames):store_case(key,payload,row,x)
    cases=[load_case(key,payload,l,s) for l in levels for s in seeds]
    assert all(c is not None for c in cases)
    return cases,dict(key=key,source_sha256=payload['source_sha256'],new_baseline_cases=len(missing))

def remember(condition,config,rows,frames):
    key,payload=identity(condition,config)
    for row,x in zip(rows,frames):store_case(key,payload,row,x)

@torch.no_grad()
def certify(model,cases,config):
    p=next(model.parameters());assert p.device.type=='cpu' and p.dtype==torch.float64 and not model.training
    torch.set_num_threads(1);width=min(8,len(cases));certified=np.zeros(len(cases),bool);mode=float(config.get('intervention','residual')=='effectiveness')
    for start in range(0,len(cases),width):
        selected=cases[start:start+width];lengths=[len(x) for _,x in selected];pending=np.ones(len(selected),bool);state=None
        for t in range(max(lengths)):
            if not pending.any():break
            obs=np.zeros((width,80),np.float64)
            for j,(_,frames) in enumerate(selected):
                if pending[j] and t<lengths[j]:obs[j]=frames[t];obs[j,-1]=mode
            actions,state=model.predict_actions(torch.as_tensor(obs,dtype=torch.float64),state);a=actions.cpu().numpy()
            for j in range(len(selected)):
                if not pending[j]:continue
                if not np.array_equal(a[j],ZERO):pending[j]=False
                elif t==lengths[j]-1:certified[start+j]=True;pending[j]=False
    return certified

def accelerated(model,condition,seeds,levels,config):
    seeds=list(seeds);levels=list(levels);cases,meta=ensure(condition,seeds,levels,config)
    off=certify(model,cases,config);rows=[];remaining=[]
    for zero,(row,frames) in zip(off,cases):
        if zero:rows.append(dict(row,configuration=resolved_config(config),zero_replay_certified=True))
        else:remaining.append((row['seed'],row['level']))
    if remaining:rows.extend(evaluate(model,condition=condition,config=config,case_pairs=remaining,workers=min(8,len(cases)),fixed_workers=True))
    rows=sorted(rows,key=lambda r:(r['level'],r['seed']))
    meta.update(trajectory_certified_zero_cases=int(off.sum()),fully_simulated_policy_cases=len(remaining),cases=len(cases),scope='Exact canonicalZERO at every tick on base trajectory,CPUfloat64 streaming;all other cases fully simulated;not an approximate model or shortened horizon.')
    return rows,meta
