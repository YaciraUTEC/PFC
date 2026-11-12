"""Shared inventory and exact zero-action certificates for evaluation."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .models import ARCHS,load_agent
from .bc import SEEDS
from .evaluate import OUT,evaluate
from ..run import save
TEST_LEVELS=[0.,.02,.05,.10,.15,.20,.25,.28,.29,.295]
STRESS=[.296,.30,.35,.50]

def groups():
    result=[]
    for kind,names,folder in [('residual',ARCHS,'models'),('observer',ARCHS,'observers'),('reward',['csil','meirl','rime'],'reward_models')]:
        for name in names:
            entries=[]
            for seed in SEEDS:
                directory=OUT/folder/name/f'seed_{seed}';path=directory/('bc.pt' if kind=='observer' else 'best.pt')
                entries.append(dict(seed=seed,directory=str(directory),path=str(path)))
            result.append(dict(id=f'{kind}_{name}',kind=kind,name=name,entries=entries,intervention='effectiveness' if kind=='observer' else 'residual'))
    return result

def core():return json.loads((OUT/'core_selection.json').read_text())['config']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def actor_hash(m):
    h=hashlib.sha256();cfg={k:v for k,v in m.cfg.items() if k not in ['lr','csil_alpha','gae_lambda']};h.update(json.dumps(cfg,sort_keys=True).encode())
    for name,value in sorted(m.state_dict().items()):
        if name.startswith('critic.') or name=='log_std':continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    h.update(str(getattr(m,'gate_logit_threshold',0.)).encode());return h.hexdigest()

def always_off(m):
    layer=m.actor[-1];upper=float(layer.bias[5].detach().double()+layer.weight[5].detach().double().abs().sum())+1e-10
    return upper<=float(getattr(m,'gate_logit_threshold',0.)),upper

def config_for(group,strategy):
    cfg=core();cfg.update(intervention=group['intervention'],gate_mode='conservative' if strategy['judge'] else 'learned')
    cfg['judge_path']=str(OUT/'judge.pt') if strategy['judge'] else None
    if strategy.get('off'):cfg['gate_mode']='off'
    return cfg

def cached_evaluation(path,label,condition,seeds,levels,config,threshold=0.,cache=None,proposal_only=False):
    target=Path(label)
    if target.exists():return json.loads(target.read_text())
    start=time.time();m=None if path is None else load_agent(path)
    if m is not None:m.gate_logit_threshold=float(threshold)
    zero=path is None or config.get('gate_mode')=='off'
    certificate=None
    if m is not None:
        certified,upper=always_off(m);certificate=dict(upper_gate_logit=upper,threshold=float(threshold),certified_zero=certified)
        zero=zero or (certified and not proposal_only)
    # A proved or explicitly selected zero policy has exactly the base trajectory.
    effective_config=dict(config)
    if zero:
        m=None;effective_config.update(intervention='residual',gate_mode='learned',judge_path=None)
    actor='ZERO' if m is None else actor_hash(m)
    if m is not None and proposal_only:
        from .policies import ProposalPolicy
        m=ProposalPolicy(m);actor='PROPOSAL:'+actor
    key=json.dumps(dict(actor=actor,condition=condition,seeds=list(seeds),levels=list(levels),config=effective_config,
        judge_hash=sha(OUT/'judge.pt') if effective_config.get('judge_path') else None),sort_keys=True)
    replay=None
    if cache is not None and key in cache:
        source=cache[key];rows=json.loads(Path(source).read_text());reason='Exact same actor/options or mathematically certified zero policy'
    else:
        if m is None:
            from .replay_cache import remember
            rows,traces=evaluate(condition=condition,seeds=seeds,levels=levels,config=effective_config,capture=True,workers=8,fixed_workers=True)
            remember(condition,effective_config,rows,traces)
        elif not proposal_only:
            from .replay_cache import accelerated
            rows,replay=accelerated(m,condition,seeds,levels,effective_config)
        else:rows=evaluate(m,condition=condition,seeds=seeds,levels=levels,config=effective_config)
        source=None;reason=None
    save(target,rows)
    save(target.with_suffix('.meta.json'),dict(seconds=time.time()-start,cases=len(rows),checkpoint=None if path is None else str(path),checkpoint_sha256=None if path is None else sha(path),
        actor_certificate=certificate,zero_action_replay=replay,proposal_only=bool(proposal_only),explicit_off=bool(config.get('gate_mode')=='off'),reused_from=source,reuse_reason=reason,effective_configuration=effective_config))
    if cache is not None:cache[key]=str(target)
    return rows
