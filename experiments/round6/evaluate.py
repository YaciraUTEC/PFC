"""Paired case executor. No policy/test metrics feed runtime decisions."""
import json,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from ..round4.models import tree_map
from ..round5.diagnostics import attach_impact_diagnostics
from .env import LandingEnv,ZERO
OUT=Path(__file__).resolve().parents[2]/'runs/intervention_round6'
LEVELS=[0.,.02,.10,.25,.28,.29,.295,.296,.30,.35,.40,.50]
SEEDS=[201,202,203,204,205]
def serial(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,np.generic):return obj.item()
    raise TypeError(type(obj).__name__)
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(value,default=serial,indent=2),encoding='utf-8');temp.replace(path)
class Scheduled(gym.Wrapper):
    def __init__(self,cases,condition,config):
        super().__init__(LandingEnv(condition=condition,config=config));self.cases=cases;self.index=0;self.inactive=False
    def reset(self,**kw):
        if self.index>=len(self.cases):self.inactive=True;return np.zeros(self.observation_space.shape,np.float32),{}
        seed,level=self.cases[self.index];self.index+=1
        o,i=self.env.reset(seed=int(seed),options=dict(level=float(level)));attach_impact_diagnostics(self.env);return o,i
    def step(self,a):
        if self.inactive:return np.zeros(self.observation_space.shape,np.float32),0.,False,False,{}
        o,r,d,t,i=self.env.step(a)
        if d or t:i={k:v for k,v in i.items() if k not in ['state','action','next_state','executed_action']};i['seed']=self.env.case_seed
        else:i={}
        return o,r,d,t,i

def evaluate(config,condition='estimated',seeds=range(611000,611008),levels=LEVELS,model=None,workers=8,case_pairs=None,capture=False):
    cases=[(s,l) for l in levels for s in seeds] if case_pairs is None else list(case_pairs)
    n=min(workers,len(cases)); factories=[lambda cc=cases[j::n]:Scheduled(cc,condition,config) for j in range(n)]
    v=SubprocVecEnv(factories,start_method='spawn') if n>1 else DummyVecEnv(factories)
    rows=[];state=None;start=time.time();histories=[[] for _ in range(n)];traces={}
    try:
        obs=v.reset()
        while len(rows)<len(cases):
            if capture:
                for j in range(n):histories[j].append(obs[j,0].copy())
            if model is None:a=np.tile(ZERO,(n,1))
            else:
                p=next(model.parameters())
                with torch.no_grad():a,state=model.predict_actions(torch.as_tensor(obs[:,0],device=p.device,dtype=p.dtype),state)
                a=a.cpu().numpy()
            obs,reward,done,infos=v.step(a)
            if state is not None:
                mask=torch.as_tensor(~done,device=p.device)
                state=tree_map(lambda x:x*mask.reshape(n,*([1]*(x.ndim-1))),state)
            for j,(d,i) in enumerate(zip(done,infos)):
                if d:
                    rows.append({k:v for k,v in i.items() if k not in ['terminal_observation','TimeLimit.truncated']})
                    if capture:traces[(i['level'],i['seed'])]=np.stack(histories[j]);histories[j]=[]
        assert len(rows)==len(cases)
    finally:v.close()
    result=dict(episodes=sorted(rows,key=lambda r:(r['level'],r['seed'])),seconds=time.time()-start)
    return (result,traces) if capture else result

def rates(rows):
    return {str(l):dict(success=sum(r['stable_landing'] for r in rows if r['level']==l),n=sum(r['level']==l for r in rows),
        percent=100*float(np.mean([r['stable_landing'] for r in rows if r['level']==l]))) for l in sorted({r['level'] for r in rows})}
