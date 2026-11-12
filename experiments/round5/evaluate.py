"""Paired evaluation: fixed CPU64 network inference, float32 action boundary."""
import os,json,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from .env import LandingEnv,ZERO,LEVELS,STRESS
from .estimator import make_estimator
from .diagnostics import attach_impact_diagnostics,InterventionTrace
from ..round4.models import tree_map
OUT=Path(__file__).resolve().parents[2]/'runs/intervention_round5'
_POOL=None;_WORKERS=None

class Scheduled(gym.Wrapper):
    def __init__(self):
        super().__init__(LandingEnv());self.cases=[];self.index=0;self.inactive=False;self.teacher=False
    def configure(self,cases,condition,config,teacher=False):
        e=self.env;oldjudge=e.config.get('judge_path');e.config.update(config);e.condition=condition
        e.c._emergency_policy=e.config['emergency'];e.c._thrust_reserve=e.config['reserve']
        noise=.002*e.config['noise_multiplier'] if e.sensor_noise else 0.
        e._sensor_sigma=noise
        e.estimator=make_estimator(e.c,e.config['estimator'],velocity_noise=0.,gyro_noise=0.)
        e.descent.enabled=e.config['readiness']
        if e.config.get('judge_path')!=oldjudge:
            if e.config.get('judge_path'):
                from .judge import BenefitJudge
                e.judge=BenefitJudge.load(e.config['judge_path'])
            else:e.judge=None
        self.cases=cases;self.index=0;self.inactive=False;self.teacher=teacher
    def reset(self,seed=None,options=None):
        if self.index>=len(self.cases):
            self.inactive=True;return np.zeros(self.observation_space.shape,np.float32),{}
        self.seed,self.level=self.cases[self.index];self.index+=1
        result=self.env.reset(seed=int(self.seed),options=dict(level=float(self.level)))
        attach_impact_diagnostics(self.env);self.trace=InterventionTrace();return result
    def step(self,action):
        if self.inactive:return np.zeros(self.observation_space.shape,np.float32),0.,False,False,{}
        if self.teacher:action=self.env.teacher_action()
        o,r,d,t,i=self.env.step(action)
        self.trace.update(self.env,i,action)
        if d or t:i.update(self.trace.summary())
        if d or t:i={k:v for k,v in i.items() if k not in ['state','action','next_state']};i.update(seed=int(self.seed),level=float(self.level))
        else:i={}
        return o,r,d,t,i

def close_pool():
    global _POOL,_WORKERS
    if _POOL is not None:_POOL.close()
    _POOL=None;_WORKERS=None

def resolved_config(config=None):
    cfg=dict(estimator='original',readiness=True,emergency='legacy',reserve=.005,intervention='residual',gate_mode='learned',gate_threshold=0.,noise_multiplier=1.,fault_clock=None,judge_path=None,physics_hz=240,contact_hz=960,solver_iterations=200,solver_residual=1e-12,sensor_model='shared_velocity_gyro')
    cfg.update(config or {});return cfg

def evaluate(model=None,condition='estimated',seeds=range(528000,528032),levels=LEVELS,config=None,teacher=False,workers=8,capture=False,case_pairs=None,fixed_workers=False):
    global _POOL,_WORKERS
    cases=[(int(s),float(l)) for l in levels for s in seeds] if case_pairs is None else [(int(s),float(l)) for s,l in case_pairs]
    assert cases and (not capture or (model is None and not teacher))
    workers=workers if fixed_workers else min(workers,len(cases))
    if _POOL is None or _WORKERS!=workers:
        close_pool();fs=[Scheduled for _ in range(workers)]
        _POOL=SubprocVecEnv(fs,start_method='spawn') if workers>1 else DummyVecEnv(fs);_WORKERS=workers
    # Send full defaults to prevent preceding job options leaking.
    cfg=resolved_config(config)
    for j in range(workers):_POOL.env_method('configure',cases[j::workers],condition,cfg,teacher,indices=j)
    obs=_POOL.reset();state=None;rows=[];histories=[[] for _ in range(workers)];traces={}
    try:
        while len(rows)<len(cases):
            if capture:
                for j in range(workers):histories[j].append(obs[j,0].copy())
            if model is None:actions=np.tile(ZERO,(workers,1))
            else:
                p=next(model.parameters())
                with torch.no_grad():a,state=model.predict_actions(torch.as_tensor(obs[:,0],device=p.device,dtype=p.dtype),state)
                actions=a.cpu().numpy()
            obs,r,done,infos=_POOL.step(actions)
            if state is not None:
                mask=torch.as_tensor(~done,device=p.device)
                state=tree_map(lambda x:x*mask.reshape(-1,*([1]*(x.ndim-1))),state)
            for j,(d,i) in enumerate(zip(done,infos)):
                if d:
                    rows.append({k:v for k,v in i.items() if k not in ['terminal_observation','TimeLimit.truncated']})
                    if capture:traces[(i['level'],i['seed'])]=np.stack(histories[j]);histories[j]=[]
    except BaseException:close_pool();raise
    assert len(rows)==len(cases)
    result=sorted(rows,key=lambda x:(x['level'],x['seed']))
    return (result,[traces[(r['level'],r['seed'])] for r in result]) if capture else result
