"""Paired evaluation, with one explicit persistent memory per active episode."""
import os,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from .env import LandingEnv,LEVELS,ZERO
from .models import tree_map
from ..run import save
OUT=Path(__file__).resolve().parents[2]/'runs/controlled_landing_round4'
_POOL=None;_KEY=None

class Scheduled(gym.Wrapper):
    def __init__(self,cases,controller,condition,guard,profile,teacher=False,compact=True):
        if controller=='historical_rpm':
            from .legacy import HistoricalRPMEnv
            env=HistoricalRPMEnv(condition=condition,landing_profile=profile)
        else:env=LandingEnv(controller=controller,condition=condition,guard=guard,landing_profile=profile)
        super().__init__(env)
        self.cases=cases;self.index=0;self.inactive=False;self.teacher=teacher;self.compact=compact
    def set_cases(self,cases):self.cases=cases;self.index=0;self.inactive=False
    def configure(self,condition,guard,profile,teacher):
        self.env.condition=condition;self.env.guard=guard
        self.env.landing_profile=profile;self.teacher=teacher
    def reset(self,seed=None,options=None):
        if self.index>=len(self.cases):
            self.inactive=True;return np.zeros(self.observation_space.shape,np.float32),{}
        self.seed,self.level=self.cases[self.index];self.index+=1
        o,i=self.env.reset(seed=int(self.seed),options=dict(level=float(self.level)))
        return o,i
    def step(self,action):
        if self.inactive:return np.zeros(self.observation_space.shape,np.float32),0.,False,False,{}
        if self.teacher:action=self.env.teacher_action()
        o,r,d,t,i=self.env.step(action)
        if d or t:i.update(seed=int(self.seed),level=float(self.level))
        if self.compact:i={k:v for k,v in i.items() if k not in ['state','action','next_state']} if d or t else {}
        return o,r,d,t,i

def close_pool():
    global _POOL,_KEY
    if _POOL is not None:_POOL.close()
    _POOL=None;_KEY=None

def evaluate(model=None,controller='updated',condition='estimated',seeds=range(440000,440030),
             levels=LEVELS,guard=True,profile='descent',teacher=False,workers=8):
    global _POOL,_KEY
    cases=[(int(s),float(l)) for l in levels for s in seeds]
    workers=min(int(os.environ.get('NICOL_EVAL_WORKERS',workers)),len(cases))
    key=(controller,workers)
    def factory(items):return lambda:Scheduled(items,controller,condition,guard,profile,teacher)
    if _POOL is None or key!=_KEY:
        close_pool();f=[factory(cases[j::workers]) for j in range(workers)]
        _POOL=SubprocVecEnv(f,start_method='spawn') if workers>1 else DummyVecEnv(f);_KEY=key
    _POOL.env_method('configure',condition,guard,profile,teacher)
    for j in range(workers):_POOL.env_method('set_cases',cases[j::workers],indices=j)
    obs=_POOL.reset();features=24 if controller=='historical_rpm' else 48
    history=np.zeros((workers,32,features),np.float32);history[:,-1]=obs[:,0,:features]
    state=None;rows=[]
    try:
        while len(rows)<len(cases):
            if model is None:actions=np.tile(ZERO,(workers,1))
            elif hasattr(model,'encoder'):
                with torch.no_grad():a,state=model.predict_actions(torch.as_tensor(obs[:,0],device=next(model.parameters()).device,dtype=next(model.parameters()).dtype),state)
                actions=a.cpu().numpy()
            else:actions=model.predict(history,deterministic=True)[0]
            obs,r,done,infos=_POOL.step(actions)
            history=np.concatenate([history[:,1:],obs[:,:,:features]],1);history[done]=0;history[done,-1]=obs[done,0,:features]
            if state is not None:
                mask=torch.as_tensor(~done,device=next(model.parameters()).device)
                state=tree_map(lambda x:x*mask.reshape(-1,*([1]*(x.ndim-1))),state)
            for d,i in zip(done,infos):
                if d:rows.append({k:v for k,v in i.items() if k not in ['terminal_observation','TimeLimit.truncated']})
    except BaseException:
        close_pool();raise
    return sorted(rows,key=lambda x:(x['level'],x['seed']))

def baseline_pilot():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    for controller in ['legacy','robust','updated_round3','updated']:
        for condition in ['exact','estimated','biased']:
            target=OUT/'pilot_governor_v2'/f'{controller}_{condition}.json'
            if target.exists():continue
            rows=evaluate(controller=controller,condition=condition,seeds=range(410010,410020),
                          levels=[0.,.10,.20,.25,.28,.29,.30,.35,.50])
            save(target,dict(seconds=time.time()-start,episodes=rows))
            print(controller,condition,{l:round(np.mean([r['stable_landing'] for r in rows if r['level']==l])*100,1) for l in sorted(set(r['level'] for r in rows))},flush=True)
    close_pool()
if __name__=='__main__':baseline_pilot()
