"""Same explicitly seeded cases as serial evaluation, batched without shared memory."""
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv,VecEnvWrapper,DummyVecEnv
from .env import TrialEnv

_POOL=None
_POOL_KEY=None

def close_pool():
    global _POOL,_POOL_KEY
    if _POOL is not None:_POOL.close()
    _POOL=None;_POOL_KEY=None

class ScheduledCases(gym.Wrapper):
    def __init__(self,cases,controller,condition,mode):
        super().__init__(TrialEnv(controller=controller,condition=condition));self.cases=cases;self.index=0;self.mode=mode;self.inactive=False;self.compact_info=False
    def set_cases(self,cases):
        self.cases=cases;self.index=0;self.inactive=False
    def reset(self,seed=None,options=None):
        if self.index>=len(self.cases):
            self.inactive=True;return np.zeros(self.observation_space.shape,np.float32),{}
        self.case_seed,self.case_level=self.cases[self.index];self.index+=1
        self.peak_yaw_rate=0.;self.min_postfault_altitude=100.
        return self.env.reset(seed=self.case_seed,options=dict(level=self.case_level))
    def step(self,action):
        if self.inactive:return np.zeros(self.observation_space.shape,np.float32),0.,False,False,{}
        if self.mode=='teacher':action=self.env.teacher_action()
        obs,r,d,t,i=self.env.step(action)
        state=self.env.core.obs_raw[0];self.peak_yaw_rate=max(self.peak_yaw_rate,abs(float(state[15])))
        if self.env.fault_step is not None:self.min_postfault_altitude=min(self.min_postfault_altitude,float(state[2]))
        if d or t:i.update(seed=self.case_seed,level=self.case_level,mode=self.mode,terminal_position=state[:3].tolist(),terminal_velocity=state[10:13].tolist(),terminal_angular_velocity=state[13:16].tolist(),peak_yaw_rate=self.peak_yaw_rate,min_postfault_altitude=None if self.min_postfault_altitude==100. else self.min_postfault_altitude)
        if getattr(self,'compact_info',False):
            i={k:v for k,v in i.items() if k not in ['state','action','next_state']} if d or t else {}
        return obs,r,d,t,i

class CompactScheduledCases(ScheduledCases):
    def __init__(self,cases,controller,condition,mode):
        gym.Wrapper.__init__(self,TrialEnv(history=1,controller=controller,condition=condition))
        self.cases=cases;self.index=0;self.mode=mode;self.inactive=False;self.compact_info=True

class CentralHistory(VecEnvWrapper):
    """Transmit one new frame; reconstruct the identical32-frame observation."""
    def __init__(self,v):
        super().__init__(v,observation_space=gym.spaces.Box(-10.,10.,(32,48),dtype=np.float32))
        self.history=np.zeros((v.num_envs,32,48),np.float32)
    def reset(self):
        raw=self.venv.reset();self.history.fill(0.);self.history[:,-1]=raw[:,0]
        return self.history.copy()
    def step_wait(self):
        raw,reward,done,infos=self.venv.step_wait()
        for j,d in enumerate(done):
            if d:infos[j]['terminal_observation']=self.history[j].copy()
        self.history=np.concatenate([self.history[:,1:],raw],axis=1)
        self.history[done]=0.;self.history[done,-1]=raw[done,0]
        return self.history.copy(),reward,done,infos

def batched_evaluate(model=None,controller='updated',condition='uncertain',seeds=range(200000,200100),levels=None,mode='zero',workers=16,compact=True):
    global _POOL,_POOL_KEY
    from .common import LEVELS,ZERO
    override=os.environ.get('NICOL_EVAL_WORKERS')
    if override:workers=int(override)
    levels=LEVELS if levels is None else levels
    cases=[(int(seed),float(level)) for level in levels for seed in seeds]
    workers=min(workers,len(cases))
    cls=CompactScheduledCases if compact else ScheduledCases
    def factory(items):return lambda:cls(items,controller,condition,mode)
    key=(controller,condition,mode,workers,compact)
    if _POOL is None or key!=_POOL_KEY:
        close_pool();factories=[factory(cases[j::workers]) for j in range(workers)];raw=DummyVecEnv(factories) if workers==1 else SubprocVecEnv(factories,start_method='spawn');_POOL=CentralHistory(raw) if compact else raw;_POOL_KEY=key
    vec=_POOL
    for j in range(workers):vec.env_method('set_cases',cases[j::workers],indices=j)
    rows=[]
    try:
        obs=vec.reset()
        while len(rows)<len(cases):
            actions=np.tile(ZERO,(workers,1)) if model is None else model.predict(obs,deterministic=True)[0]
            obs,r,d,infos=vec.step(actions)
            for done,i in zip(d,infos):
                if done:
                    row={k:v for k,v in i.items() if k not in ['state','action','next_state','terminal_observation','TimeLimit.truncated']}
                    rows.append(row)
    except BaseException:
        close_pool();raise
    return sorted(rows,key=lambda r:(r['level'],r['seed']))