"""Mixed-condition vector flights with the frozen strict touchdown contract."""
import time,itertools,hashlib
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from ..round6.env import LandingEnv,ZERO
from ..round4.models import tree_map
from .common import config,silent,save
from .episode import terminal_info

class Cases(gym.Wrapper):
    def __init__(self,cases,capture=False):
        with silent():super().__init__(LandingEnv(config=config()))
        self.cases=cases;self.index=0;self.inactive=False;self.capture=capture
    def reset(self,**kwargs):
        if self.index>=len(self.cases):self.inactive=True;return np.zeros(self.observation_space.shape,np.float32),{}
        case=self.cases[self.index];self.index+=1;self.current=case
        e=self.env;e.condition=case['condition'];e.config['noise_multiplier']=case['noise'];e._sensor_sigma=.002*case['noise'];e.config['fault_clock']=case.get('clock')
        with silent():return e.reset(seed=case['seed'],options=dict(level=case['level']))
    def step(self,action):
        if self.inactive:return np.zeros(self.observation_space.shape,np.float32),0.,False,False,{}
        before_true=self.env.true_eta.copy() if self.capture else None;before_est=self.env.estimated_eta.copy() if self.capture else None
        obs,r,d,t,info=self.env.step(action)
        result=dict(terminal_info(info),**self.current) if d or t else {}
        if self.capture:result['_trace']=dict(true_eta_before=before_true,estimated_eta_before=before_est,next_measured=self.env.measured.copy(),motor_command_rpm=info['motor_command_rpm'],executed_action=info['executed_action'],projected_motor_delta=info['projected_motor_delta'])
        return obs,r,d,t,result

def cases(seeds,levels=(.295,.296),conditions=('exact','estimated'),noises=(1.,2.),clocks=(None,)):
    return [dict(seed=int(s),level=float(l),condition=c,noise=float(n),clock=k) for s,l,c,n,k in itertools.product(seeds,levels,conditions,noises,clocks)]

def evaluate(model,case_list,workers=8,path=None,capture_directory=None,capture_baseline=None,inference_device='cpu'):
    if not case_list:raise ValueError('No cases')
    start=time.time();n=min(workers,len(case_list));factories=[lambda cc=case_list[j::n]:Cases(cc,capture=capture_directory is not None) for j in range(n)]
    v=SubprocVecEnv(factories,start_method='spawn') if n>1 else DummyVecEnv(factories);state=None;rows=[];histories=[[] for _ in range(n)]
    protected={key(r):r for r in capture_baseline['episodes']} if capture_baseline is not None else None
    if model is not None:model=model.to(inference_device).double().eval();parameter=next(model.parameters())
    try:
        obs=v.reset()
        with torch.no_grad():
            while len(rows)<len(case_list):
                if model is None:a=np.tile(ZERO,(n,1))
                else:
                    a,state=model.predict_actions(torch.as_tensor(obs[:,0],dtype=parameter.dtype,device=parameter.device),state);a=a.cpu().numpy()
                if capture_directory is not None:
                    dd=getattr(model,'last_diagnostics',{}) if model is not None else {}
                    dd={k:v.detach().cpu().numpy() for k,v in dd.items()}
                    for j in range(n):histories[j].append(dict(observation=obs[j,0].copy(),requested_action=a[j].copy(),**{'policy_'+k:v[j].copy() for k,v in dd.items()}))
                obs,_,done,infos=v.step(a)
                if state is not None:
                    keep=torch.as_tensor(~done,device=parameter.device);state=tree_map(lambda x:x*keep.reshape(n,*([1]*(x.ndim-1))),state)
                for j,(d,info) in enumerate(zip(done,infos)):
                    if capture_directory is not None and info.get('_trace'):histories[j][-1].update(info['_trace'])
                    if d:
                        row={k:v for k,v in info.items() if k not in ['terminal_observation','TimeLimit.truncated','_trace']}
                        if capture_directory is not None:
                            wanted=protected is None or bool(row['stable_landing'])!=bool(protected[key(row)]['stable_landing'])
                            if wanted:
                                folder=Path(capture_directory);folder.mkdir(parents=True,exist_ok=True);tag=hashlib.sha256(repr(key(row)).encode()).hexdigest()[:12];tracepath=folder/(tag+'.npz')
                                fields=set.intersection(*(set(x) for x in histories[j]));np.savez_compressed(tracepath,**{k:np.asarray([x[k] for x in histories[j]]) for k in fields});row['trace_path']=str(tracepath)
                            histories[j]=[]
                        rows.append(row)
        assert len(rows)==len(case_list)
    finally:v.close()
    result=dict(episodes=rows,seconds=time.time()-start,inference_device=str(parameter.device) if model is not None else 'cpu')
    if path:save(path,result)
    return result

def key(r):return tuple(r[k] for k in ['seed','level','condition','noise','clock'])
def compare(result,baseline):
    b={key(r):r for r in baseline['episodes']};rows=result['episodes']
    assert len(rows)==len(b)
    gain=loss=critical=0;diff=[]
    for r in rows:
        base=b[key(r)];d=int(r['stable_landing'])-int(base['stable_landing']);gain+=d>0;loss+=d<0
        if r['condition']=='estimated' and r['level'] in [.295,.296]:diff.append(d)
    return dict(cases=len(rows),success=sum(r['stable_landing'] for r in rows)/len(rows),gains=int(gain),losses=int(loss),
        critical_difference=float(np.mean(diff)) if diff else 0.,mild_losses=sum(b[key(r)]['stable_landing'] and not r['stable_landing'] and r['level']<=.25 for r in rows),
        exact_losses=sum(b[key(r)]['stable_landing'] and not r['stable_landing'] and r['condition']=='exact' for r in rows))
