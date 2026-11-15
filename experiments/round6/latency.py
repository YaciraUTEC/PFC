"""Dedicated sequential PC timing on recorded validation inputs, after training.
Measures control/observer/governor and neural inference separately. This is a
runtime benchmark, not a formal worst-case execution-time or flight guarantee.
"""
import json,time,contextlib,os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .evaluate import OUT,save,SEEDS
from .env import LandingEnv
from .models import Agent,ARCHS
from .offpolicy import OffPolicy

def inputs(frame):
    f=np.asarray(frame,float);R=f[6:15].reshape(3,3)
    position=np.r_[-4*f[18:20],1.2*f[40]];rotation=Rotation.from_matrix(R)
    state=np.zeros(20);state[:3]=position;state[3:7]=rotation.as_quat();state[7:10]=rotation.as_euler('xyz')
    state[10:13]=f[3:6]*2;state[13:16]=R@(f[15:18]*10)
    kwargs=dict(control_timestep=1/48,cur_pos=position,cur_quat=state[3:7],cur_vel=state[10:13],cur_ang_vel=state[13:16],target_pos=position+f[:3],target_vel=f[60:63])
    return state,kwargs,np.clip(1-.25*f[31:35],0,1)
def stats(values):
    x=np.asarray(values)*1000
    return dict(samples=len(x),median_ms=float(np.median(x)),p95_ms=float(np.quantile(x,.95)),p99_ms=float(np.quantile(x,.99)),max_ms=float(x.max()))
def traces():
    folder=OUT/'cache/validation/selected/estimated_noise2'
    result=[]
    for level in [0.,.25,.295,.296]:
        p=folder/f'{level:.6f}_611008.npz';raw=np.load(p)['observations'];raw=raw[raw[:,57]<.5]
        fault=np.flatnonzero(np.min(1-.25*raw[:,31:35],axis=1)<.98)
        start=max(0,int(fault[0])-16) if len(fault) else max(0,len(raw)//2-48)
        index=np.r_[np.arange(start,min(start+64,len(raw))),np.arange(max(0,len(raw)-64),len(raw))]
        result.append((level,raw[index]))
    return result

def main():
    torch.set_num_threads(1);start=time.time();c1=json.loads((OUT/'phase1_conclusion.json').read_text());data=traces()
    names={'selected':c1['config']}
    fast=max([n for n in c1['development'] if n.startswith('nmpc_fast')],key=lambda n:c1['development'][n]['score'])
    for name in [fast,c1['best_slow_nmpc']]:names[name]=c1['all_configs'][name]
    controls={}
    for name,config in names.items():
        timings=[]
        with open(os.devnull,'w') as f,contextlib.redirect_stdout(f):env=LandingEnv(condition='estimated',config=dict(config,gate_mode='off'))
        try:
            for level,raw in data:
                with open(os.devnull,'w') as f,contextlib.redirect_stdout(f):env.reset(seed=610030,options=dict(level=level))
                previous=None;command=np.full(4,env.c.HOVER_RPM)
                for index,frame in enumerate(raw):
                    state,args,eta=inputs(frame);env.estimated_eta=eta
                    env.c.integral_pos_e=frame[25:28]*2;env.c.integral_rpy_e=frame[28:31]*.5
                    t=time.perf_counter()
                    if frame[56]>.5:env.descent.reference(state,np.array([0.,0.,.1]),1/48)
                    env.c.set_motor_effectiveness(eta);w=env.c.prepare_control(**args);rpm,_=env.c.allocate_wrench(w,commit=False)
                    actual,_,_=env.c.apply_prepared_q((rpm/env.c.HOVER_RPM)**2)
                    if previous is not None:env.estimator.update(previous,state,command,1/48,contact=False)
                    elapsed=time.perf_counter()-t;previous=state;command=actual
                    if index>=16:timings.append(elapsed)
            controls[name]=stats(timings);print('DEDICATED_CONTROL_LATENCY',name,controls[name],flush=True)
        finally:env.close()
    p2=json.loads((OUT/'phase2_conclusion.json').read_text());neural={};raw=np.concatenate([r for l,r in data])
    for arch in ARCHS:
        schedule=p2['schedule_by_arch'][arch]
        for method in ['ppo','flashsac','fql']:
            path=OUT/'phase2'/arch/'seed_201'/schedule/'best.pt' if method=='ppo' else OUT/'phase3'/method/arch/'seed_201'/'best.pt'
            m=(Agent.load(path,'cpu') if method=='ppo' else OffPolicy.load(path,'cpu')).double().eval();state=None;timings=[]
            with torch.no_grad():
                for index,frame in enumerate(raw):
                    t=time.perf_counter();action,state=m.predict_actions(torch.as_tensor(frame[None],dtype=torch.float64),state);action.cpu().numpy().astype(np.float32)
                    if index>=16:timings.append(time.perf_counter()-t)
            neural[f'{arch}_{method}']=stats(timings);del m
    result=dict(controllers=controls,neural=neural,seconds=time.time()-start,control_period_ms=1000/48,
        scope='Sequential,1CPUthread,no experimental training workers;validation observations only. Includes governor/controller/observer;neural timing separate. Not a hard real-time proof.',
        selection_note='Phase1 screening used loaded timing. This dedicated measurement is reported separately;no learning hyperparameters or test data are used.')
    save(OUT/'latency.json',result)
if __name__=='__main__':main()
