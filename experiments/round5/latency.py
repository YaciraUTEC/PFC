"""Measure full controller compute,excluding simulator physics,on thisPC."""
import json,time,contextlib,os
import numpy as np
import torch
from .env import LandingEnv,ZERO
from .inventory import groups
from .confirm import specifications
from .models import load_agent
from .policies import ProposalPolicy
from .evaluate import OUT
from ..run import save

def main():
    torch.set_num_threads(1);jobs=specifications()
    for group in groups():
        entry=group['entries'][0]
        from .inventory import core
        jobs.append(dict(id='family_'+group['id'],path=entry['path'],config=dict(core(),intervention=group['intervention']),threshold=0.,shadow=False))
    results=[]
    for spec in jobs:
        target=OUT/'latency'/f"{spec['id']}.json"
        if target.exists():results.append(json.loads(target.read_text()));continue
        m=None if spec['path'] is None else load_agent(spec['path'])
        if m is not None:
            m.gate_logit_threshold=spec['threshold']
            if spec['shadow']:m=ProposalPolicy(m)
        with open(os.devnull,'w') as silent,contextlib.redirect_stdout(silent):env=LandingEnv(config=spec['config'])
        try:
            obs,_=env.reset(seed=528025,options=dict(level=.295));state=None;simulator=[0.];original=env._impact_step
            def physics(command):
                t=time.perf_counter();value=original(command);simulator[0]=time.perf_counter()-t;return value
            env._impact_step=physics;measured=[];net=[];resets=0
            for step in range(288):
                before=time.perf_counter()
                if m is None:action=ZERO
                else:
                    with torch.no_grad():a,state=m.predict_actions(torch.as_tensor(obs[-1:,:],dtype=torch.float64),state)
                    action=a[0].numpy()
                network=time.perf_counter()-before;obs,_,d,tr,info=env.step(action);elapsed=time.perf_counter()-before-simulator[0]
                if step>=32:measured.append(elapsed);net.append(network)
                if d or tr:resets+=1;obs,_=env.reset(seed=528025+resets%7,options=dict(level=.295));state=None
            row=dict(id=spec['id'],samples=len(measured),mean_seconds=float(np.mean(measured)),p95_seconds=float(np.quantile(measured,.95)),max_seconds=float(max(measured)),
                network_mean_seconds=float(np.mean(net)),network_p95_seconds=float(np.quantile(net,.95)),control_period_seconds=1/48,
                realtime_pass=bool(np.quantile(measured,.95)<=1/48),scope='CPUfloat64,batch1,full normalisation/observer/PID/gate/motor allocation and bookkeeping;physics time excluded,warm32steps,256measuredsteps;portable kernels.')
            save(target,row);results.append(row);print('LATENCY',spec['id'],'p95ms',round(1000*row['p95_seconds'],3),flush=True)
        finally:env.close()
    save(OUT/'latency_complete.json',dict(results=results,measured=True))
if __name__=='__main__':main()
