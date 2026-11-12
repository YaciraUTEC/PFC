"""Oracle diagnosis of actual frozen proposals on reserved validation routes.
Only diagnostic: privileged counterfactual rollout outcomes never authorize the
controller or select its weights/thresholds. Five fixed query phases per flight.
"""
import json,time,contextlib,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
import numpy as np
import torch
import pybullet as p
from .inventory import OUT,groups,core,sha
from .models import load_agent
from .env import LandingEnv
from .counterfactual import snapshot,restore,rollout_cost
from ..run import save


def compare_proposal(env,proposal,mode,horizon=None):
    remaining=1440-env.t if horizon is None else horizon;saved=snapshot(env)
    try:
        restore(env,saved);base,base_outcome=rollout_cost(env,np.array([0.,0.,0.,0.,-1.,-1.],np.float32),mode,horizon=remaining)
        restore(env,saved);candidate,outcome=rollout_cost(env,proposal,mode,horizon=remaining)
        gain=base-candidate;margin=max(.015,.03*abs(base))
        return dict(gain=gain,margin=margin,beneficial=bool(gain>margin),base_cost=base,candidate_cost=candidate,base_outcome=base_outcome,outcome=outcome,
            binary_rescue=bool(base_outcome!='stable_landing' and outcome=='stable_landing'),binary_loss=bool(base_outcome=='stable_landing' and outcome!='stable_landing'),burst_ticks=12,horizon_ticks=remaining)
    finally:restore(env,saved);p.removeState(saved[0],physicsClientId=env.core._env.CLIENT)


def job(group,entry):
    torch.set_num_threads(1);target=OUT/'oracle_proposal_audit'/f"{group['id']}_{entry['seed']}.json"
    if target.exists():return json.loads(target.read_text())
    start=time.time();before=sha(entry['path']);model=load_agent(entry['path']);cfg=dict(core(),intervention=group['intervention']);records=[];episodes=[]
    with open(os.devnull,'w') as silent,contextlib.redirect_stdout(silent):env=LandingEnv(condition='estimated',config=cfg)
    try:
        for seed in range(528016,528020):
            obs,_=env.reset(seed=seed,options=dict(level=.295));state=None;seen=set()
            while True:
                with torch.no_grad():action,proposal,state=model.propose_actions(torch.as_tensor(obs[-1:,:],dtype=torch.float64),state)
                executed=action[0].numpy().astype(np.float32);candidate=proposal[0].numpy().astype(np.float32);phases=[]
                if env.t==0:phases.append('startup')
                if env.t==48:phases.append('one_second')
                if env.fault_step is not None and env.t-env.fault_step>=2 and 'postfault' not in seen:phases.append('postfault')
                if env.route_reached and 'descent_start' not in seen:phases.append('descent_start')
                if env.route_reached and env.measured[2]<.20 and 'near_contact' not in seen:phases.append('near_contact')
                if phases:
                    seen.update(phases);result=compare_proposal(env,candidate,group['intervention'])
                    records.append(dict(seed=seed,training_seed=entry['seed'],group=group['id'],t=env.t,phases=phases,requested_on=bool(executed[5]>0),proposal=candidate.tolist(),**result))
                obs,_,done,truncated,info=env.step(executed)
                if done or truncated:
                    episodes.append(dict(seed=seed,outcome=info['outcome'],steps=info['episode_steps'],query_count=len(seen)));break
    finally:env.close()
    assert sha(entry['path'])==before
    result=dict(group=group['id'],seed=entry['seed'],checkpoint_sha256=before,seconds=time.time()-start,records=records,episodes=episodes)
    save(target,result);return result


def main():
    target=OUT/'oracle_proposal_audit_complete.json'
    if target.exists():return
    start=time.time();jobs=[(g,e) for g in groups() for e in g['entries']];results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,g,e) for g,e in jobs]
        for future in as_completed(futures):
            results.append(future.result());save(OUT/'oracle_proposal_progress.json',dict(completed=len(results),total=len(jobs),seconds=time.time()-start))
            print('ORACLE_PROPOSALS',len(results),'/',len(jobs),flush=True)
    summary=[]
    for group in groups():
        rows=[r for result in results if result['group']==group['id'] for r in result['records']];c=Counter()
        for r in rows:
            on=r['requested_on'];benefit=r['beneficial'];c['true_positive' if on and benefit else 'false_positive' if on else 'false_negative' if benefit else 'true_negative']+=1
        summary.append(dict(group=group['id'],queries=len(rows),routes=4,training_seeds=5,beneficial=sum(r['beneficial'] for r in rows),binary_rescue_proposals=sum(r['binary_rescue'] for r in rows),binary_loss_proposals=sum(r['binary_loss'] for r in rows),
            **c,precision_when_on=None if c['true_positive']+c['false_positive']==0 else c['true_positive']/(c['true_positive']+c['false_positive']),
            recall_benefits=None if c['true_positive']+c['false_negative']==0 else c['true_positive']/(c['true_positive']+c['false_negative'])))
    save(target,dict(seconds=time.time()-start,models=len(results),episodes=sum(len(r['episodes']) for r in results),groups=summary,
        scope='Diagnostic only on4reserved validation routesx5training seedsx15families,29.5percent estimated.Raw learned gate vs own actual forcedON12tick proposal,then base to terminal.Multiple correlated queries per route;not a global optimal-control bound,not final-test or deployment evidence.',query_phases=['startup','one_second','postfault','descent_start','near_contact']))
if __name__=='__main__':main()
