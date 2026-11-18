"""Full-flight evaluation with decision and actual-actuation summaries for every case.
The plant, strict contact monitor and state-reset behavior are unchanged from R7.
"""
import time,hashlib
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from ..round7.evaluation import Cases,key
from ..round6.env import ZERO
from ..round4.models import tree_map
from .common import save

def summary(history):
 starts=[i for i,r in enumerate(history) if bool(r.get('policy_option_started',False))];applied=[i for i,r in enumerate(history) if np.max(np.abs(r.get('projected_motor_delta',np.zeros(4))))>1e-7]
 result=dict(options_started=len(starts),actual_motor_intervention_steps=len(applied),first_actual_intervention_seconds=applied[0]/48 if applied else None,option_start_seconds=starts[0]/48 if starts else None)
 if starts:
  r=history[starts[0]];result['option_id']=int(r['policy_selected_option']);result['initial_forecasts']={k[len('policy_option_initial_'):]:float(v) for k,v in r.items() if k.startswith('policy_option_initial_')}
 return result

def evaluate(model,case_list,workers=8,path=None,capture_directory=None,capture_baseline=None,inference_device='cpu'):
 if not case_list:raise ValueError('No cases')
 start=time.time();n=min(workers,len(case_list));factories=[lambda cc=case_list[j::n]:Cases(cc,capture=capture_directory is not None) for j in range(n)];v=SubprocVecEnv(factories,start_method='spawn') if n>1 else DummyVecEnv(factories);state=None;rows=[];histories=[[] for _ in range(n)];protected={key(r):r for r in capture_baseline['episodes']} if capture_baseline is not None else None
 if model is not None:model=model.to(inference_device).double().eval();parameter=next(model.parameters())
 try:
  obs=v.reset()
  with torch.no_grad():
   while len(rows)<len(case_list):
    if model is None:a=np.tile(ZERO,(n,1))
    else:a,state=model.predict_actions(torch.as_tensor(obs[:,0],dtype=parameter.dtype,device=parameter.device),state);a=a.cpu().numpy()
    if capture_directory is not None:
     dd=getattr(model,'last_diagnostics',{}) if model is not None else {};dd={k:v.detach().cpu().numpy() for k,v in dd.items()}
     for j in range(n):histories[j].append(dict(observation=obs[j,0].copy(),requested_action=a[j].copy(),**{'policy_'+k:v[j].copy() for k,v in dd.items()}))
    obs,_,done,infos=v.step(a)
    if state is not None:
     keep=torch.as_tensor(~done,device=parameter.device);state=tree_map(lambda x:x*keep.reshape(n,*([1]*(x.ndim-1))),state)
    for j,(d,info) in enumerate(zip(done,infos)):
     if capture_directory is not None and info.get('_trace'):histories[j][-1].update(info['_trace'])
     if d:
      row={k:v for k,v in info.items() if k not in ['terminal_observation','TimeLimit.truncated','_trace']}
      if capture_directory is not None:
       row['policy_summary']=summary(histories[j]);wanted=protected is None or bool(row['stable_landing'])!=bool(protected[key(row)]['stable_landing'])
       if wanted:
        folder=Path(capture_directory);folder.mkdir(parents=True,exist_ok=True);tag=hashlib.sha256(repr(key(row)).encode()).hexdigest()[:12];tracepath=folder/(tag+'.npz');fields=set.intersection(*(set(x) for x in histories[j]));np.savez_compressed(tracepath,**{k:np.asarray([x[k] for x in histories[j]]) for k in fields});row['trace_path']=str(tracepath)
       histories[j]=[]
      rows.append(row)
  assert len(rows)==len(case_list)
 finally:v.close()
 result=dict(episodes=rows,seconds=time.time()-start,inference_device=str(parameter.device) if model is not None else 'cpu',all_episode_decision_summaries=capture_directory is not None)
 if path:save(path,result)
 return result
