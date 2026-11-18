"""Closed feedback options with explicitly defined native-veto semantics."""
import torch
import numpy as np
from ..round6.offpolicy import OffPolicy
from ..round7.policy import propose
from ..round7.episode import action_from_delta,terminal_info
from ..round6.env import ZERO
from ..round4.models import tree_map
OPTIONS=[(1,1.,True),(24,.5,True),(96,1.,True),(1440,1.,True),(1,1.,False),(24,.5,False),(96,1.,False),(1440,1.,False)]

def raw_proposal(model,obs,state=None):
 if not isinstance(model,OffPolicy):
  raw,native,new=propose(model,obs,state);return raw,new
 memory=None if state is None else state[0];clock=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device) if state is None else state[1]
 f,new=model.features(obs,memory);raw,_=model.proposal(f,deterministic=True,noise=None);return raw,[new,clock+1]

def projected_delta(obs,delta):
 base=obs[...,64:68];return (((base+.2*delta).clamp(0,1)-base)/.2)*((delta.abs().amax(-1)>.002)&(obs[...,57]<.5))[...,None]

def option_feature(obs,context,raw,identity,duration,alpha,native,veto_on):
 delta=raw*alpha if not native or veto_on else np.zeros(4)
 base=obs[64:68];applied=(np.clip(base+.2*delta,0,1)-base)/.2 if np.max(np.abs(delta))>.002 else np.zeros(4)
 return np.r_[context,delta,identity,duration/96,float(native),applied].astype(np.float32)

def rollout(e,model,memory,raw,initial_on,alpha,duration,native):
 state=tree_map(lambda t:t.detach().clone(),memory);delta=np.asarray(raw,float)*alpha if not native or initial_on else np.zeros(4);first=e.t;projection_error=0.
 for k in range(1440-e.t):
  action=action_from_delta(delta) if k<duration else ZERO
  before=np.asarray(e.frames[-1]);expected=(np.clip(before[64:68]+.2*delta,0,1)-before[64:68])/.2 if k<duration and np.max(np.abs(delta))>.002 and before[57]<.5 else np.zeros(4)
  obs,_,done,truncated,info=e.step(action)
  if k==0:projection_error=float(np.max(np.abs(np.asarray(info['projected_motor_delta'])-expected)));assert projection_error<3e-6,('Actuator projection mismatch',projection_error)
  if done or truncated:return dict(terminal_info(info),continuation_steps=e.t-first,projection_feature_error=projection_error)
  if k+1<duration:
   with torch.no_grad():
    if native:next_raw,on,state=propose(model,torch.as_tensor(obs[-1:],dtype=torch.float64),state);delta=next_raw[0].numpy()*alpha if bool(on[0]) else np.zeros(4)
    else:next_raw,state=raw_proposal(model,torch.as_tensor(obs[-1:],dtype=torch.float64),state);delta=next_raw[0].numpy()*alpha
 raise AssertionError('Option must reach the strict mission terminal')
