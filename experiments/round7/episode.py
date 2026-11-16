"""Reusable strict flights. Final-only metadata avoids storing full traces twice."""
import copy,time
import numpy as np
import torch
from ..round6.env import LandingEnv,ZERO
from ..round6.models import Agent
from ..round6.offpolicy import OffPolicy
from .common import config,silent

def load_policy(path):
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d.get('config',{}).get('proposal_repair'):
        from .proposal_model import ProposalAgent
        cls=ProposalAgent
    else:cls=OffPolicy if 'method' in d else Agent
    model=cls.load(path,'cpu').double().eval();model.checkpoint_path=str(path);return model
def action_from_delta(delta):
    x=np.asarray(delta,float);amount=float(np.max(np.abs(x)))
    return ZERO.copy() if amount<=.002 else np.r_[x/max(amount,1e-9),2*min(amount,1.)-1,1.].astype(np.float32)
def residual(action):
    a=np.asarray(action);return a[:4]*(a[4]+1)/2 if a[5]>0 else np.zeros(4)
def infer(model,obs,state=None,proposal=False):
    if model is None:return ZERO.copy(),state
    with torch.no_grad():
        parameter=next(model.parameters());x=torch.as_tensor(obs[-1:,:],dtype=parameter.dtype,device=parameter.device)
        if proposal and isinstance(model,OffPolicy):
            memory=None if state is None else state[0];clock=0 if state is None else int(state[1].item())
            features,new=model.features(x,memory);a,_=model.proposal(features,deterministic=True,noise=model.evaluation_noise(torch.tensor([clock],device=x.device)) if model.method=='fql' else None)
            return action_from_delta(a[0].cpu().numpy()),[new,torch.tensor([clock+1.],dtype=x.dtype)]
        if proposal:
            _,a,new=model.propose_actions(x,state)
        else:a,new=model.predict_actions(x,state)
    return a[0].cpu().numpy().astype(np.float32),new

def terminal_info(info):
    return {k:v for k,v in info.items() if not isinstance(v,np.ndarray) and k not in ['terminal_observation']}

def fly(seed,level,condition='estimated',noise=1.,clock=None,model=None,collect=False,controller=None):
    with silent():e=LandingEnv(condition=condition,config=config(noise_multiplier=noise,fault_clock=clock))
    started=time.time();state=None;rows=[]
    try:
        obs,_=e.reset(seed=int(seed),options=dict(level=float(level)))
        while True:
            a,state=infer(model,obs,state)
            if controller is not None:a=controller(e,obs,a)
            before=obs[-1].copy();obs,reward,d,t,info=e.step(a)
            if collect:rows.append(dict(s=before,ns=obs[-1].copy(),a=info['executed_action'].copy(),reward=reward,done=d or t))
            if d or t:break
        result=dict(terminal_info(info),seed=int(seed),level=float(level),condition=condition,noise=float(noise),clock=clock,seconds=time.time()-started)
        return (result,rows) if collect else result
    finally:e.close()
