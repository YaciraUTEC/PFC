"""Sufficient complete-policy OFF replay, never a safety certificate.
Under the induction that no earlier proposal actuated, option memory is zero and
all observed states are the captured baseline. If even the permissive numerical
margin cannot start any admissible option at any such state, the entire baseline
is identical. Any uncertain/borderline/nonfinite case gets fresh simulation.
"""
import numpy as np
import torch
from ..round6.models import physics_observation
from ..round6.offpolicy import OffPolicy
from .policy import OPTIONS,option_durations
MARGIN=1e-7


def certify(model,histories):
    lengths=np.asarray([len(x) for x in histories]);n=len(lengths);p=next(model.parameters());pm=rm=None;off=np.ones(n,bool);peak=np.zeros(n,float)
    source=model.proposer
    with torch.no_grad():
        for first in range(0,int(lengths.max()),64):
            width=min(64,int(lengths.max())-first);data=np.zeros((n,width,88),np.float32);valid=np.arange(first,first+width)[None]<lengths[:,None]
            for j,h in enumerate(histories):
                end=min(len(h),first+width)
                if end>first:data[j,:end-first]=h[first:end]
            obs=torch.as_tensor(data,device=p.device,dtype=p.dtype);clean=physics_observation(obs);flat=obs.reshape(-1,88)
            if isinstance(source,OffPolicy):
                z,pm=source.prior.encoder(clean,pm);f=torch.cat([clean,z],-1).reshape(n*width,-1)
                clock=torch.arange(first,first+width,device=p.device).repeat(n);noise=source.evaluation_noise(clock) if source.method=='fql' else None
                raw,_=source.proposal(f,deterministic=True,noise=noise)
                if source.gate:
                    difference=(source.benefit_q(f,raw)-source.benefit_q(f,torch.zeros_like(raw))).amin(0);native=difference>=source.threshold-MARGIN
                else:native=torch.ones(len(flat),device=p.device,dtype=torch.bool)
            else:
                mu,_,pm=source(obs,pm);mu=mu.reshape(n*width,-1)
                if hasattr(source,'physical_delta'):raw=source.physical_delta(mu)
                else:
                    a=mu[:,:5].clamp(-5,5).tanh();raw=a[:,:4]*(a[:,4:5]+1)/2
                native=mu[:,5]>=-MARGIN
            bad=~torch.isfinite(raw).all(-1)
            if model.mode=='off':possible=torch.zeros_like(native)
            elif model.mode in ['native','forced']:
                delta=raw if model.mode=='forced' else torch.where(native[:,None],raw,torch.zeros_like(raw))
                possible=delta.abs().amax(-1)>=.002-MARGIN
                if model.screen is not None:
                    duration=torch.ones(len(flat),device=p.device,dtype=p.dtype);candidate=model.screen.penalty(flat,delta,duration);base=model.screen.penalty(flat,torch.zeros_like(delta),duration)
                    sure_veto=torch.isfinite(candidate)&torch.isfinite(base)&(candidate>base+model.screen.allowance+MARGIN)
                    possible &=~sure_veto
            elif model.mode=='counterfactual':
                rz,rm=model.risk.encoder(clean,rm);features=torch.cat([clean,rz],-1).reshape(n*width,-1)
                delta=torch.where(native[:,None],raw,torch.zeros_like(raw));actions=torch.stack([delta*a for t,a in OPTIONS],1);budgets=option_durations(flat)
                x=torch.cat([features[:,None].expand(-1,len(OPTIONS),-1),actions,model.controller_id[None,None].expand(len(flat),len(OPTIONS),-1),budgets[:,:,None]/96],-1)
                probabilities=model.gate.probabilities(x);adv=probabilities[...,1]-probabilities[...,2]
                gain=adv.mean(0)-model.uncertainty_penalty*adv.std(0,unbiased=False);harm=probabilities[...,2].mean(0)+model.uncertainty_penalty*probabilities[...,2].std(0,unbiased=False)
                allowed=(gain>=model.minimum_gain-MARGIN)&(harm<=model.maximum_harm+MARGIN)&(actions.abs().amax(-1)>=.002-MARGIN)
                bad |=~torch.isfinite(probabilities).all((0,2,3))
                if model.screen is not None:
                    repeated=flat[:,None].expand(-1,len(OPTIONS),-1).reshape(-1,88);aa=actions.reshape(-1,4);tt=budgets.reshape(-1)
                    candidate=model.screen.penalty(repeated,aa,tt);base=model.screen.penalty(repeated,torch.zeros_like(aa),tt)
                    veto=(torch.isfinite(candidate)&torch.isfinite(base)&(candidate>base+model.screen.allowance+MARGIN)).reshape(-1,len(OPTIONS))
                    allowed &=~veto
                possible=allowed.any(-1)
            else:raise ValueError(model.mode)
            possible=(possible|bad)&(flat[:,57]<.5+MARGIN);possible=possible.reshape(n,width).cpu().numpy()&valid
            off &=~possible.any(1);peak=np.maximum(peak,possible.any(1).astype(float))
    return off,peak
