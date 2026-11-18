"""Counterfactual decision support requires observed effectiveness degradation.
This is an onboard observable support restriction, not a true-fault oracle.
"""
import torch
from .policy_v3 import EntryPreserveOptions
from ..round7.policy import expand_delta
from .fast_policy import FastMixin
class ObservedLossOptions(EntryPreserveOptions):
 observed_loss_min=.02
 sigma_multiplier=0.
 def assess(self,x,delta,margin=0.):
  allowed,m=super().assess(x,delta,margin);loss=x[...,31:35]*.25;uncertainty=x[...,48:52].clamp_min(0)*.1;observable=(loss-self.sigma_multiplier*uncertainty).amax(-1)>self.observed_loss_min-margin
  return allowed&observable,m
 @classmethod
 def from_spec(cls,spec):
  model=super().from_spec(spec);model.observed_loss_min=float(spec.get('observed_loss_min',.02));model.sigma_multiplier=float(spec.get('sigma_multiplier',0.));return model
class FastObservedLossOptions(FastMixin,ObservedLossOptions):
 def predict_actions(self,obs,state=None):
  if state is None:
   pm=rm=None;remaining=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device);used=remaining.clone();choice=remaining.clone();forecast=torch.zeros(len(obs),6,dtype=obs.dtype,device=obs.device)
  else:pm,rm,remaining,used,choice,forecast=state
  loss=obs[:,31:35]*.25;uncertainty=obs[:,48:52].clamp_min(0)*.1;support=(loss-self.sigma_multiplier*uncertainty).amax(-1)>self.observed_loss_min;candidate=(used<1)&(obs[:,57]<.5)&support
  spent=bool(((used>=1)&(remaining<=0)).all())
  if not spent and not bool(candidate.any()) and not bool((remaining>0).any()):
   # Keep the complete GTrXL history while residual decisions are inadmissible.
   memory=None if pm is None else pm[0];clock=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device) if pm is None else pm[1];_,new=self.proposer.features(obs,memory);pm=[new,clock+1];rm=new;zero=torch.zeros(len(obs),4,dtype=obs.dtype,device=obs.device);off=torch.zeros(len(obs),dtype=torch.bool,device=obs.device);self.diagnostics(zero,off,zero,remaining,used,choice,off,forecast,True);self.last_diagnostics['proposal_evaluated']=off;return expand_delta(zero),[pm,rm,remaining,used,choice,forecast]
  action,new=super().predict_actions(obs,state);self.last_diagnostics['proposal_evaluated']=self.last_diagnostics['neural_evaluated'];return action,new

class GuardedFactory:
 @classmethod
 def from_spec(cls,spec):return FastObservedLossOptions.from_spec(spec)
