"""One latched feedback option followed by DSLPID for the rest of the flight.
Unlike repeated local authorization, the executed continuation matches the label.
A native veto may be part of an option, but is never secretly added or removed.
"""
import torch
from torch import nn
from ..round7.policy import canonical_risk,risk_features,propose,expand_delta,controller_descriptor,option_durations,OPTIONS,OutcomeGate
from ..round7.episode import load_policy
from .factor_gate import FactorGate,measures
class PreservePolicy(nn.Module):
 def __init__(self,proposer,gate,minimum_gain=.005,maximum_harm=.005,penalty=1.,native=True,risk_mode='conditional',allowed_options=None,max_options=1):
  super().__init__();self.proposer=proposer;self.risk=canonical_risk();self.gate=gate
  self.register_buffer('controller_id',torch.as_tensor(controller_descriptor(proposer.checkpoint_path),dtype=torch.float64))
  self.minimum_gain=float(minimum_gain);self.maximum_harm=float(maximum_harm);self.uncertainty_penalty=float(penalty);self.native=bool(native);self.risk_mode=risk_mode;self.allowed_options=allowed_options;self.max_options=int(max_options);self.mode='preserve_rescue'
  for p in self.parameters():p.requires_grad_(False)
 def option_features(self,obs,features,delta):
  aa=torch.stack([delta*alpha for ticks,alpha in OPTIONS],1);budgets=option_durations(obs)
  x=torch.cat([features[:,None].expand(-1,len(OPTIONS),-1),aa,self.controller_id[None,None].expand(len(obs),len(OPTIONS),-1),budgets[:,:,None]/96],-1)
  return aa,budgets,x
 def assess(self,x,delta,margin=0.):
  m=measures(self.gate,x,self.uncertainty_penalty)
  risk=m['conditional_harm'] if self.risk_mode=='conditional' else m['joint_harm']
  allowed=(m['lower_gain']>self.minimum_gain-margin)&(risk<self.maximum_harm+margin)&(delta.abs().amax(-1)>.002-margin)
  if self.allowed_options is not None:
   use=torch.zeros(len(OPTIONS),dtype=torch.bool,device=x.device);use[self.allowed_options]=True;allowed &=use
  return allowed,m
 def predict_actions(self,obs,state=None):
  if state is None:
   pm=rm=None;scale=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device);remaining=scale.clone();used=scale.clone();chosen=scale.clone();forecast=torch.zeros(len(obs),6,dtype=obs.dtype,device=obs.device)
  else:pm,rm,scale,remaining,used,chosen,forecast=state
  raw,native_on,pm=propose(self.proposer,obs,pm);delta=torch.where(native_on[:,None],raw,torch.zeros_like(raw)) if self.native else raw
  features,rm=risk_features(self.risk,obs,rm);aa,budgets,x=self.option_features(obs,features,delta);allowed,m=self.assess(x,aa)
  available=(remaining<=0)&(used<self.max_options)&(obs[:,57]<.5)
  scores=torch.where(allowed,m['lower_gain'],torch.full_like(m['lower_gain'],-1e9));index=scores.argmax(-1);start=allowed.any(-1)&available
  idx=torch.arange(len(obs),device=obs.device);alphas=torch.as_tensor([a for t,a in OPTIONS],device=obs.device,dtype=obs.dtype)
  scale=torch.where(start,alphas[index],scale);remaining=torch.where(start,budgets[idx,index],remaining);used=used+start
  chosen=torch.where(start,index.to(obs.dtype),chosen);values=torch.stack([m[k][idx,index] for k in ['base_success','preservation','mean_rescue','mean_harm','conditional_harm','lower_gain']],-1);forecast=torch.where(start[:,None],values,forecast)
  plan=(remaining>0)&(obs[:,57]<.5);selected=torch.where(plan[:,None],delta*scale[:,None],torch.zeros_like(delta));remaining=torch.where(plan,(remaining-1).clamp_min(0),torch.zeros_like(remaining))
  self.last_diagnostics=dict(proposal_delta=raw.detach(),native_on=native_on.detach(),authorized_delta=selected.detach(),final_delta=selected.detach(),remaining=remaining.detach(),options_used=used.detach(),option_started=start.detach(),option_initial_base_success=forecast[:,0].detach(),option_initial_preservation=forecast[:,1].detach(),option_initial_joint_rescue=forecast[:,2].detach(),option_initial_joint_harm=forecast[:,3].detach(),option_initial_conditional_harm=forecast[:,4].detach(),option_initial_lower_gain=forecast[:,5].detach(),selected_option=chosen.detach())
  return expand_delta(selected),[pm,rm,scale,remaining,used,chosen,forecast]
 @classmethod
 def from_spec(cls,spec):
  if spec.get('version')==2:
   from .policy_v2 import PreserveOptions
   return PreserveOptions.from_spec(spec)
  gate=(FactorGate if spec.get('gate_kind','factor')=='factor' else OutcomeGate).load(spec['gate'])
  model=cls(load_policy(spec['proposer']),gate,minimum_gain=spec.get('minimum_gain',.005),maximum_harm=spec.get('maximum_harm',.005),penalty=spec.get('penalty',1.),native=spec.get('native',True),risk_mode=spec.get('risk_mode','conditional'),allowed_options=spec.get('allowed_options'),max_options=spec.get('max_options',1))
  return model.double().eval()
