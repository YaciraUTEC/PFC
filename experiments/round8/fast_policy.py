"""Exact-task execution optimization for fixed201 single-option controllers.
The two encoders have identical weights/history. Once an option ended, no neural
state can affect actions until episode reset. Frozen slots are reset to zero just
like evaluated slots. Proposal diagnostics are marked when no network is run.
"""
import torch
from .policy_v2 import PreserveOptions
from .policy_v3 import EntryPreserveOptions
from .active_data import shared_proposal_context
from ..round7.policy import expand_delta

class FastMixin:
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);assert self.proposer.method=='flashsac';a=self.proposer.prior.encoder.state_dict();b=self.risk.encoder.state_dict();assert a.keys()==b.keys() and all(torch.equal(a[k].double(),b[k].double()) for k in a),'Shared encoder optimization requires identical proposal/risk weights'
 def diagnostics(self,raw,on,selected,remaining,used,choice,start,forecast,evaluated):
  ids=torch.as_tensor(self.option_ids,device=raw.device)[choice.long()]
  self.last_diagnostics=dict(proposal_delta=raw.detach(),native_on=on.detach(),authorized_delta=selected.detach(),final_delta=selected.detach(),remaining=remaining.detach(),options_used=used.detach(),option_started=start.detach(),selected_option=ids.detach(),option_initial_base_success=forecast[:,0].detach(),option_initial_preservation=forecast[:,1].detach(),option_initial_joint_rescue=forecast[:,2].detach(),option_initial_joint_harm=forecast[:,3].detach(),option_initial_conditional_harm=forecast[:,4].detach(),option_initial_lower_gain=forecast[:,5].detach(),neural_evaluated=torch.full_like(on,evaluated))
 def predict_actions(self,obs,state=None):
  if state is None:
   pm=rm=None;remaining=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device);used=remaining.clone();choice=remaining.clone();forecast=torch.zeros(len(obs),6,dtype=obs.dtype,device=obs.device)
  else:pm,rm,remaining,used,choice,forecast=state
  if bool(((used>=1)&(remaining<=0)).all()):
   zero=torch.zeros(len(obs),4,dtype=obs.dtype,device=obs.device);off=torch.zeros(len(obs),dtype=torch.bool,device=obs.device);self.diagnostics(zero,off,zero,remaining,used,choice,off,forecast,False);return expand_delta(zero),[pm,rm,remaining,used,choice,forecast]
  raw,on,pm,context=shared_proposal_context(self.proposer,obs,pm);rm=pm[0];aa,budgets,x=self.build_options(obs,context,raw,on);idx=torch.arange(len(obs),device=obs.device);needs=(used<1)&(obs[:,57]<.5)
  if bool(needs.any()):
   allowed,m=self.assess(x,aa);scores=torch.where(allowed,m['lower_gain'],torch.full_like(m['lower_gain'],-1e9));best=scores.argmax(-1);start=allowed.any(-1)&needs;remaining=torch.where(start,budgets[idx,best],remaining);choice=torch.where(start,best.to(obs.dtype),choice);used=used+start;initial=torch.stack([m[k][idx,best] for k in ['base_success','preservation','mean_rescue','mean_harm','decision_conditional_harm','lower_gain']],-1);forecast=torch.where(start[:,None],initial,forecast)
  else:start=torch.zeros(len(obs),dtype=torch.bool,device=obs.device)
  plan=(remaining>0)&(obs[:,57]<.5);selected=torch.where(plan[:,None],aa[idx,choice.long()],torch.zeros_like(raw));remaining=torch.where(plan,(remaining-1).clamp_min(0),torch.zeros_like(remaining));self.diagnostics(raw,on,selected,remaining,used,choice,start,forecast,True);return expand_delta(selected),[pm,rm,remaining,used,choice,forecast]
class FastOptions(FastMixin,PreserveOptions):pass
class FastEntryOptions(FastMixin,EntryPreserveOptions):pass
class FastFactory:
 @classmethod
 def from_spec(cls,spec):return (FastEntryOptions if spec.get('version')==3 else FastOptions).from_spec(spec)
