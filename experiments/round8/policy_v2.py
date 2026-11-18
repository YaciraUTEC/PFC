"""Choose among explicitly labeled native and unvetoed feedback options once."""
import torch
from .policy import PreservePolicy
from .factor_gate import FactorGate,measures
from .options import OPTIONS,projected_delta
from ..round7.episode import load_policy
from ..round7.policy import propose,risk_features,expand_delta

class PreserveOptions(PreservePolicy):
 def __init__(self,proposer,gate,option_mode='both',risk_aggregation='weighted_std',**kw):
  super().__init__(proposer,gate,**kw);assert gate.dim==201
  self.risk_aggregation=risk_aggregation;assert risk_aggregation in ['weighted_std','max_member'];self.option_mode=option_mode;self.option_ids=[i for i,(_,_,native) in enumerate(OPTIONS) if option_mode=='both' or native==(option_mode=='native')]
  assert option_mode in ['both','native','raw'];self.options=[OPTIONS[i] for i in self.option_ids]
 def build_options(self,obs,context,raw,on):
  aa=torch.stack([torch.where(on[:,None],raw,torch.zeros_like(raw))*a if native else raw*a for t,a,native in self.options],1)
  available=(obs[:,72]*1440).round().clamp(1,1440);budgets=torch.stack([torch.minimum(torch.full_like(available,float(t)),available) for t,a,native in self.options],1)
  n=len(self.options);flags=torch.as_tensor([native for t,a,native in self.options],dtype=obs.dtype,device=obs.device)[None,:,None].expand(len(obs),-1,-1)
  projected=projected_delta(obs[:,None].expand(-1,n,-1),aa)
  x=torch.cat([context[:,None].expand(-1,n,-1),aa,self.controller_id[None,None].expand(len(obs),n,-1),budgets[:,:,None]/96,flags,projected],-1)
  return aa,budgets,x
 def assess(self,x,delta,margin=0.):
  m=measures(self.gate,x,self.uncertainty_penalty);risk=m['conditional_harm_max_member'] if self.risk_aggregation=='max_member' else m['conditional_harm'];m['decision_conditional_harm']=risk;allowed=(m['lower_gain']>self.minimum_gain-margin)&(risk<self.maximum_harm+margin)&(delta.abs().amax(-1)>.002-margin)
  if self.allowed_options is not None:
   use=torch.as_tensor([i in self.allowed_options for i in self.option_ids],dtype=torch.bool,device=x.device);allowed &=use
  return allowed,m
 def predict_actions(self,obs,state=None):
  if state is None:
   pm=rm=None;remaining=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device);used=remaining.clone();choice=remaining.clone();forecast=torch.zeros(len(obs),6,dtype=obs.dtype,device=obs.device)
  else:pm,rm,remaining,used,choice,forecast=state
  raw,on,pm=propose(self.proposer,obs,pm);context,rm=risk_features(self.risk,obs,rm);aa,budgets,x=self.build_options(obs,context,raw,on);allowed,m=self.assess(x,aa)
  scores=torch.where(allowed,m['lower_gain'],torch.full_like(m['lower_gain'],-1e9));best=scores.argmax(-1);idx=torch.arange(len(obs),device=obs.device);start=allowed.any(-1)&(used<1)&(obs[:,57]<.5)
  remaining=torch.where(start,budgets[idx,best],remaining);choice=torch.where(start,best.to(obs.dtype),choice);used=used+start
  initial=torch.stack([m[k][idx,best] for k in ['base_success','preservation','mean_rescue','mean_harm','decision_conditional_harm','lower_gain']],-1);forecast=torch.where(start[:,None],initial,forecast)
  plan=(remaining>0)&(obs[:,57]<.5);selected=aa[idx,choice.long()];selected=torch.where(plan[:,None],selected,torch.zeros_like(selected));remaining=torch.where(plan,(remaining-1).clamp_min(0),torch.zeros_like(remaining))
  ids=torch.as_tensor(self.option_ids,device=obs.device)[choice.long()]
  self.last_diagnostics=dict(proposal_delta=raw.detach(),native_on=on.detach(),authorized_delta=selected.detach(),final_delta=selected.detach(),remaining=remaining.detach(),options_used=used.detach(),option_started=start.detach(),selected_option=ids.detach(),option_initial_base_success=forecast[:,0].detach(),option_initial_preservation=forecast[:,1].detach(),option_initial_joint_rescue=forecast[:,2].detach(),option_initial_joint_harm=forecast[:,3].detach(),option_initial_conditional_harm=forecast[:,4].detach(),option_initial_lower_gain=forecast[:,5].detach())
  return expand_delta(selected),[pm,rm,remaining,used,choice,forecast]
 @classmethod
 def from_spec(cls,spec):
  model=cls(load_policy(spec['proposer']),FactorGate.load(spec['gate']),option_mode=spec.get('option_mode','both'),risk_aggregation=spec.get('risk_aggregation','weighted_std'),minimum_gain=spec.get('minimum_gain',.005),maximum_harm=spec.get('maximum_harm',.005),penalty=spec.get('penalty',1.),allowed_options=spec.get('allowed_options'))
  return model.double().eval()
