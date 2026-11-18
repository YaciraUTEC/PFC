"""Explicit baseline, conditional preservation and conditional rescue.
The baseline output is structurally independent of every option descriptor.
Ensemble spread is a heuristic, not a calibrated statistical safety bound.
"""
import torch
from torch import nn

def network(dim,out):
 return nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.LayerNorm(128),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,out))
class FactorGate(nn.Module):
 def __init__(self,dim=196,context_dim=184,members=3):
  super().__init__();self.dim=dim;self.context_dim=context_dim
  self.register_buffer('center',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim));self.register_buffer('temperature',torch.ones(3))
  self.base_members=nn.ModuleList([network(context_dim,1) for _ in range(members)])
  self.option_members=nn.ModuleList([network(dim,2) for _ in range(members)])
 def forward(self,x):
  z=((x-self.center)/self.scale).clamp(-12,12)
  return torch.stack([torch.cat([b(z[...,:self.context_dim]),o(z)],-1)/self.temperature for b,o in zip(self.base_members,self.option_members)])
 def components(self,x):return self(x).sigmoid()
 def probabilities(self,x):
  p,s,r=self.components(x).unbind(-1)
  return torch.stack([p*s,(1-p)*r,p*(1-s),(1-p)*(1-r)],-1)
 def save(self,path,**metadata):
  torch.save(dict(kind='factorized_preservation_rescue',dim=self.dim,context_dim=self.context_dim,members=len(self.base_members),state=self.state_dict(),metadata=metadata),path)
 @classmethod
 def load(cls,path):
  d=torch.load(path,map_location='cpu',weights_only=False);m=cls(d['dim'],d['context_dim'],d['members']);m.load_state_dict(d['state']);return m.eval()

def measures(gate,x,penalty=1.):
 # Condition the joint mixture using base-success weights. Equal averaging of
 # conditional heads would be inconsistent when replicas disagree about p_B.
 # Log weights also avoid erasing preservation risk when p_B is nearly zero.
 from torch.nn import functional as F
 if hasattr(gate,'components'):
  logits=gate(x);p,s,r=logits.sigmoid().unbind(-1)
  logbase=F.logsigmoid(logits[...,0]);logfail=F.logsigmoid(-logits[...,0])
  q=torch.stack([p*s,(1-p)*r,p*(1-s),(1-p)*(1-r)],-1)
 else:
  logq=gate(x).log_softmax(-1);q=logq.exp();logbase=torch.logsumexp(logq[...,[0,2]],-1);logfail=torch.logsumexp(logq[...,[1,3]],-1)
  p=logbase.exp();s=(logq[...,0]-logq[...,2]).sigmoid();r=(logq[...,1]-logq[...,3]).sigmoid()
 benefit=q[...,1]-q[...,2];harm=1-s;conditional=(logbase.softmax(0)*harm).sum(0);rescue=(logfail.softmax(0)*r).sum(0)
 return dict(lower_gain=benefit.mean(0)-penalty*benefit.std(0,unbiased=False),conditional_harm=(conditional+penalty*harm.std(0,unbiased=False)).clamp(0,1),conditional_harm_mean=conditional,conditional_harm_max_member=harm.amax(0),joint_harm=q[...,2].mean(0)+penalty*q[...,2].std(0,unbiased=False),base_success=p.mean(0),preservation=1-conditional,rescue=rescue,mean_rescue=q[...,1].mean(0),mean_harm=q[...,2].mean(0))
