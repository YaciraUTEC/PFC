import torch
from torch import nn
from experiments.tests.test_round8 import FakeProposer,fixed_gate
from experiments.round8.policy_guarded import ObservedLossOptions

def model(monkeypatch):
 import experiments.round8.policy as parent
 import experiments.round8.policy_v2 as module
 monkeypatch.setattr(parent,'canonical_risk',lambda:nn.Linear(1,1));monkeypatch.setattr(module,'propose',lambda m,obs,state=None:(torch.full((len(obs),4),.2,dtype=obs.dtype),torch.ones(len(obs),dtype=torch.bool),torch.zeros(len(obs),dtype=obs.dtype)));monkeypatch.setattr(module,'risk_features',lambda m,obs,state=None:(torch.cat([obs,torch.zeros(len(obs),96,dtype=obs.dtype)],-1),torch.zeros(len(obs),dtype=obs.dtype)))
 return ObservedLossOptions(FakeProposer(),fixed_gate(.2,.9999,.8,201),allowed_options=[5],minimum_gain=.01,maximum_harm=.02).double()

def test_healthy_observed_effectiveness_blocks_even_high_predicted_benefit(monkeypatch):
 m=model(monkeypatch);obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1.;state=None
 for _ in range(12):a,state=m.predict_actions(obs,state);assert a[0,5]<0
 assert state[3].item()==0
 obs[:,31]=.12;a,state=m.predict_actions(obs,state);assert a[0,5]>0;obs[:,31]=0
 for _ in range(23):a,state=m.predict_actions(obs,state);assert a[0,5]>0
 a,state=m.predict_actions(obs,state);assert a[0,5]<0;assert state[3].item()==1

def test_uncertainty_units_and_entry_mask_are_explicit(monkeypatch):
 m=model(monkeypatch);m.sigma_multiplier=2.;obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1.;obs[:,31]=.12;obs[:,48]=.1;a,_=m.predict_actions(obs);assert a[0,5]<0;obs[:,48]=0.;a,_=m.predict_actions(obs);assert a[0,5]>0

def test_off_proof_margin_does_not_hide_near_boundary_candidates(monkeypatch):
 m=model(monkeypatch);obs=torch.zeros(1,88,dtype=torch.float64);obs[:,31]=.07999998;obs[:,72]=1.;raw=torch.full((1,4),.2,dtype=obs.dtype);ctx=torch.cat([obs,torch.zeros(1,96,dtype=obs.dtype)],-1);aa,_,x=m.build_options(obs,ctx,raw,torch.tensor([True]));actual,_=m.assess(x,aa);permissive,_=m.assess(x,aa,margin=1e-7);assert not actual[0,5];assert permissive[0,5]


def test_memory_is_kept_when_heads_are_skipped_outside_support_with_partial_resets():
 import numpy as np
 from experiments.round8.common import OUT,read
 from experiments.round8.policy_guarded import GuardedFactory
 from experiments.round7.fast_gtrxl import install
 from experiments.round4.models import tree_map
 torch.set_num_threads(1);spec=dict(read(OUT/'h5_conservative_validation_conclusion.json')['entries'][0]['spec'],version=4,option_mode='both',allowed_options=[5],observed_loss_min=.02,sigma_multiplier=0.);slow=install(ObservedLossOptions.from_spec(spec));fast=install(GuardedFactory.from_spec(spec));slow.gate=fixed_gate(.2,.9999,.8,201);fast.gate=fixed_gate(.2,.9999,.8,201);slow.proposer.threshold=fast.proposer.threshold=-1e20;ss=fs=None;rng=np.random.default_rng(771);skipped=0
 with torch.no_grad():
  for tick in range(180):
   if tick==90:
    keep=torch.tensor([False,True,True]);ss=tree_map(lambda x:x*keep.reshape(3,*([1]*(x.ndim-1))),ss);fs=tree_map(lambda x:x*keep.reshape(3,*([1]*(x.ndim-1))),fs)
   obs=torch.as_tensor(rng.normal(0,.1,(3,88)),dtype=torch.float64);obs[:,57]=0.;obs[:,72]=1.;obs[:,64:68]=.4;obs[:,31:35]=0.;obs[0,31]=.4 if 30<=tick<89 or tick>=110 else 0.;obs[1,31]=.4 if tick>=50 else 0.;obs[2,31]=.4 if tick>=145 else 0.
   a,ss=slow.predict_actions(obs,ss);b,fs=fast.predict_actions(obs,fs);torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ss[2],fs[2],rtol=0,atol=0);torch.testing.assert_close(ss[3],fs[3],rtol=0,atol=0);torch.testing.assert_close(ss[5],fs[5],rtol=0,atol=0);skipped+=not bool(fast.last_diagnostics['proposal_evaluated'].any())
 assert skipped>30
