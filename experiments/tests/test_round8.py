"""Structural risk and whole-option semantics, not merely training convergence."""
import math
import numpy as np
import pytest
import torch
from torch import nn
from experiments.round8.factor_gate import FactorGate,measures
from experiments.round8.policy import PreservePolicy

def fixed_gate(p=.00659,s=.4306,r=.0456,dim=196):
 m=FactorGate(dim=dim).double()
 with torch.no_grad():
  for param in m.parameters():param.zero_()
  for b,o in zip(m.base_members,m.option_members):
   b[-1].bias.fill_(math.log(p/(1-p)));o[-1].bias.copy_(torch.tensor([math.log(s/(1-s)),math.log(r/(1-r))],dtype=torch.float64))
 return m

def test_rare_base_success_does_not_hide_conditional_harm():
 m=fixed_gate();v=measures(m,torch.zeros(2,196,dtype=torch.float64));assert (v['joint_harm']<.005).all();assert (v['conditional_harm']>.56).all();assert (v['lower_gain']>0).all()

def test_baseline_invariant_to_all_option_inputs():
 torch.manual_seed(7);m=FactorGate().double();x=torch.randn(11,196,dtype=torch.float64);x[:,0:184]=x[0,0:184];p=m.probabilities(x);b=p[...,0]+p[...,2]
 torch.testing.assert_close(b,b[:,0:1].expand_as(b),rtol=0,atol=2e-16)

@pytest.mark.parametrize('p,s,r',[(.1,.99,.8),(.9,.999,.2),(.5,.01,.99)])
def test_factorization_is_a_valid_joint_distribution(p,s,r):
 m=fixed_gate(p,s,r);q=m.probabilities(torch.zeros(1,196,dtype=torch.float64));torch.testing.assert_close(q.sum(-1),torch.ones(3,1,dtype=torch.float64));assert (q>=0).all();assert (q<=1).all();torch.testing.assert_close(q[...,1]-q[...,2],torch.full((3,1),(1-p)*r-p*(1-s),dtype=torch.float64))

class FakeProposer(nn.Module):
 def __init__(self):super().__init__();self.dummy=nn.Parameter(torch.zeros(1));self.checkpoint_path='x/flashsac/seed_201/best.pt'

@pytest.fixture
def policy(monkeypatch):
 import experiments.round8.policy as module
 monkeypatch.setattr(module,'canonical_risk',lambda:nn.Linear(1,1))
 def proposal(m,obs,state=None):
  new=torch.zeros(len(obs),device=obs.device,dtype=obs.dtype) if state is None else state+1
  return torch.ones(len(obs),4,device=obs.device,dtype=obs.dtype)*.2,torch.ones(len(obs),device=obs.device,dtype=torch.bool),new
 monkeypatch.setattr(module,'propose',proposal);monkeypatch.setattr(module,'risk_features',lambda m,obs,state=None:(torch.zeros(len(obs),184,dtype=obs.dtype,device=obs.device),torch.zeros(len(obs),dtype=obs.dtype,device=obs.device)))
 return PreservePolicy(FakeProposer(),fixed_gate(.2,.9999,.8),minimum_gain=.01,maximum_harm=.005,allowed_options=[1]).double()

def test_one_feedback_option_then_permanent_base(policy):
 obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1;state=None;active=[]
 for _ in range(60):
  a,state=policy.predict_actions(obs,state);active.append(a[0,5].item()>0)
 assert active==[True]*24+[False]*36;assert state[4].item()==1;assert state[5].item()==1

def test_option_does_not_reinterpret_later_hypothetical_scores(policy):
 obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1;a,state=policy.predict_actions(obs);forecast=state[-1].clone()
 with torch.no_grad():
  for option in policy.gate.option_members:option[-1].bias[0]=-20
 a,state=policy.predict_actions(obs,state);assert a[0,5]>0;torch.testing.assert_close(state[-1],forecast);assert state[3].item()==22

def test_touchdown_stops_an_option(policy):
 obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1;_,state=policy.predict_actions(obs);obs[:,57]=1;a,state=policy.predict_actions(obs,state);assert a[0,5]<0;assert state[3].item()==0

def test_batch_memory_reset_allows_only_new_episodes(policy):
 from experiments.round4.models import tree_map
 obs=torch.zeros(2,88,dtype=torch.float64);obs[:,72]=1;_,state=policy.predict_actions(obs)
 keep=torch.tensor([False,True]);state=tree_map(lambda x:x*keep.reshape(2,*([1]*(x.ndim-1))),state)
 _,state=policy.predict_actions(obs,state);assert state[3].tolist()==[23.,22.];assert state[4].tolist()==[1.,1.]


def test_prediction_metrics_accept_boolean_outcomes(monkeypatch):
 from experiments.round8 import fit
 monkeypatch.setattr(fit,'logits',lambda model,x:torch.zeros(3,len(x),3))
 result=fit.diagnostic(None,np.zeros((4,196)),np.arange(4))
 assert np.isfinite(result['joint_nll']);assert result['joint_nll']==pytest.approx(math.log(4));assert all(np.isfinite(v['nll']) for v in result['components'].values())


@pytest.fixture
def option_policy(monkeypatch):
 import experiments.round8.policy as parent
 import experiments.round8.policy_v2 as module
 monkeypatch.setattr(parent,'canonical_risk',lambda:nn.Linear(1,1))
 def proposal(m,obs,state=None):
  tick=torch.zeros(len(obs),dtype=obs.dtype) if state is None else state+1
  on=(tick<7)|(tick>=12)
  return torch.ones(len(obs),4,dtype=obs.dtype)*.2,on,tick
 monkeypatch.setattr(module,'propose',proposal);monkeypatch.setattr(module,'risk_features',lambda m,obs,state=None:(torch.zeros(len(obs),184,dtype=obs.dtype),torch.zeros(len(obs),dtype=obs.dtype)))
 def make(option):return module.PreserveOptions(FakeProposer(),fixed_gate(.2,.9999,.8,dim=201),allowed_options=[option]).double()
 return make

@pytest.mark.parametrize('option,pattern',[(1,[True]*7+[False]*5+[True]*12),(5,[True]*24)])
def test_v2_internal_veto_is_part_of_the_latched_option(option_policy,option,pattern):
 model=option_policy(option);obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1;state=None;actual=[]
 for _ in range(40):
  action,state=model.predict_actions(obs,state);actual.append(bool(action[0,5]>0))
 assert actual==pattern+[False]*16;assert state[3].item()==1;assert model.last_diagnostics['selected_option'].item()==option

def test_v2_base_probability_does_not_depend_on_projection_or_veto_flag():
 model=FactorGate(dim=201).double();x=torch.randn(8,201,dtype=torch.float64);x[:,:184]=x[0,:184];parts=model.components(x)
 # Equal-context matrix rows can differ by one floating-point ULP.
 torch.testing.assert_close(parts[:,:,0],parts[:,0:1,0].expand_as(parts[:,:,0]),rtol=0,atol=2e-15)

def test_v2_features_expose_saturation_separately_from_request(option_policy):
 model=option_policy(5);obs=torch.zeros(1,88,dtype=torch.float64);obs[:,72]=1;obs[:,64]=1
 aa,_,x=model.build_options(obs,torch.zeros(1,184,dtype=torch.float64),torch.full((1,4),.2,dtype=torch.float64),torch.tensor([False]))
 assert x.shape==(1,8,201);assert torch.all(aa[0,:4]==0);assert torch.all(aa[0,4:]>0);assert x[0,5,184]>.09;assert x[0,5,-4]==0;assert x[0,5,-3]>.09

def test_v2_checkpoint_factory_retains_option_semantics(tmp_path,monkeypatch):
 import experiments.round8.policy_v2 as module
 import experiments.round8.policy as parent
 monkeypatch.setattr(parent,'canonical_risk',lambda:nn.Linear(1,1));monkeypatch.setattr(module,'load_policy',lambda p:FakeProposer())
 path=tmp_path/'factor201.pt';fixed_gate(dim=201).save(path)
 model=PreservePolicy.from_spec(dict(version=2,proposer='x',gate=str(path),option_mode='raw',allowed_options=[4]))
 assert model.option_ids==[4,5,6,7];assert model.allowed_options==[4];assert model.gate.dim==201


def test_conditioning_the_ensemble_weights_base_success():
 model=fixed_gate(.01,.999,.2)
 with torch.no_grad():
  model.base_members[2][-1].bias.fill_(math.log(.99/.01));model.option_members[2][-1].bias[0]=math.log(.001/.999)
 m=measures(model,torch.zeros(1,196,dtype=torch.float64),penalty=0.)
 expected=(.01*.001+.01*.001+.99*.999)/1.01
 assert m['conditional_harm_mean'].item()==pytest.approx(expected);assert m['conditional_harm'].item()>.97;assert m['preservation'].item()<.03

def test_nearly_impossible_baseline_success_does_not_erase_conditional_risk():
 model=fixed_gate(.1,.2,.9)
 with torch.no_grad():
  for base in model.base_members:base[-1].bias.fill_(-1000.)
 m=measures(model,torch.zeros(1,196,dtype=torch.float64));assert m['base_success'].item()==0.;assert m['conditional_harm'].item()==pytest.approx(.8)
