import torch
from torch import nn
from experiments.tests.test_round8 import fixed_gate,FakeProposer
from experiments.round8.policy_v3 import EntryPreserveOptions

def test_entry_veto_does_not_cancel_a_committed_raw_option_and_is_per_episode(monkeypatch):
 import experiments.round8.policy as parent
 import experiments.round8.policy_v2 as module
 monkeypatch.setattr(parent,'canonical_risk',lambda:nn.Linear(1,1))
 def proposal(model,obs,state=None):return torch.full((len(obs),4),.2,dtype=obs.dtype),obs[:,0]>0,torch.zeros(len(obs),dtype=obs.dtype)
 monkeypatch.setattr(module,'propose',proposal);monkeypatch.setattr(module,'risk_features',lambda model,obs,state=None:(torch.zeros(len(obs),184,dtype=obs.dtype),torch.zeros(len(obs),dtype=obs.dtype)))
 model=EntryPreserveOptions(FakeProposer(),fixed_gate(.2,.9999,.8,201),allowed_options=[5]).double();obs=torch.zeros(2,88,dtype=torch.float64);obs[:,72]=1.;state=None;active=[]
 for tick in range(40):
  obs[:,0]=torch.tensor([tick==0,tick==5],dtype=obs.dtype);a,state=model.predict_actions(obs,state);active.append((a[:,5]>0).tolist())
 assert [x[0] for x in active]==[True]*24+[False]*16;assert [x[1] for x in active]==[False]*5+[True]*24+[False]*11;assert state[3].tolist()==[1.,1.]

def test_version3_factory_preserves_explicit_raw_option_and_entry_rule(tmp_path,monkeypatch):
 from experiments.round8.loading import Policy
 import experiments.round8.policy as parent
 import experiments.round8.policy_v2 as module
 monkeypatch.setattr(parent,'canonical_risk',lambda:nn.Linear(1,1));monkeypatch.setattr(module,'load_policy',lambda path:FakeProposer());p=tmp_path/'gate.pt';fixed_gate(dim=201).save(p)
 model=Policy.from_spec(dict(version=3,proposer='x',gate=str(p),option_mode='both',allowed_options=[1,5]));assert isinstance(model,EntryPreserveOptions);assert model.allowed_options==[1,5]
