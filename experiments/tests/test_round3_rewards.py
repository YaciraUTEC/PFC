import numpy as np
import torch
import pytest
from gymnasium.spaces import Box
from experiments.round3.common import BoundedPolicy,load_data
from experiments.round3.rewards import CSIL,MEIRL,RIME,tensor

@pytest.fixture(scope='module')
def data():return load_data()

def test_bounded_policy_density_matches_sampled_actions():
    p=BoundedPolicy(Box(-10,10,(48,),dtype=np.float32),Box(-1,1,(5,),dtype=np.float32),lambda _:3e-4)
    obs=torch.randn(32,48);actions,values,lp=p(obs)
    assert torch.all(torch.abs(actions)<1)
    _,lp_again,_=p.evaluate_actions(obs,actions)
    torch.testing.assert_close(lp,lp_again,rtol=1e-4,atol=1e-4)

@pytest.mark.parametrize('cls',[CSIL,MEIRL,RIME])
def test_reward_batch_and_gradient_contract(data,cls):
    model=cls(data);ix=np.flatnonzero(data['expert'])[:8]
    s=tensor(data['s'][ix]);a=tensor(data['a'][ix]);ns=tensor(data['ns'][ix]);done=tensor(data['done'][ix])
    reward=model(s,a,ns,done)
    assert reward.shape==(8,);assert torch.isfinite(reward).all()
    reward.mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if cls is RIME:
        result=model.members(s.reshape(2,2,2,48),a.reshape(2,2,2,5))
        assert result.shape==(3,2,2,2)

def test_meirl_absorbing_potential_and_discount(data):
    model=MEIRL(data)
    with torch.no_grad():
        for p in model.r.parameters():p.zero_()
        for p in model.phi.parameters():p.zero_()
        model.phi[-1].bias.fill_(2.)
    s=torch.zeros(2,48,device='cuda');a=torch.zeros(2,5,device='cuda')
    v=model.shaped(s,a,tensor([0,1]))
    torch.testing.assert_close(v,tensor([-.01,-2.]),rtol=1e-4,atol=1e-5)