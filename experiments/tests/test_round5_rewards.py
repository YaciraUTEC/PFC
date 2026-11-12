import numpy as np
import pytest
import torch
from experiments.round5.bc import setup,starting_config
from experiments.round5.models import Agent

def reward_data():
    rng=np.random.default_rng(44);n=6*128;s=rng.normal(0,.2,(n,80)).astype(np.float32);s[:,79]=0
    action=np.tile(np.array([0,0,0,0,-1,-1],np.float32),(n,1));reward=np.full(n,-.001,np.float32)
    done=np.zeros(n,bool);expert=np.zeros(n,bool);train=np.zeros(n,bool);validation=np.zeros(n,bool);starts=np.zeros(n,np.int64);records=[]
    for i in range(6):
        first=i*128;stop=first+128;exact=i in [2,3];success=i%2==0;split='train' if i<4 else 'validation'
        s[first:stop,48:52]=0 if exact else .1;done[stop-1]=True;reward[stop-1]=100 if success else -100
        train[first:stop]=split=='train';validation[first:stop]=split=='validation';expert[first:stop]=split=='train' and success;starts[first:stop]=first
        records.append(dict(start=first,stop=stop,split=split,seed=1 if i<4 else 2,level=.295,condition='exact' if exact else'estimated',stable_landing=success,configuration=dict(intervention='residual')))
    return dict(s=s,a=action,executed=action,ns=s+.001,done=done,train=train,validation=validation,expert=expert,episode_start=starts,records=records,reward=reward)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='LocalGPUreward fit')
def test_transition_reward_handles_mixed_actions_and_actual_prefixes():
    from experiments.round5.rewards import MEIRL
    setup(41);data=reward_data();reward=MEIRL(data);report=reward.fit_dynamics(data,41,updates=1)
    assert len(report['elites'])==5
    policy=Agent(starting_config('lstm')).cuda();ids=np.r_[np.arange(64),np.arange(256,320)]
    learner=dict(s=data['s'][ids],a=data['a'][ids],done=data['done'][ids],logp=np.zeros(len(ids),np.float32))
    loss=reward.adversarial_update_mixed(data,learner,policy,41,updates=1)
    assert np.isfinite(loss)
    with torch.no_grad():value=reward(torch.tensor(data['s'][:8],device='cuda'),torch.tensor(data['a'][:8],device='cuda'))
    assert torch.isfinite(value).all() and value.abs().max()<=5

@pytest.mark.skipif(not torch.cuda.is_available(),reason='LocalGPUreward fit')
def test_preference_reward_runs_matching_landing_pairs_and_filter():
    from experiments.round5.rewards import RIME
    setup(43);data=reward_data();reward=RIME(data);report=reward.fit(data,43,warm_updates=1,preference_updates=1,epochs=1)
    assert report['pairs']>0 and report['validation_pairs']>0
    assert 0<=report['heldout_landing_preference_accuracy']<=1
