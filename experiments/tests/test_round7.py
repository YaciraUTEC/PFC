"""Contracts required before counterfactual data acquisition."""
import numpy as np
import pytest
from experiments.round7.episode import action_from_delta,residual
from experiments.round7.protocol import create

def test_motor_action_roundtrip_and_abstention():
    for x in [np.zeros(4),np.array([.2,-.6,.1,.4]),np.array([-1.,.8,.2,0.])]:
        np.testing.assert_allclose(residual(action_from_delta(x)),x,atol=1e-7)
    assert action_from_delta(np.full(4,.001))[5]<0

def test_prospective_partitions_are_disjoint_and_transformer_only():
    p=create();parts=[range(*p[k]) for k in ['demonstration_seeds','development_seeds','gate_training_seeds','gate_calibration_seeds','final_test_seeds','confirmation_seeds','shifted_clock_seeds']]
    for i,a in enumerate(parts):
        for b in parts[i+1:]:assert set(a).isdisjoint(b)
    assert p['architecture']=='gtrxl' and p['final_test_locked']


def test_mixed_case_evaluator_preserves_conditions_and_reference_flight():
    from experiments.round7.evaluation import evaluate,cases
    from experiments.round7.episode import fly
    cc=cases([710000],levels=[.295],conditions=['exact'],noises=[2.],clocks=[4.])
    row=evaluate(None,cc,workers=1)['episodes'][0]
    ref=fly(710000,.295,'exact',2.,4.)
    assert row['condition']=='exact' and row['noise']==2. and row['clock']==4.
    assert row['outcome']==ref['outcome'] and row['episode_steps']==ref['episode_steps']
    np.testing.assert_allclose(row['terminal_position'],ref['terminal_position'],atol=0,rtol=0)

def test_sequence_sampling_never_crosses_an_episode_or_supervises_student_prefix():
    from experiments.round7.data import ProposalSequences
    from experiments.round6.env import ZERO
    n=48;actions=np.tile(ZERO,(n,1));actions[8:24]=[.3,-.2,.1,0.,-.2,1.]
    states=np.zeros((n,88),np.float32);states[:24]=1;states[24:]=2;states[:,1]=np.arange(n)
    valid=np.ones(n,bool);valid[:8]=False
    d=dict(s=states,a=actions,valid=valid,records=[dict(start=0,stop=24,seed=1),dict(start=24,stop=48,seed=2)])
    ds=ProposalSequences(d);x,a,m,v=ds.sample(np.random.default_rng(5),batch=64,length=12,burnin=8)
    for xx,mm in zip(x,m):
        values=np.unique(xx[mm>0,0]);assert len(values)<=1
        if len(values) and values[0]==1:assert np.all(xx[mm>0,1]>=8)
    assert len(ds.active_records)==1


def test_independent_replay_matches_the_natural_terminal_and_an_alternate_branch():
    from experiments.round7.replay_fork import ReplayFork
    from experiments.round7.branches import continuation
    from experiments.round7.common import config,silent
    from experiments.round6.env import LandingEnv,ZERO
    with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=2.,fault_clock=4.))
    try:
        with silent():e.reset(seed=710001,options=dict(level=.296))
        history=[]
        for _ in range(300):history.append(ZERO.copy());_,_,d,t,_=e.step(ZERO);assert not(d or t)
        with ReplayFork(e,history) as fork:
            natural=continuation(e)
            branch=fork.restore();first=continuation(branch)
            branch=fork.restore();continuation(branch,np.array([.05,-.03,.04,-.02]),6)
            branch=fork.restore();second=continuation(branch)
            for row in [first,second]:
                assert row['outcome']==natural['outcome'] and row['episode_steps']==natural['episode_steps']
                assert row['stable_seconds']==natural['stable_seconds']
                np.testing.assert_array_equal(row['terminal_position'],natural['terminal_position'])
            assert branch.descent.controller is branch.c and fork.maximum_error==0
    finally:e.close()


def test_native_transformer_wrapper_preserves_executed_flash_actions():
    import torch
    from experiments.round7.common import OLD,old_model
    from experiments.round7.episode import load_policy
    from experiments.round7.policy import TransformerPolicy
    from experiments.round6.offpolicy import compress_actions
    torch.set_num_threads(1)
    native=load_policy(old_model());wrapper=TransformerPolicy(load_policy(old_model()),mode='native').double().eval()
    with np.load(OLD/'data/route_000.npz') as z:
        obs=z['s'];obs=obs[obs[:,57]<.5][300:324]
    ns=ws=None
    with torch.no_grad():
        for row in obs:
            x=torch.as_tensor(row[None],dtype=torch.float64);a,ns=native.predict_actions(x,ns);b,ws=wrapper.predict_actions(x,ws)
            np.testing.assert_allclose(compress_actions(a.numpy()),compress_actions(b.numpy()),atol=1e-9,rtol=1e-9)


def test_physical_screen_uses_observable_inputs_and_preserves_off():
    import torch
    from experiments.round7.screen import LandingScreen
    from experiments.round7.common import OLD
    with np.load(OLD/'data/route_000.npz') as z:raw=z['s'][200:232]
    x=torch.as_tensor(raw,dtype=torch.float64);screen=LandingScreen().double()
    zero=torch.zeros((len(x),4),dtype=x.dtype);dt=torch.full((len(x),),6.,dtype=x.dtype)
    torch.testing.assert_close(screen(x,zero,dt),zero,atol=0,rtol=0)
    delta=torch.full_like(zero,.05);out=screen(x,delta,dt)
    assert torch.isfinite(out).all() and ((out==delta)|(out==0)).all()
    # Artificial corruption of masked previous-gate channels cannot change the physical screen.
    changed=x.clone();changed[:,35:40]=.99;changed[:,59]=.99;changed[:,77:79]=.99
    torch.testing.assert_close(screen(changed,delta,dt),out,atol=0,rtol=0)


def test_running_intervention_can_abort_before_its_duration_expires():
    import torch
    from torch import nn
    from experiments.round7.policy import TransformerPolicy
    from experiments.round7.common import OLD
    from experiments.round7.episode import residual
    class FixedProposal(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(()))
        def forward(self,x,state=None):
            mu=torch.zeros((*x.shape[:2],6),dtype=x.dtype);mu[...,:4]=.2;mu[...,5]=10
            return mu,torch.zeros(x.shape[:2],dtype=x.dtype),torch.zeros((len(x),1),dtype=x.dtype)
    class ConditionalOutcome(nn.Module):
        def probabilities(self,x):
            safe=x[...,72]>.5;duration=x[...,-1].clamp(max=1.);p=torch.zeros((*x.shape[:-1],4),dtype=x.dtype)
            p[...,1]=torch.where(safe,.5+.4*duration,0.)
            p[...,2]=torch.where(safe,.001,.99)
            p[...,0]=1-p[...,1]-p[...,2]
            return p[None].expand(3,*p.shape)
    policy=TransformerPolicy(FixedProposal(),ConditionalOutcome(),mode='counterfactual').double().eval()
    with np.load(OLD/'data/route_000.npz') as z:row=z['s'][100].copy()
    row[57]=0;row[72]=.9;x=torch.as_tensor(row[None],dtype=torch.float64)
    with torch.no_grad():
        first,state=policy.predict_actions(x)
        assert np.max(np.abs(residual(first[0].numpy())))>.002 and state[-1].item()==95
        x[:,72]=.1;second,state=policy.predict_actions(x,state)
        assert np.max(np.abs(residual(second[0].numpy())))==0 and state[-1].item()==0


def test_each_recovery_controller_sees_every_environmental_stratum():
    import itertools
    from experiments.round7.gate_data import case_settings
    desired=set(itertools.product(['exact','estimated'],[1.,2.],[None,4.,6.,8.]))
    cells=[case_settings(i,10) for i in range(160)]
    for controller in range(10):
        got={(r['condition'],r['noise'],r['clock']) for r in cells if r['controller']==controller}
        assert got==desired


def test_controller_identity_is_known_policy_metadata_not_route_identity():
    from experiments.round7.policy import controller_descriptor
    from experiments.round7.common import OLD,OUT
    old=controller_descriptor(OLD/'phase3/flashsac/gtrxl/seed_203/best.pt')
    new=controller_descriptor(OUT/'refinement/flashsac/seed_303/model.pt')
    np.testing.assert_array_equal(old,[1,0,0,1,0,0,0])
    np.testing.assert_array_equal(new,[1,0,0,1,0,0,1])

def test_decision_logging_does_not_change_the_physical_flight(tmp_path):
    from experiments.round7.evaluation import cases,evaluate
    cc=cases([710002],levels=[.295],conditions=['exact'],noises=[2.],clocks=[4.])
    plain=evaluate(None,cc,workers=1)['episodes'][0]
    traced=evaluate(None,cc,workers=1,capture_directory=tmp_path)['episodes'][0]
    assert plain['outcome']==traced['outcome'] and plain['episode_steps']==traced['episode_steps']
    np.testing.assert_array_equal(plain['terminal_position'],traced['terminal_position'])
    with np.load(traced['trace_path']) as z:
        assert len(z['observation'])==traced['episode_steps'] and z['observation'].shape[1]==88
        assert z['motor_command_rpm'].shape[1]==4 and np.isfinite(z['next_measured']).all()


def test_repeated_noise_settings_do_not_inflate_route_confidence():
    from experiments.round7.statistics import paired
    def fixture(noises):
        baseline=[];models=[[] for _ in range(5)]
        for route in range(4):
            for level in [.295,.296]:
                for noise in noises:
                    row=dict(seed=route,level=level,condition='estimated',noise=noise,clock=None,stable_landing=route!=0)
                    baseline.append(row)
                    for i in range(5):models[i].append(dict(row,stable_landing=route!=1))
        return [dict(episodes=x) for x in models],dict(episodes=baseline)
    one=paired(*fixture([1.]));two=paired(*fixture([1.,2.]))
    assert one['route_clusters']==two['route_clusters']==4 and one['model_seeds']==two['model_seeds']==5
    assert one['lower95']==two['lower95'] and one['upper95']==two['upper95']
    assert two['losses']==2*one['losses'] and two['harmed_route_clusters']==one['harmed_route_clusters']



def test_repaired_proposal_has_unique_bounded_physical_output_and_gradients():
    import torch
    from experiments.round7.proposal_model import ProposalAgent
    from experiments.round6.models import configuration
    model=ProposalAgent(dict(configuration('gtrxl'),proposal_repair=True,proposal_gate='observed_fault',direct_motor_head=True))
    mu=torch.zeros(2,3,6,requires_grad=True);target=torch.full((2,3,4),.2)
    pred=model.physical_delta(mu);(pred-target).square().mean().backward()
    assert mu.grad[...,:4].abs().sum()>0 and mu.grad[...,4:].abs().sum()==0
    assert model.physical_delta(torch.full_like(mu,100)).abs().max()<=1


def test_repaired_eligibility_is_observable_latched_and_sequence_consistent():
    import torch
    from experiments.round7.proposal_model import ProposalAgent
    from experiments.round6.models import configuration
    torch.set_num_threads(1)
    model=ProposalAgent(dict(configuration('gtrxl'),proposal_repair=True,proposal_gate='observed_fault',direct_motor_head=True)).eval()
    x=torch.zeros(1,8,88);x[:,3,31]=.12
    with torch.no_grad():
        full,_,_=model(x);memory=None;stream=[]
        for j in range(x.shape[1]):
            out,_,memory=model(x[:,j:j+1],memory);stream.append(out)
        streamed=torch.cat(stream,1)
    assert (full[0,:3,5]<0).all() and (full[0,3:,5]>0).all()
    torch.testing.assert_close(full,streamed,atol=2e-5,rtol=2e-5)
    changed=x.clone();changed[:,:,35:40]=.9;changed[:,:,59]=.8
    with torch.no_grad():other,_,_=model(changed)
    torch.testing.assert_close(other,full,atol=0,rtol=0)


def test_repaired_proposal_checkpoint_preserves_actions(tmp_path):
    import torch
    from experiments.round7.proposal_model import ProposalAgent
    from experiments.round7.episode import load_policy
    from experiments.round6.models import configuration
    torch.set_num_threads(1)
    model=ProposalAgent(dict(configuration('gtrxl'),proposal_repair=True,proposal_gate='observed_fault',direct_motor_head=True)).double().eval()
    path=tmp_path/'proposal.pt';model.save(path);restored=load_policy(path)
    x=torch.zeros(1,88,dtype=torch.float64);x[:,31]=.2
    with torch.no_grad():a,_=model.predict_actions(x);b,_=restored.predict_actions(x)
    torch.testing.assert_close(a,b,atol=0,rtol=0)



def test_terminal_feedback_option_uses_the_actual_remaining_mission_time():
    import torch
    from experiments.round7.policy import option_durations
    x=torch.zeros(2,88);x[:,72]=torch.tensor([.5,10/1440])
    ticks=option_durations(x)
    torch.testing.assert_close(ticks,torch.tensor([[1.,24.,96.,720.],[1.,10.,10.,10.]]))



def test_complete_policy_off_replay_checks_the_authorizer_not_only_native_gate():
    import torch
    from torch import nn
    from experiments.round7.episode import load_policy
    from experiments.round7.policy import TransformerPolicy
    from experiments.round7.composed_replay import certify
    from experiments.round7.common import OUT,OLD,read
    torch.set_num_threads(1)
    class ConstantGate(nn.Module):
        def __init__(self,accept):super().__init__();self.accept=accept
        def probabilities(self,x):
            p=torch.zeros((3,*x.shape[:-1],4),dtype=x.dtype);p[...,1 if self.accept else 0]=1.;return p
    from experiments.round7.proposal_model import ProposalAgent
    from experiments.round6.models import configuration
    def source():
        m=ProposalAgent(dict(configuration('gtrxl'),proposal_repair=True,proposal_gate='observed_fault',direct_motor_head=True)).double().eval()
        with torch.no_grad():
            for p in m.actor.parameters():p.zero_()
            head=[x for x in m.actor.modules() if isinstance(x,nn.Linear)][-1];head.bias[:4]=.2
        return m
    history=np.zeros((96,88),np.float32);history[:,31]=.2;history[:,72]=.5
    for accept in [False,True]:
        model=TransformerPolicy(source(),ConstantGate(accept),mode='counterfactual',minimum_gain=.01,maximum_harm=.02).double().eval()
        flags,_=certify(model,[history]);state=None;all_off=True
        with torch.no_grad():
            for row in history:
                action,state=model.predict_actions(torch.as_tensor(row[None],dtype=torch.float64),state);all_off &=bool(action[0,5]<0)
        if accept:assert not flags[0] and not all_off
        else:assert flags[0] and all_off


def test_terminal_targets_do_not_count_duplicate_options_as_independent_futures():
    from experiments.round7.refine import terminal_targets
    x=np.zeros((2,196),np.float32);x[:,184]=.1;x[:,-1]=1
    z=dict(x=x,y=np.array([1,1]),trace_index=np.array([10,10]),future=np.array([0,0]))
    assert terminal_targets(z)=={}
    z['future'][1]=1
    target=terminal_targets(z)[10];assert target[1] and target[0][0]==np.float32(.1)



def test_nonnegative_continuation_rejects_an_option_that_the_model_now_values_negatively():
    import torch
    from torch import nn
    from experiments.round7.policy import TransformerPolicy
    torch.set_num_threads(1)
    class FixedProposal(nn.Module):
        def __init__(self):super().__init__();self.anchor=nn.Parameter(torch.zeros(()))
        def forward(self,x,state=None):
            mu=torch.zeros((*x.shape[:2],6),dtype=x.dtype);mu[...,:4]=.2;mu[...,5]=10
            return mu,torch.zeros(x.shape[:2],dtype=x.dtype),torch.zeros((len(x),1),dtype=x.dtype)
    class ChangeOfBenefit(nn.Module):
        def probabilities(self,x):
            high=x[...,72]>.95;d=x[...,-1].clamp(max=1);p=torch.zeros((*x.shape[:-1],4),dtype=x.dtype)
            p[...,1]=torch.where(high,.2+.1*d,.01);p[...,2]=torch.where(high,.001,.012);p[...,0]=1-p[...,1]-p[...,2]
            return p[None].expand(3,*p.shape)
    for threshold in [-.005,0.]:
        model=TransformerPolicy(FixedProposal(),ChangeOfBenefit(),mode='counterfactual',minimum_gain=.005,maximum_harm=.02,continuation_gain=threshold).double().eval()
        x=torch.zeros(1,88,dtype=torch.float64);x[:,72]=1
        with torch.no_grad():
            first,state=model.predict_actions(x);assert state[-1].item()==95
            x[:,72]=.9;second,state=model.predict_actions(x,state)
        assert bool(second[0,5]>0)==(threshold<0)
        if threshold==0:assert state[-1].item()==0



@pytest.mark.parametrize('condition,noise',[('exact',1.),('exact',2.),('estimated',1.),('estimated',2.)])
def test_natural_replay_fork_never_changes_the_caller_or_next_physical_step(condition,noise):
    from experiments.round7.replay_fork import ReplayFork
    from experiments.round7.common import config,silent
    from experiments.round6.env import LandingEnv,ZERO
    with silent():e=LandingEnv(condition=condition,config=config(noise_multiplier=noise))
    try:
        with silent():obs,_=e.reset(seed=729811,options=dict(level=.295))
        history=[]
        for _ in range(80):history.append(ZERO.copy());obs,_,d,t,_=e.step(ZERO);assert not(d or t)
        with ReplayFork(e,history) as fork:
            natural,_,_,_,_=e.step(ZERO);physical=e.core.obs_raw[0].copy()
            branch=fork.restore();again,_,_,_,_=branch.step(ZERO)
            np.testing.assert_array_equal(natural,again);np.testing.assert_array_equal(physical,branch.core.obs_raw[0])
            branch=fork.restore();branch.step(np.array([.03,-.02,.01,0.,-.5,1.],np.float32))
            assert e.t==81;np.testing.assert_array_equal(e.frames[-1],natural[-1])
            branch=fork.restore();again,_,_,_,_=branch.step(ZERO);np.testing.assert_array_equal(again,natural)
    finally:e.close()



def test_frozen_gtrxl_cache_matches_outputs_memory_resets_and_branch_switches():
    import copy,torch
    from experiments.round6.models import Agent,configuration
    from experiments.round7.fast_gtrxl import install
    torch.set_num_threads(1);torch.manual_seed(761)
    cfg=dict(configuration('gtrxl'),width=32,layers=2,memory=16)
    a=Agent(cfg).double().eval();b=copy.deepcopy(a)
    for m in [a,b]:
        for p in m.parameters():p.requires_grad_(False)
    install(b);rng=np.random.default_rng(97);x=torch.as_tensor(rng.normal(0,.2,(3,150,88)),dtype=torch.float64);sa=sb=None;saved=None
    with torch.no_grad():
        for t in range(150):
            ma,va,sa=a(x[:,t:t+1],sa);mb,vb,sb=b(x[:,t:t+1],sb)
            torch.testing.assert_close(ma,mb,atol=1e-12,rtol=1e-12);torch.testing.assert_close(va,vb,atol=1e-12,rtol=1e-12);torch.testing.assert_close(sa,sb,atol=1e-12,rtol=1e-12)
            if t==35:saved=(sa.clone(),sb.clone())
            if t==65:
                keep=torch.tensor([False,True,False])[:,None,None,None];sa=sa*keep;sb=sb*keep
            if t==95:sa,sb=(z.clone() for z in saved)
        ma,_,sa=a(x[:,:4],sa);mb,_,sb=b(x[:,:4],sb)
        torch.testing.assert_close(ma,mb,atol=1e-12,rtol=1e-12);torch.testing.assert_close(sa,sb,atol=1e-12,rtol=1e-12)
