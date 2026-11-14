import numpy as np
import pytest
from experiments.round6.supervisor import RecoveryDescent
from experiments.round6.env import LandingEnv,ZERO
from experiments.round6.controllers import CoordinatedDSLPID
from gym_pybullet_drones.utils.enums import DroneModel

def state(height=1.,spin=0.):
    s=np.zeros(20);s[6]=1.;s[2]=height;s[15]=spin;return s

def test_low_spin_kinematic_descent_has_no_artificial_altitude_equilibrium():
    d=RecoveryDescent();s=state();goal=np.zeros(3)
    for _ in range(900):
        target,vel=d.reference(s,goal,1/48)
        s[12]=vel[2];s[2]+=vel[2]/48
        if s[2]<.035:break
    assert s[2]<.035 and d.started

def test_high_spin_stops_above_terminal_envelope_and_cannot_force_touchdown():
    d=RecoveryDescent();s=state(.18,4.);goal=np.zeros(3)
    for _ in range(600):target,vel=d.reference(s,goal,1/48)
    assert vel[2]==0 and target[2]>=.17 and d.phase=='recovery'

def test_reference_does_not_follow_upward_sensor_excursions():
    d=RecoveryDescent();s=state(.8);goal=np.zeros(3)
    for _ in range(30):target,_=d.reference(s,goal,1/48)
    height=target[2];s[2]=1.;s[15]=4.
    target,_=d.reference(s,goal,1/48)
    assert target[2]<=height+1e-12

def test_environment_contract_and_canonical_executed_off():
    e=LandingEnv(config=dict(control_kind='coordinated',manager='coordinated'))
    try:
        o,_=e.reset(seed=610099,options=dict(level=.295))
        assert o.shape==(1,88) and np.isfinite(o).all()
        o,r,d,t,i=e.step(np.r_[np.ones(5),-1.])
        assert not i['gate_active']
        np.testing.assert_array_equal(i['executed_action'][-2:],[-1.,-1.])
        before=e._state().copy();e.true_eta[:]=.1;e.level=.9;e.motor=3
        np.testing.assert_array_equal(before,e._state())
    finally:e.close()

def test_fault_heading_release_retains_yaw_braking():
    c=CoordinatedDSLPID(DroneModel.CF2X,filter_tau=0.,yaw_priority=1.)
    c.set_motor_effectiveness([.705,1,1,1])
    w=c.prepare_control(control_timestep=1/48,cur_pos=np.array([0.,0.,1.]),cur_quat=np.array([0.,0.,0.,1.]),cur_vel=np.zeros(3),cur_ang_vel=np.array([0.,0.,3.]),target_pos=np.array([0.,0.,1.]))
    assert w[3]<0
    rpm,_,_=c.apply_prepared();assert (rpm<=21700+1e-9).all() and (rpm>=0).all()
from experiments.round6.curriculum import NoiseSchedule
from experiments.round6.models import Agent,configuration
from experiments.round6.offpolicy import OffPolicy,project_distribution,compress_actions
from experiments.round6.train_offpolicy import Trainer,Replay
import torch

def test_noise_schedule_linear_plateaus_and_adaptive_uses_severe_strata():
    s=NoiseSchedule('linear',100)
    assert s.update(0,[])==1 and s.update(40,[])==1.5 and s.update(100,[])==2
    a=NoiseSchedule('adaptive',100)
    mild=[dict(level=0.,stable_landing=True)]*20
    assert a.update(20,mild)==1.
    severe=[dict(level=l,stable_landing=True) for l in [.28,.29,.295] for _ in range(4)]
    assert a.update(40,severe)>1.
    b=NoiseSchedule('adaptive',100);b.upper=1.5
    assert b.update(40,mild+[dict(level=l,stable_landing=False) for l in [.28,.29,.295] for _ in range(4)])<1.5

def test_distributional_terminal_projection_preserves_mass_and_reward():
    log=torch.full((3,101),-np.log(101),dtype=torch.float32)
    reward=torch.tensor([-5.,0.,5.]);done=torch.ones(3);entropy=torch.zeros(3)
    p=project_distribution(log,reward,done,entropy)
    torch.testing.assert_close(p.sum(-1),torch.ones(3))
    value=(p*torch.linspace(-5,5,101)).sum(-1)
    torch.testing.assert_close(value,reward,atol=1e-5,rtol=0)

@pytest.mark.parametrize('method',['flashsac','fql'])
def test_offpolicy_gradient_update_is_finite_and_does_not_change_memory(method):
    torch.set_num_threads(1);torch.manual_seed(7)
    prior=Agent(configuration('mamba3')).cuda();m=OffPolicy(prior,method).cuda()
    frozen=[p.detach().clone() for p in m.prior.parameters()]
    trainer=Trainer(m);trainer.normalizer.G_r_max.fill_(100.);trainer.normalizer.G_rms.var.fill_(400.)
    x=torch.randn(16,m.dim,device='cuda');a=torch.rand(16,4,device='cuda')*2-1
    batch=(x,a,torch.randn(16,device='cuda'),x+.01,torch.ones(16,device='cuda'))
    result=trainer.update(batch)
    assert np.isfinite(result['critic_loss'])
    for before,after in zip(frozen,m.prior.parameters()):torch.testing.assert_close(before,after,rtol=0,atol=0)
    actions,state=m.predict_actions(torch.randn(2,88,device='cuda'))
    assert actions.shape==(2,6) and torch.isfinite(actions).all()

def test_replay_keeps_actions_matched_to_the_transition():
    replay=Replay(3,capacity=3)
    x=np.arange(12,dtype=np.float32).reshape(4,3);a=np.arange(16,dtype=np.float32).reshape(4,4)
    replay.append(x,a,np.arange(4),x+100,np.ones(4))
    assert replay.n==3
    for i in range(3):
        raw=int(replay.x[i,0]/3);np.testing.assert_array_equal(replay.a[i],a[raw]);np.testing.assert_array_equal(replay.nx[i],x[raw]+100)

@pytest.mark.parametrize('arch',['mamba3','gtrxl'])
def test_streaming_architecture_matches_contiguous_sequence(arch):
    torch.set_num_threads(1);torch.manual_seed(8);m=Agent(configuration(arch)).eval();x=torch.randn(2,5,88)
    with torch.no_grad():
        full,_=m.encoder(x);state=None;out=[]
        for t in range(5):z,state=m.encoder(x[:,t:t+1],state);out.append(z)
    torch.testing.assert_close(torch.cat(out,1),full,atol=3e-5,rtol=3e-4)
from experiments.round6.replay import certify

def test_off_replay_certification_matches_streaming_gates():
    torch.set_num_threads(1);torch.manual_seed(16)
    m=Agent(configuration('gtrxl')).double().eval();x=np.random.default_rng(6).normal(size=(145,88)).astype(np.float32)
    flags,_=certify(m,[x,x[:77]])
    assert flags.all()
    with torch.no_grad():m.actor[-1].bias[5]=2.
    flags,_=certify(m,[x,x[:77]])
    assert not flags.any()

def test_fql_evaluation_noise_is_batch_and_episode_independent():
    m=OffPolicy(Agent(configuration('mamba3')),'fql').double().eval();m.evaluation_seed=201
    a=m.evaluation_noise(torch.arange(7));b=torch.cat([m.evaluation_noise(torch.arange(3)),m.evaluation_noise(torch.arange(3,7))])
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.isfinite(a).all() and a.std()>.3
from experiments.round6.evaluate import evaluate

def test_replay_off_policy_keeps_the_same_physical_flight():
    cfg=dict(control_kind='dslpid',manager='legacy',readiness=False,estimator='original',emergency='legacy',reserve=.005,gate_mode='learned',noise_multiplier=2.)
    model=Agent(configuration('mamba3')).double().eval()
    baseline,traces=evaluate(cfg,'estimated',seeds=[610097],levels=[.295],workers=1,capture=True)
    flags,_=certify(model,[traces[(.295,610097)]])
    assert flags[0]
    random_state=torch.get_rng_state().clone();cuda_state=torch.cuda.get_rng_state().clone()
    actual=evaluate(cfg,'estimated',seeds=[610097],levels=[.295],model=model,workers=1)
    torch.testing.assert_close(torch.get_rng_state(),random_state,rtol=0,atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(),cuda_state,rtol=0,atol=0)
    a=actual['episodes'][0];b=baseline['episodes'][0]
    assert a['outcome']==b['outcome'] and a['stable_landing']==b['stable_landing']
    np.testing.assert_array_equal(a['terminal_position'],b['terminal_position'])
    np.testing.assert_array_equal(a['terminal_angular_velocity'],b['terminal_angular_velocity'])
import copy

def test_coordinated_public_control_accepts_the_original_positional_interface():
    c=CoordinatedDSLPID(DroneModel.CF2X,filter_tau=.04);other=copy.deepcopy(c)
    args=(1/48,np.array([0.,0.,1.]),np.array([0.,0.,0.,1.]),np.zeros(3),np.zeros(3),np.array([.1,.1,1.]))
    w=c.prepare_control(*args)
    v=other.prepare_control(control_timestep=args[0],cur_pos=args[1],cur_quat=args[2],cur_vel=args[3],cur_ang_vel=args[4],target_pos=args[5])
    np.testing.assert_array_equal(w,v)
    c.reset();rpm,_,_=c.computeControl(*args);assert np.isfinite(rpm).all()
@pytest.mark.parametrize('arch',['mamba3','gtrxl'])
def test_replay_long_sequence_signal_matches_stepwise_memory(arch):
    torch.set_num_threads(1);torch.manual_seed(28)
    m=Agent(configuration(arch)).double().eval()
    with torch.no_grad():
        m.actor[-1].weight.normal_(0,.12);m.actor[-1].bias[5]=0.
    x=np.random.default_rng(29).normal(size=(2,150,88)).astype(np.float32)
    _,signal=certify(m,[x[0],x[1,:83]])
    state=None;values=[]
    with torch.no_grad():
        for t in range(150):
            mu,_,state=m(torch.tensor(x[:,t:t+1],dtype=torch.float64),state);values.append(mu[:,0,5].numpy())
    values=np.stack(values,1)
    np.testing.assert_allclose(signal,[values[0].max(),values[1,:83].max()],atol=1e-9,rtol=1e-8)

def test_motor_shutdown_cannot_reactivate_residual_during_contact_bounce():
    e=LandingEnv()
    try:
        e.reset(seed=610095);e.touchdown_seen=True;e.contact=False
        _,gate=e._command(np.ones(6,np.float32))
        assert gate==0 and e.last_applied[-1]==-1 and e.contact is False
    finally:e.close()

def test_physical_envelope_allows_gentle_contact_without_a_height_deadlock():
    c=CoordinatedDSLPID(DroneModel.CF2X,thrust_reserve_fraction=.001)
    d=RecoveryDescent(capacity_envelope=True);d.bind(c,lambda:np.array([.704,1.,1.,1.]))
    s=state(.04);target,velocity=d.reference(s,np.zeros(3),1/48)
    assert d.recovery_feasible and velocity[2]<0 and target[2]<.04

def test_physical_envelope_rejects_spin_and_speed_without_braking_height():
    c=CoordinatedDSLPID(DroneModel.CF2X,thrust_reserve_fraction=.001)
    d=RecoveryDescent(capacity_envelope=True);d.bind(c,lambda:np.array([.704,1.,1.,1.]))
    s=state(.10,8.);s[12]=-.22
    target,velocity=d.reference(s,np.zeros(3),1/48)
    assert not d.recovery_feasible and velocity[2]==0 and d.phase=='recovery'

def test_shutdown_records_canonical_action_in_the_next_observation():
    e=LandingEnv()
    try:
        e.reset(seed=610092);e.touchdown_seen=True;e.contact=False
        o,r,d,t,i=e.step(np.ones(6,np.float32))
        np.testing.assert_array_equal(o[-1,35:39],np.zeros(4))
        np.testing.assert_array_equal(i['executed_action'],ZERO)
        assert e.requested_on_ticks==1 and not i['gate_active']
    finally:e.close()
from experiments.round6.data import SequenceData
from experiments.round6.models import physics_observation

@pytest.mark.parametrize('arch',['mamba3','gtrxl'])
def test_decision_cannot_copy_previous_intervention_flags(arch):
    torch.set_num_threads(1);torch.manual_seed(41);m=Agent(configuration(arch)).double().eval()
    x=torch.randn(2,6,88,dtype=torch.float64);other=x.clone()
    other[...,35:40]=1.;other[...,59]=1.;other[...,77:79]=1.
    x[...,35:39]=0.;x[...,39]=-1.;x[...,59]=0.;x[...,77:79]=-1.
    with torch.no_grad():
        m.actor[-1].weight.normal_(0,.1)
        mu,val,_=m(x);other_mu,other_val,_=m(other)
    torch.testing.assert_close(mu,other_mu,rtol=0,atol=0)
    torch.testing.assert_close(val,other_val,rtol=0,atol=0)

def test_onset_sampling_preserves_the_first_activation_and_off_context():
    n=100;a=np.tile(ZERO,(n,1));a[50:]=np.array([1.,0.,0.,0.,0.,1.])
    data=dict(s=np.zeros((n,88),np.float32),a=a,records=[dict(start=0,stop=n,expert=True,noise=1.,split='train')])
    dataset=SequenceData(data,max_noise=1.)
    assert len(dataset.onsets)==1 and dataset.onsets[0][1]==50
    seen=False
    rng=np.random.default_rng(42)
    for _ in range(10):
        x,target,mask,valid=dataset.sample(rng,batch=4,length=32,burnin=16)
        for t,m in zip(target,mask):
            transitions=np.flatnonzero((t[1:,5]>0)&(t[:-1,5]<0))+1
            if any(m[i]>0 and m[i-1]>0 for i in transitions):seen=True
    assert seen

def test_offpolicy_features_cannot_reintroduce_gate_history():
    m=OffPolicy(Agent(configuration('mamba3')),'flashsac').double().eval()
    x=torch.randn(2,88,dtype=torch.float64);y=x.clone();y[:,35:40]=1.;y[:,59]=1.;y[:,77:79]=1.
    x[:,35:39]=0.;x[:,39]=-1.;x[:,59]=0.;x[:,77:79]=-1.
    a,_=m.features(x);b,_=m.features(y)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_exploration_switch_uses_boolean_mask_and_preserves_baseline():
    from experiments.round6.train_offpolicy import behavior_actions
    class Stub:
        def select(self,features,proposal):return torch.full_like(proposal,-.25)
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        proposal=torch.arange(16,device=device,dtype=torch.float32).reshape(4,4)/10
        original=proposal.clone();features=torch.zeros((4,3),device=device)
        selected=behavior_actions(Stub(),features,proposal,np.array([True,True,False,False]))
        assert torch.equal(selected[0],torch.zeros(4,device=device))
        assert torch.equal(selected[1],proposal[1])
        assert torch.equal(selected[2:],torch.full((2,4),-.25,device=device))
        assert torch.equal(proposal,original)
