import copy,inspect
import numpy as np
import pytest
import torch
from experiments.round5.env import LandingEnv,ZERO
from experiments.round5.models import Agent,ARCHS,configurations
from experiments.round5.supervisor import DescentSupervisor
from experiments.round5.estimator import ImpulseEstimator
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

def control():return DSLPIDControl(DroneModel.CF2X,thrust_reserve_fraction=.005,emergency_thrust_ratio=1.0001)
def prepare(c,eta=.705):
    c.set_motor_effectiveness([eta,1,1,1])
    return c.prepare_control(1/48,np.array([0.,0.,1.]),np.array([0.,0.,0.,1.]),np.zeros(3),np.array([0.,0.,2.]),np.array([.5,.2,1.]))
@pytest.mark.parametrize('eta',[1.,.98,.75,.705,.5,0.])
def test_zero_q_matches_original_allocation_and_antiwindup(eta):
    c=control();w=prepare(c,eta);rpm,_=c.allocate_wrench(w,commit=False);other=copy.deepcopy(c)
    expected=c.apply_prepared(np.zeros(4))[0];actual=other.apply_prepared_q((rpm/c.HOVER_RPM)**2)[0]
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-13)
    np.testing.assert_allclose(other.integral_pos_e,c.integral_pos_e,atol=1e-12)
    np.testing.assert_allclose(other.integral_rpy_e,c.integral_rpy_e,atol=1e-12)
    assert other._prepared is None

def test_motor_space_correction_is_bounded():
    c=control();prepare(c);rpm,_,_=c.apply_prepared_q(np.array([-10.,100.,.2,.3]))
    assert np.all((rpm>=0)&(rpm<=21700.+1e-9))

def test_emergency_damped_keeps_yaw_braking():
    legacy=control();w=prepare(legacy,.69)
    damped=control();damped._emergency_policy='damped';v=prepare(damped,.69)
    assert w[3]==0 and v[3]<0

def test_invalid_emergency_policy_rejected():
    with pytest.raises(ValueError):DSLPIDControl(DroneModel.CF2X,emergency_policy='unknown')

def sample_state():
    s=np.zeros(20);s[6]=1.;s[2]=.5;return s

def test_readiness_prevents_descending_with_spin():
    supervisor=DescentSupervisor();supervisor.reset();s=sample_state();s[15]=2.;goal=np.zeros(3)
    for _ in range(20):target,v=supervisor.reference(s,goal,1/48)
    assert target[2]==s[2] and v[2]==0 and not supervisor.started
    s[15]=0
    for _ in range(3):target,v=supervisor.reference(s,goal,1/48)
    assert v[2]<0 and supervisor.started

def test_readiness_brakes_before_target_continues_down():
    supervisor=DescentSupervisor();supervisor.reset();s=sample_state();goal=np.zeros(3)
    for _ in range(20):target,v=supervisor.reference(s,goal,1/48)
    s[15]=3.;target,v=supervisor.reference(s,goal,1/48)
    assert target[2]>=s[2] and v[2]==0

@pytest.mark.parametrize('level',[0.,.02,.25,.295,.5])
def test_teacher_exact_is_zero_including_saturation(level):
    e=LandingEnv(condition='exact')
    try:
        e.reset(seed=525003,options=dict(level=level));e.fault_step=0;e._prepare()
        np.testing.assert_array_equal(e.teacher_action(),ZERO)
    finally:e.close()

def test_student_observation_contains_no_privileged_fault_fields():
    e=LandingEnv(condition='estimated')
    try:
        e.reset(seed=525004);o=e._state().copy();e.true_eta=np.array([.1,.5,1.,1.]);e.motor=3;e.level=.9
        np.testing.assert_array_equal(o,e._state())
        assert list(inspect.signature(ImpulseEstimator.update).parameters)==['self','before','after','command','dt','contact']
    finally:e.close()

def test_explicit_off_discards_nonzero_direction_and_amount():
    e=LandingEnv()
    try:
        e.reset(seed=525000);a=np.r_[np.ones(5),-1.].astype(np.float32)
        rpm,gate=e._command(a)
        assert gate==0 and e.last_applied[-1]==-1
        np.testing.assert_allclose((rpm/e.c.HOVER_RPM)**2,e.baseq,atol=1e-12)
    finally:e.close()

def test_task_deadline_is_terminal_and_observation_is_current():
    e=LandingEnv(condition='exact')
    try:
        e.reset(seed=525000);s=e.core.obs_raw[0].copy();s[2]=1.;s[10:13]=0.;s[13:16]=0.
        e.core.obs_raw=s[None];e.t=1439
        e._impact_step=lambda rpm:(s[None].copy(),False,None)
        o,r,terminated,truncated,info=e.step(ZERO)
        assert terminated and not truncated and info['outcome']=='timeout' and info['deadline_terminal']
        np.testing.assert_allclose(o[0,:3],e.reference-s[:3],atol=1e-6)
        np.testing.assert_allclose(o[0,6:15],np.eye(3).flatten(),atol=1e-6)
    finally:e.close()

@pytest.mark.parametrize('arch',ARCHS)
def test_binary_policy_streaming_and_likelihood(arch):
    torch.set_num_threads(1);torch.manual_seed(41);m=Agent(configurations(arch)[0]);x=torch.randn(2,7,80)
    torch.nn.init.normal_(m.actor[-1].weight,std=.03)
    mu,_,_=m(x);out=[];st=None
    for t in range(7):
        one,_,st=m(x[:,t:t+1],st);out.append(one)
    torch.testing.assert_close(mu,torch.cat(out,1),atol=3e-5,rtol=3e-5)
    a,lp,v,_=m.act(x[:,0]);assert set(a[:,5].tolist())<=set([-1.,1.])
    reconstructed=m.log_prob(m(x[:,:1])[0][:,0],a)
    torch.testing.assert_close(lp,reconstructed)
    assert torch.isfinite(lp).all()
    mu.square().mean().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())

@pytest.mark.parametrize('arch',ARCHS)
def test_float64_binary_policy_and_save_load(tmp_path,arch):
    m=Agent(configurations(arch)[0]).double();x=torch.randn(2,80,dtype=torch.float64)
    a,_=m.predict_actions(x);path=tmp_path/'model.pt';m.save(path);other=Agent.load(path,'cpu').double()
    b,_=other.predict_actions(x);torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_value_loss_does_not_update_actor_encoder():
    m=Agent(configurations('lstm')[0]);_,v,_=m(torch.randn(2,4,80));v.square().mean().backward()
    assert all(p.grad is None for p in m.encoder.parameters())
    assert any(p.grad is not None for p in m.critic.parameters())

def test_inactive_policy_marginalizes_unused_continuous_actions():
    m=Agent(configurations('lstm')[0]);mu=torch.zeros(3,6);mu[:,5]=-2.
    off=torch.tensor(ZERO).repeat(3,1);a=m.log_prob(mu,off)
    other=mu.clone();other[:,:5]=100.;b=m.log_prob(other,off)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert torch.isfinite(a).all()

def test_paired_branch_restores_physical_and_controller_state():
    from experiments.round5.counterfactual import choose
    import json
    e=LandingEnv(condition='estimated')
    try:
        e.reset(seed=525001);state=e.core.obs_raw.copy();q=e.c._previous_q.copy();integ=e.c.integral_pos_e.copy();obs=e._state().copy()
        rng=json.dumps(e.estimator.rng.bit_generator.state)
        labels,records,detail=choose(e,horizon=4,explore=True)
        assert detail['branched'] and len(records)==2 and all(len(r['features'])==86 for r in records)
        assert e.t==0;np.testing.assert_array_equal(e.core.obs_raw,state);np.testing.assert_array_equal(e.c._previous_q,q)
        np.testing.assert_array_equal(e.c.integral_pos_e,integ);np.testing.assert_array_equal(e._state(),obs)
        assert json.dumps(e.estimator.rng.bit_generator.state)==rng
        labels2,records2,_=choose(e,horizon=4,explore=True)
        np.testing.assert_array_equal([r['gain'] for r in records],[r['gain'] for r in records2])
    finally:e.close()

def test_branch_restores_contact_timestep_change():
    import pybullet as p
    from experiments.round5.counterfactual import snapshot,restore
    e=LandingEnv(config=dict(contact_hz=960))
    try:
        e.reset(seed=525000);saved=snapshot(e);original=e.core._env.PYB_TIMESTEP
        e.touchdown_seen=True;e.step(ZERO)
        assert e.core._env.PYB_FREQ==960
        restore(e,saved)
        assert e.core._env.PYB_FREQ==240
        assert p.getPhysicsEngineParameters(physicsClientId=e.core._env.CLIENT)['fixedTimeStep']==original
        p.removeState(saved[0],physicsClientId=e.core._env.CLIENT)
    finally:e.close()

def test_hybrid_uses_same_measurements_and_retains_original_off_boundary():
    import json
    from experiments.round5.estimator import HybridEstimator,OriginalEstimator
    c=control();a=OriginalEstimator(c);b=HybridEstimator(c);a.reset(9);b.reset(9)
    rpm=np.ones(4)*c.HOVER_RPM;beta=np.array([1.,.81,1.,1.]);wrench=c._force_geometry@(c.KF*rpm**2*beta)
    for _ in range(50):
        before=sample_state();after=before.copy();after[12]=(wrench[0]/c.MASS-9.8)/48;after[13:16]=np.linalg.solve(c.J,wrench[1:])/48
        a.update(before,after,rpm,1/48);b.update(before,after,rpm,1/48)
        np.testing.assert_array_equal(a.eta,b.eta)
        assert json.dumps(b.original.rng.bit_generator.state)==json.dumps(b.shared_rng.bit_generator.state)
    assert b.weight==0 and np.isfinite(b.uncertainty).all()

@pytest.mark.parametrize('arch',ARCHS)
def test_mixed_recurrent_ppo_reconstruction_and_padding(arch):
    if not torch.cuda.is_available():pytest.skip('GPU training verification')
    from experiments.round5.train import sequences,seq_batch,advance_burnin,padded,mixed_kl
    from experiments.round5.models import tree_map
    from experiments.round5.bc import tensor,setup
    setup(41);m=Agent(configurations(arch)[0]).cuda();torch.nn.init.normal_(m.actor[-1].weight,std=.03)
    with torch.no_grad():m.actor[-1].bias[5]=.2
    obs=np.random.default_rng(41).normal(size=(19,2,80)).astype(np.float32);dones=np.zeros((19,2),bool);dones[[4,13],0]=True;dones[[8,16],1]=True
    state=None;memories=[];actions=[];logps=[];mus=[]
    with torch.no_grad():
        for t in range(19):
            if state is None:state=m.encoder.initial(2)
            memories.append(tree_map(lambda x:x.detach().cpu().clone(),state))
            a,lp,_,state=m.act(tensor(obs[t]),state);actions.append(a.cpu().numpy());logps.append(lp.cpu().numpy())
            keep=tensor(~dones[t]);state=tree_map(lambda x:x*keep.reshape(2,*([1]*(x.ndim-1))),state)
        items=sequences(dones,length=6);x,mask,initial,groups=seq_batch(items,obs,memories)
        initial=advance_burnin(m,items,obs,initial,groups);mu,_,_=m(x,initial)
        a=padded(items,np.asarray(actions),x.shape[1],canonical_actions=True)
        oldlp=padded(items,np.asarray(logps)[...,None],x.shape[1]).squeeze(-1)
        actual=m.log_prob(mu,a)
        assert torch.max(torch.abs(actual-oldlp)*mask)<2e-4
        assert torch.isfinite(actual).all()
        assert float((mixed_kl(m,mu,mu,m.log_std)*mask).abs().max())<1e-6

def test_controller_estimator_and_policy_share_noisy_measurements():
    e=LandingEnv(condition='estimated')
    try:
        e.reset(seed=525007);captured=[];original=e.estimator.update
        def observe(before,after,command,dt,contact=False):
            captured.append((before.copy(),after.copy()));return original(before,after,command,dt,contact)
        e.estimator.update=observe
        measured_before=e.measured.copy();e.step(ZERO);before,after=captured[0]
        np.testing.assert_array_equal(before,measured_before);np.testing.assert_array_equal(after,e.measured)
        np.testing.assert_allclose(e._arguments['cur_vel'],e.measured[10:13])
        np.testing.assert_allclose(e._arguments['cur_ang_vel'],e.measured[13:16])
        np.testing.assert_allclose(e._state()[3:6],e.measured[10:13]/2,atol=1e-7)
        assert e.estimator.velocity_noise==0 and e.estimator.gyro_noise==0
        assert not np.array_equal(e.measured[10:16],e.core.obs_raw[0,10:16])
    finally:e.close()

def test_hidden_clean_velocity_does_not_leak_into_policy_observation():
    e=LandingEnv(condition='estimated')
    try:
        e.reset(seed=525006);old=e._state().copy();e.core.obs_raw[0,10:16]+=100.
        np.testing.assert_array_equal(e._state(),old)
    finally:e.close()

def test_success_return_dominates_every_failure_with_bounded_secondary_rewards():
    from experiments.round5.train import GAMMA
    ticks=np.arange(1,1441);discounted_budget=.01*(1-GAMMA**ticks)/(1-GAMMA)
    success_lower=100*GAMMA**(ticks-1)-discounted_budget
    failure_upper=-100*GAMMA**(ticks-1)+discounted_budget
    assert success_lower.min()>failure_upper.max() and success_lower.min()>16.

def test_shadow_proposal_remains_available_when_gate_is_off():
    m=Agent(configurations('lstm')[0])
    with torch.no_grad():m.actor[-1].bias[:4]=.3;m.actor[-1].bias[5]=-10.
    action,proposal,_=m.propose_actions(torch.zeros(1,80))
    torch.testing.assert_close(action,torch.tensor(ZERO)[None])
    assert proposal[0,5]==1 and torch.linalg.norm(proposal[0,:4])>0

@pytest.mark.parametrize('arch',ARCHS)
def test_used_gpu_policy_can_move_to_cpu64_without_stale_caches(arch):
    if not torch.cuda.is_available():pytest.skip('GPU cache regression')
    m=Agent(configurations(arch)[0]).cuda()
    with torch.no_grad():m.predict_actions(torch.randn(2,80,device='cuda'))
    reference=copy.deepcopy(m).cpu().double()
    with torch.no_grad():a,_=reference.predict_actions(torch.randn(2,80,dtype=torch.float64))
    assert a.dtype==torch.float64 and torch.isfinite(a).all()

def test_action_generator_is_independent_of_reward_model_random_draws():
    m=Agent(configurations('lstm')[0]);obs=torch.randn(8,80)
    a,lp,_,_=m.act(obs,generator=torch.Generator().manual_seed(217))
    torch.randn(10000)
    b,bp,_,_=m.act(obs,generator=torch.Generator().manual_seed(217))
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(lp,bp,atol=0,rtol=0)
