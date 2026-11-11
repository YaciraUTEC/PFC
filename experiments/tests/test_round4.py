import copy
import numpy as np
import pytest
import torch
from experiments.round4.landing import LandingMonitor
from experiments.round4.estimator import EffectivenessEstimator
from experiments.round4.models import Agent,ARCHS,configurations,tree_map

def state():
    s=np.zeros(20);s[2]=.014;s[6]=1
    return s
def test_waypoint_without_contact_never_success():
    m=LandingMonitor()
    for _ in range(100):ok,bad=m.update(state(),np.zeros(3),False,True)
    assert not ok
def test_requires_completed_route():
    m=LandingMonitor()
    for _ in range(100):ok,bad=m.update(state(),np.zeros(3),True,False)
    assert not ok
def test_continuous_one_second():
    m=LandingMonitor()
    for _ in range(47):assert not m.update(state(),np.zeros(3),True,True)[0]
    assert m.update(state(),np.zeros(3),True,True)[0]
def test_hold_resets_on_motion():
    m=LandingMonitor()
    for _ in range(40):m.update(state(),np.zeros(3),True,True)
    s=state();s[10]=.2;m.update(s,np.zeros(3),True,True)
    assert m.stable_ticks==0
def test_preimpact_speed_rejects_disguised_crash():
    s=state();impact=s.copy();impact[12]=-1.8
    ok,bad=LandingMonitor().update(s,np.zeros(3),True,True,impact_state=impact)
    assert bad and not ok
@pytest.mark.parametrize('field,value',[(0,.4),(7,.4),(10,.5),(12,-.6),(13,2.)])
def test_bad_contact(field,value):
    s=state();s[field]=value
    ok,bad=LandingMonitor().update(s,np.zeros(3),True,True)
    assert bad and not ok
@pytest.mark.parametrize('arch',ARCHS)
def test_stream_chunk_and_gradients(arch):
    torch.set_num_threads(1);torch.manual_seed(17)
    m=Agent(configurations(arch)[0]);torch.nn.init.normal_(m.actor[-1].weight,std=.03)
    x=torch.randn(2,7,64);whole,_,_=m(x)
    outputs=[];st=None
    for t in range(x.shape[1]):
        mu,_,st=m(x[:,t:t+1],st);outputs.append(mu)
    assert torch.allclose(whole,torch.cat(outputs,1),atol=2e-5,rtol=2e-5)
    whole.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
@pytest.mark.parametrize('arch',ARCHS)
def test_episode_reset_and_row_isolation(arch):
    torch.manual_seed(1);m=Agent(configurations(arch)[0]);torch.nn.init.normal_(m.actor[-1].weight,std=.03)
    _,_,st=m(torch.randn(2,5,64))
    st=tree_map(lambda v:v*torch.tensor([0.,1.]).reshape(2,*([1]*(v.ndim-1))),st)
    x=torch.randn(2,1,64);a,_,_=m(x,st);b,_,_=m(x[:1])
    assert torch.allclose(a[:1],b,atol=2e-5)
def test_rls_known_identifiability_without_fault_metadata():
    import inspect
    assert 'true' not in inspect.signature(EffectivenessEstimator.update).parameters
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
    c=DSLPIDControl(DroneModel.CF2X);est=EffectivenessEstimator(c,velocity_noise=0,gyro_noise=0)
    beta=np.array([1.,.64,1.,1.]);rpm=np.ones(4)*c.HOVER_RPM
    wrench=c._force_geometry@(c.KF*rpm**2*beta)
    for _ in range(100):
        a=state();a[2]=1.;b=a.copy()
        b[12]=(wrench[0]/c.MASS-9.8)/48
        b[13:16]=np.linalg.solve(c.J,wrench[1:])/48
        est.update(a,b,rpm,1/48)
    assert np.max(np.abs(est.eta-np.sqrt(beta)))<.025

@pytest.mark.parametrize('arch',ARCHS)
def test_recurrent_ppo_likelihood_reconstruction(arch):
    if not torch.cuda.is_available():pytest.skip('GPU experiment')
    from experiments.round4.train import sequences,seq_batch,advance_burnin,padded_values,tensor
    m=Agent(configurations(arch)[0]).cuda();torch.nn.init.normal_(m.actor[-1].weight,std=.03)
    rng=np.random.default_rng(2);obs=rng.normal(size=(19,2,64)).astype(np.float32)
    dones=np.zeros((19,2),bool);dones[[4,13],0]=True;dones[[8,16],1]=True
    state=None;memories=[];mus=[]
    with torch.no_grad():
        for t in range(19):
            if state is None:state=m.encoder.initial(2)
            memories.append(tree_map(lambda x:x.detach().cpu().clone(),state))
            mu,_,state=m(tensor(obs[t])[:,None],state);mus.append(mu[:,0].cpu().numpy())
            keep=tensor(~dones[t]);state=tree_map(lambda x:x*keep.reshape(2,*([1]*(x.ndim-1))),state)
        items=sequences(dones,length=6);x,mask,initial,groups=seq_batch(items,obs,memories)
        st=advance_burnin(m,items,obs,initial,groups);actual,_,_=m(x,st)
        expected=padded_values(items,np.asarray(mus),x.shape[1])
        assert torch.max(torch.abs(actual-expected)*mask[...,None])<3e-5

def test_governor_near_hover_limit_retains_horizontal_authority():
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
    c=DSLPIDControl(DroneModel.CF2X,thrust_reserve_fraction=.005,emergency_thrust_ratio=1.0001)
    c.set_motor_effectiveness([.71,1,1,1])
    c.prepare_control(1/48,np.array([0.,0,1.]),np.array([0.,0,0,1.]),np.zeros(3),np.zeros(3),np.array([1.,0,1.]))
    assert not c._prepared['emergency']
    assert c._prepared['force'][0]>0

@pytest.mark.parametrize('arch',['mamba','mamba2','mamba3'])
def test_new_ssm_matches_previous_validated_reference(arch):
    from experiments.round4.models import StreamingCore
    torch.manual_seed(11);cfg=configurations(arch)[0];core=StreamingCore(64,cfg)
    block=core.blocks[0];x=torch.randn(2,13,cfg['width'])
    actual,_=core._ssm(block,x,core.initial(2)[0]);expected=block(x)
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-4)
    ga=torch.autograd.grad(actual.square().mean(),list(block.parameters()),retain_graph=True,allow_unused=True)
    ge=torch.autograd.grad(expected.square().mean(),list(block.parameters()),allow_unused=True)
    for a,b in zip(ga,ge):
        if a is not None:torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-4)

def test_rng_replay_on_reset_sequence():
    from experiments.round4.env import LandingEnv
    a=LandingEnv();b=LandingEnv()
    try:
        a.reset(seed=27);b.reset(seed=27)
        for _ in range(3):
            x,_=a.reset();y,_=b.reset()
            assert a.level==b.level and a.motor==b.motor
            np.testing.assert_array_equal(a.core.waypoints,b.core.waypoints)
            np.testing.assert_array_equal(x,y)
    finally:a.close();b.close()

def test_student_inputs_do_not_expose_truth_or_fault_motor():
    from experiments.round4.env import LandingEnv
    e=LandingEnv(condition='estimated')
    try:
        e.reset(seed=51);before=e._state().copy()
        e.true_eta=np.array([.5,1,1,1]);e.motor=3;e.level=.5
        np.testing.assert_array_equal(e._state(),before)
    finally:e.close()

@pytest.mark.parametrize('reserve,ratio',[(-.1,1.),(.5,1.),(float('nan'),1.),(.005,.99),(.005,float('nan'))])
def test_invalid_governor_parameters(reserve,ratio):
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.utils.enums import DroneModel
    with pytest.raises(ValueError):DSLPIDControl(DroneModel.CF2X,thrust_reserve_fraction=reserve,emergency_thrust_ratio=ratio)

def test_mild_selection_checks_paired_routes_and_levels():
    from experiments.round4.safety_selection import mild_regressions
    baseline=[dict(condition=c,seed=s,level=l,stable_landing=True) for c in ['exact','estimated'] for s in [1,2] for l in [0.,.02,.05]]
    selected=copy.deepcopy(baseline)
    assert mild_regressions(baseline,selected)==[]
    selected[4]['stable_landing']=False
    assert mild_regressions(baseline,selected)==[dict(condition=selected[4]['condition'],seed=selected[4]['seed'],level=selected[4]['level'])]
    with pytest.raises(ValueError):mild_regressions(baseline,selected[:-1])
def test_mild_selection_does_not_reject_a_matching_failure():
    from experiments.round4.safety_selection import mild_regressions
    row=dict(condition='exact',seed=1,level=.02,stable_landing=False)
    assert mild_regressions([row],[row])==[]

@pytest.mark.parametrize('arch',ARCHS)
def test_actor_only_inference_equals_deterministic_full_policy(arch):
    torch.manual_seed(22);m=Agent(configurations(arch)[0]);torch.nn.init.normal_(m.actor[-1].weight,std=.03)
    x=torch.randn(3,64)
    with torch.no_grad():
        expected,_,_,_=m.act(x,deterministic=True)
        actual,_=m.predict_actions(x)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)

@pytest.mark.parametrize('arch',ARCHS)
def test_float64_reference_inference_supported(arch):
    torch.manual_seed(13);m=Agent(configurations(arch)[0]).double()
    x=torch.randn(2,64,dtype=torch.float64)
    with torch.no_grad():
        a,state=m.predict_actions(x);b,state=m.predict_actions(x,state)
    assert a.dtype==torch.float64 and torch.isfinite(b).all()


def test_historical_cache_hash_keeps_attention_value_projection():
    from types import SimpleNamespace
    from experiments.round4.campaign import actor_hash
    weights={'pi_features_extractor.attention.value.weight':torch.ones(2),
             'mlp_extractor.value_net.0.weight':torch.ones(2)}
    fake=SimpleNamespace(policy=SimpleNamespace(state_dict=lambda:weights),
                         policy_kwargs={},observation_space=SimpleNamespace(shape=(32,24)))
    original=actor_hash(fake)
    weights['mlp_extractor.value_net.0.weight']+=1
    assert actor_hash(fake)==original
    weights['pi_features_extractor.attention.value.weight']+=1
    assert actor_hash(fake)!=original
