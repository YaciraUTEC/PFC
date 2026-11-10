import numpy as np
import pytest
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.LegacyDSLPIDControl import DSLPIDControl as Legacy
from gym_pybullet_drones.utils.enums import DroneModel
from experiments.round3.common import histories

def control():return DSLPIDControl(DroneModel.CF2X)
def args(**kw):
    base=dict(control_timestep=1/48,cur_pos=np.array([0.,0.,1.]),cur_quat=np.array([0.,0.,0.,1.]),cur_vel=np.zeros(3),cur_ang_vel=np.zeros(3),target_pos=np.array([0.,0.,1.]))
    base.update(kw);return base

def test_hover_unchanged_physics():
    c=control();rpm,e,y=c.computeControl(**args())
    np.testing.assert_allclose(rpm,c.HOVER_RPM,rtol=1e-6)
    np.testing.assert_allclose(c._allocation_matrix()@((rpm/c.HOVER_RPM)**2),[c.GRAVITY,0,0,0],atol=1e-8)

@pytest.mark.parametrize('eta',[.98,.9,.8,.75])
def test_exact_effectiveness_is_squared_once(eta):
    c=control();c.set_motor_effectiveness([eta,1,1,1]);rpm,_,_=c.computeControl(**args())
    np.testing.assert_allclose(rpm*np.array([eta,1,1,1]),c.HOVER_RPM,rtol=3e-5)
    assert max(rpm)<=21700

@pytest.mark.parametrize('motor',range(4))
def test_rotor_loss_returns_finite_bounded_commands(motor):
    c=control();eff=np.ones(4);eff[motor]=0.;c.set_motor_effectiveness(eff)
    rpm,_,_=c.computeControl(**args(target_pos=np.array([2.,1.,3.])))
    assert rpm[motor]==0.;assert np.isfinite(rpm).all();assert np.all((rpm>=0)&(rpm<=21700))
    assert c.last_diagnostics['emergency_required']

def test_body_rate_feedback_opposes_roll_rate():
    c=control();w=c.prepare_control(**args(cur_ang_vel=np.array([2.,0.,0.])))
    assert w[1]<0.;assert abs(w[2])<1e-12

def test_allocator_prioritizes_roll_pitch_over_yaw():
    c=control();desired=np.array([c.GRAVITY,0,0,.1]);rpm,d=c.allocate_wrench(desired)
    assert np.linalg.norm(d['allocation_error'][1:3])<.01
    assert d['saturated_high'].any() or d['saturated_low'].any()
    assert d['allocation_error'][3]<0

def test_backcalculation_limits_integrals_and_reset():
    c=control();c.set_motor_effectiveness([.7,1,1,1])
    for i in range(100):c.computeControl(**args(target_pos=np.array([20.,-20.,20.])))
    assert np.max(np.abs(c.integral_pos_e))<=2.;assert abs(c.integral_pos_e[2])<=.15
    assert c.integral_rpy_e[2]==0.;assert c.control_counter==100
    c.reset();np.testing.assert_array_equal(c.integral_pos_e,0);np.testing.assert_array_equal(c.motor_rpm_factors,1)

def test_prepared_action_consumed_once():
    c=control();c.prepare_control(**args());c.apply_prepared()
    with pytest.raises(RuntimeError):c.apply_prepared()
    assert c.control_counter==1

@pytest.mark.parametrize('effect',[[1,1,1],[1,np.nan,1,1],[-.1,1,1,1],[1.1,1,1,1]])
def test_invalid_effectiveness(effect):
    with pytest.raises(ValueError):control().set_motor_effectiveness(effect)

def test_history_never_crosses_episode_boundary():
    d=dict(s=np.arange(24).reshape(6,4),done=np.array([False,True,False,False,True,True]))
    h=histories(d,np.array([0,1,2,4,5]),3)
    np.testing.assert_array_equal(h[2,:2],0)
    np.testing.assert_array_equal(h[2,-1],d['s'][2])
    np.testing.assert_array_equal(h[3],d['s'][2:5])
    np.testing.assert_array_equal(h[4,:2],0)

def test_legacy_preserved():
    assert Legacy.__module__.endswith('LegacyDSLPIDControl')
    c=Legacy(DroneModel.CF2X);rpm,_,_=c.computeControl(**args())
    np.testing.assert_allclose(rpm,control().HOVER_RPM,rtol=1e-4)
def test_rng_persists_across_automatic_resets():
    from experiments.round3.env import TrialEnv
    a=TrialEnv();b=TrialEnv()
    try:
        for seed in [771,None,None]:
            oa,_=a.reset(seed=seed);ob,_=b.reset(seed=seed)
            np.testing.assert_array_equal(oa,ob)
            assert (a.level,a.motor,a.bias,a.delay)==(b.level,b.motor,b.bias,b.delay)
    finally:a.close();b.close()
def test_public_pid_retuning_uses_physical_attitude_gains():
    c=control();old=c.prepare_control(**args(cur_ang_vel=np.array([1.,0.,0.])))[1]
    c.reset();c.setPIDCoefficients(d_coeff_att=c.KW*2)
    new=c.prepare_control(**args(cur_ang_vel=np.array([1.,0.,0.])))[1]
    assert new==pytest.approx(2*old)
    np.testing.assert_array_equal(c.D_COEFF_TOR,c.KW)

def test_roll_rate_feedforward_is_respected():
    c=control();w=c.prepare_control(**args(target_rpy_rates=np.array([.5,0.,0.])))
    assert w[1]==pytest.approx(.5*c.KW[0])

def test_bad_residual_wrench_rejected():
    c=control();c.prepare_control(**args())
    with pytest.raises(ValueError):c.apply_prepared(np.zeros(3))

def test_updated_gains_do_not_change_legacy_robust_gains():
    from gym_pybullet_drones.control.RobustDSLPIDControl import RobustDSLPIDControl
    c=RobustDSLPIDControl(DroneModel.CF2X)
    assert c.P_COEFF_TOR[0]>1000
    assert control().P_COEFF_TOR[0]<1
def test_student_observation_does_not_expose_true_effectiveness():
    from experiments.round3.env import TrialEnv
    e=TrialEnv()
    try:
        e.reset(seed=84210);before=e._state().copy()
        e.true_eta[:]=.4;e.level=.6
        np.testing.assert_array_equal(e._state(),before)
        for _ in range(5):
            e.teacher_action()
            e.step(np.array([0.,0.,0.,0.,-1.]))
        assert e.c.control_counter==e.t+1
    finally:e.close()

def test_validation_checkpoint_follows_completed_policy_update():
    import gymnasium as gym
    from stable_baselines3 import PPO
    from experiments.round3.run import Progress
    class Toy(gym.Env):
        observation_space=gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
        action_space=gym.spaces.Box(-1.,1.,(1,),dtype=np.float32)
        def reset(self,seed=None,options=None):return np.zeros(2,np.float32),{}
        def step(self,a):return np.zeros(2,np.float32),0.,False,False,{}
    class Capture(Progress):
        def validate(self,steps):self.validation.append((steps,self.model._n_updates))
    c=Capture('.', 'unit',0);m=PPO('MlpPolicy',Toy(),n_steps=16,batch_size=16,n_epochs=1,device='cpu')
    m.learn(64,callback=c)
    assert c.validation==[(0,0),(64,4)]
    m.get_env().close()