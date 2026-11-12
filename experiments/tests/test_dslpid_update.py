import copy,inspect
import numpy as np
import pytest
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

@pytest.mark.parametrize('eta',[.8,.705])
def test_adaptive_observer_identifies_effectiveness_from_motion(eta):
    assert list(inspect.signature(ImpulseEstimator.update).parameters)==[
        'self','before','after','command','dt','contact']
    c=control()
    est=ImpulseEstimator(c,variant='adaptive',velocity_noise=0,gyro_noise=0)
    beta=np.array([1.,eta**2,1.,1.])
    rpm=np.ones(4)*c.HOVER_RPM
    wrench=c._force_geometry@(c.KF*rpm**2*beta)
    for _ in range(100):
        before=sample_state();before[2]=1.;after=before.copy()
        after[12]=(wrench[0]/c.MASS-9.8)/48
        after[13:16]=np.linalg.solve(c.J,wrench[1:])/48
        est.update(before,after,rpm,1/48)
    np.testing.assert_allclose(est.eta,np.sqrt(beta),atol=.025)
    saved=est.eta.copy();updates=est.updates
    est.update(before,after,rpm,1/48,contact=True)
    np.testing.assert_array_equal(est.eta,saved)
    assert est.updates==updates


@pytest.mark.parametrize('reserve,ratio',[
    (-.1,1.),(.5,1.),(float('nan'),1.),(.005,.99),(.005,float('nan'))])
def test_invalid_governor_parameters(reserve,ratio):
    with pytest.raises(ValueError):
        DSLPIDControl(DroneModel.CF2X,thrust_reserve_fraction=reserve,
                      emergency_thrust_ratio=ratio)
