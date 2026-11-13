"""Small CPU-only tests for the published round-6 control snapshot."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.round6.controllers import CoordinatedDSLPID, NMPCAdapter
from experiments.round6.env import LandingEnv, ZERO
from experiments.round6.estimator import StructuredEstimator
from experiments.round6.supervisor import RecoveryDescent
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

def test_coordinated_public_control_accepts_the_original_positional_interface():
    c=CoordinatedDSLPID(DroneModel.CF2X,filter_tau=.04);other=copy.deepcopy(c)
    args=(1/48,np.array([0.,0.,1.]),np.array([0.,0.,0.,1.]),np.zeros(3),np.zeros(3),np.array([.1,.1,1.]))
    w=c.prepare_control(*args)
    v=other.prepare_control(control_timestep=args[0],cur_pos=args[1],cur_quat=args[2],cur_vel=args[3],cur_ang_vel=args[4],target_pos=args[5])
    np.testing.assert_array_equal(w,v)
    c.reset();rpm,_,_=c.computeControl(*args);assert np.isfinite(rpm).all()

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

@pytest.mark.parametrize("motor", range(4))
@pytest.mark.parametrize("effectiveness", [.705, .8, .95])
def test_structured_observer_identifies_each_single_fault_from_impulses(motor, effectiveness):
    # Independent synthetic intervals test the impulse equations, not flight stability.
    controller = CoordinatedDSLPID(DroneModel.CF2X)
    observer = StructuredEstimator(controller, velocity_noise=0., gyro_noise=0.)
    command = np.full(4, controller.HOVER_RPM)
    eta = np.ones(4)
    eta[motor] = effectiveness
    wrench = controller._force_geometry @ (controller.KF*(command*eta)**2)
    before = state()
    after = before.copy()
    dt = 1/48
    after[10:13] = np.array([0., 0., wrench[0]/controller.MASS-9.8])*dt
    after[13:16] = np.linalg.solve(controller.J, wrench[1:])*dt
    for _ in range(64):
        observer.update(before, after, command, dt)
    assert observer.candidate == motor
    assert observer.eta[motor] == pytest.approx(effectiveness, abs=.003)
    np.testing.assert_array_equal(np.delete(observer.eta, motor), np.ones(3))
    assert np.isfinite(observer.uncertainty).all()


def test_structured_observer_healthy_contact_and_low_altitude_contract():
    controller = CoordinatedDSLPID(DroneModel.CF2X)
    observer = StructuredEstimator(controller, velocity_noise=0., gyro_noise=0.)
    before = state()
    command = np.full(4, controller.HOVER_RPM)
    for _ in range(24):
        observer.update(before, before.copy(), command, 1/48)
    np.testing.assert_array_equal(observer.eta, np.ones(4))
    updates = observer.updates
    abnormal = before.copy()
    abnormal[12] = -1.
    observer.update(before, abnormal, command, 1/48, contact=True)
    observer.update(state(.04), state(.04), command, 1/48)
    assert observer.updates == updates
    np.testing.assert_array_equal(observer.eta, np.ones(4))


def test_verified_profile_reproduces_the_selected_control_stack():
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root/"experiments/round6/configs/verified_control.json").read_text())
    evidence = json.loads((root/"results/round6/phase1/phase1_conclusion.json").read_text())
    assert config == evidence["config"]
    assert evidence["selected"] == "structured_governor"
    env = LandingEnv(config=config)
    try:
        assert isinstance(env.estimator, StructuredEstimator)
        assert isinstance(env.descent, RecoveryDescent)
        observation, _ = env.reset(seed=611032, options=dict(level=.295))
        assert observation.shape == (1, 88) and np.isfinite(observation).all()
        _, _, _, _, info = env.step(ZERO.copy())
        assert not info["gate_active"]
    finally:
        env.close()


def test_nmpc_adapter_accepts_the_published_yaw_braking_option():
    pytest.importorskip("casadi")
    controller = NMPCAdapter(DroneModel.CF2X, fault_braking=True,
                             nmpc_options=dict(horizon=2, solver_time_limit=.1))
    assert controller.nmpc.fault_yaw_rate_fraction == 1.
