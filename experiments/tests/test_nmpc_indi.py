"""Small CPU-only controller tests: no neural models or long flight campaigns."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("casadi")
from gym_pybullet_drones.control.NMPCINDIControl import NMPCINDIControl
from gym_pybullet_drones.utils.enums import DroneModel


@pytest.fixture(scope="module")
def shared():
    return NMPCINDIControl(DroneModel.CF2X, horizon=4, solver_time_limit=.25)


@pytest.fixture
def control(shared):
    shared.reset()
    yield shared
    shared.reset()


def arguments(**kwargs):
    result = dict(control_timestep=1/48, cur_pos=np.array([0., 0., 1.]),
                  cur_quat=np.array([0., 0., 0., 1.]), cur_vel=np.zeros(3),
                  cur_ang_vel=np.zeros(3), target_pos=np.array([0., 0., 1.]))
    result.update(kwargs)
    return result


@pytest.mark.parametrize("kwargs", [
    dict(horizon=0), dict(horizon=1.2), dict(horizon=True),
    dict(integration_substeps=0), dict(prediction_timestep=0.),
    dict(motor_time_constant=-1.), dict(min_rpm=-1.), dict(max_rpm=0.),
    dict(min_rpm=22000.), dict(g=np.nan), dict(solver_maxiter=0),
    dict(solver_time_limit=0.), dict(indi_gain=1.1),
    dict(angular_drag=[0., -1., 0.]), dict(angular_drag=[0., 0.]),
    dict(position_weights=[0., 1., 1.]), dict(velocity_weights=[-1., 1., 1.]),
    dict(indi_filter_time_constant=0.), dict(indi_torque_limit_fraction=0.),
    dict(max_position_error=0.), dict(prediction_timestep=1.)])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        NMPCINDIControl(DroneModel.CF2X, **kwargs)


def test_hover_solved_with_expected_force_and_units(control):
    rpm, error, yaw = control.computeControl(**arguments())
    np.testing.assert_allclose(rpm, control.HOVER_RPM, rtol=2e-5)
    np.testing.assert_array_equal(error, np.zeros(3))
    assert yaw == pytest.approx(0.)
    assert control.last_diagnostics["solver_success"]
    assert control.last_diagnostics["solution_accepted"]
    assert control.last_diagnostics["objective"] <= 1e-7
    np.testing.assert_allclose(control.last_diagnostics["estimated_wrench"],
                               [control.GRAVITY, 0., 0., 0.], atol=1e-7)


@pytest.mark.parametrize("offset,sign", [(.15, 1.), (-.15, -1.)])
def test_optimizer_changes_thrust_for_altitude_request(control, offset, sign):
    rpm, _, _ = control.computeControl(**arguments(target_pos=np.array([0., 0., 1.+offset])))
    assert sign*(np.mean(rpm)-control.HOVER_RPM) > 1.
    assert control.last_diagnostics["solution_accepted"]
    assert control.last_diagnostics["objective"] < control.last_diagnostics["seed_objective"]


def test_optimizer_produces_pitch_for_lateral_motion(control):
    rpm, _, _ = control.computeControl(**arguments(target_pos=np.array([.3, 0., 1.])))
    wrench = control._force_geometry @ (control.KF*rpm**2)
    assert wrench[2] > 0.
    assert control.last_diagnostics["solution_accepted"]


@pytest.mark.parametrize("motor", range(4))
def test_complete_motor_failure_stays_zero_and_other_commands_bounded(control, motor):
    factors = np.ones(4); factors[motor] = 0.
    control.set_motor_effectiveness(factors)
    rpm, _, _ = control.computeControl(**arguments())
    assert rpm[motor] == 0.
    assert np.isfinite(rpm).all() and np.all((rpm >= 0.) & (rpm <= 21700.))
    assert control.last_diagnostics["yaw_relaxed"]
    assert control.last_diagnostics["unavailable_motors"][motor]


def test_all_unavailable_skips_optimizer_and_returns_zero(control, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Optimizer must not be called")
    monkeypatch.setattr(control, "_solver", forbidden)
    control.set_motor_effectiveness(np.zeros(4))
    rpm, error, _ = control.computeControl(**arguments())
    np.testing.assert_array_equal(rpm, np.zeros(4))
    assert control.last_diagnostics["solver_status"] == "all_motors_unavailable"
    assert control.control_counter == 1


@pytest.mark.parametrize("bad", [[1, 1, 1], [1, np.nan, 1, 1], [-.1, 1, 1, 1], [1.1, 1, 1, 1]])
def test_bad_effectiveness_is_rejected_without_state_change(control, bad):
    with pytest.raises(ValueError):
        control.set_motor_effectiveness(bad)
    np.testing.assert_array_equal(control.motor_rpm_factors, np.ones(4))


def test_physical_rpm_feedback_does_not_apply_effectiveness_twice(control):
    control.set_motor_effectiveness([.8, 1., 1., 1.])
    measured = np.array([.8, 1., 1., 1.])*control.HOVER_RPM
    control.set_motor_speeds(measured)
    actual = control._observe_motors(1/48)
    np.testing.assert_allclose(actual, control.KF*measured**2)
    assert actual[0] == pytest.approx(.64*control._hover_force)
    assert control._pending_motor_rpm is None


@pytest.mark.parametrize("bad", [[0, 0, 0], [-1, 0, 0, 0], [22000, 0, 0, 0], [np.inf, 0, 0, 0]])
def test_bad_motor_feedback_is_rejected(control, bad):
    with pytest.raises(ValueError):
        control.set_motor_speeds(bad)


def test_motor_lag_and_quaternion_normalization_in_prediction(control):
    x = np.zeros(17); x[2] = 1.; x[6] = 1.; x[10:13] = [2., 3., 10.]
    capacity = control._capacity()/control._hover_force
    next_state = np.asarray(control._step(x, np.ones(4), capacity)).ravel()
    expected = capacity*(1-np.exp(-control.prediction_timestep/control.motor_time_constant))
    np.testing.assert_allclose(next_state[13:17], expected, rtol=1e-3)
    assert np.linalg.norm(next_state[3:7]) == pytest.approx(1., abs=1e-12)
    assert np.isfinite(next_state).all()


def test_world_angular_velocity_is_transformed_to_body_frame(control):
    quat = Rotation.from_euler("z", np.pi/2).as_quat()
    control.computeControl(**arguments(cur_quat=quat, cur_ang_vel=np.array([0., 2., 0.])))
    np.testing.assert_allclose(control.last_diagnostics["body_angular_velocity"],
                               [2., 0., 0.], atol=1e-12)


def test_reduced_attitude_discards_yaw_target_after_failure(control):
    control.set_motor_effectiveness([0., 1., 1., 1.])
    a = control.computeControl(**arguments(target_rpy=np.zeros(3)))[0]
    control.reset(); control.set_motor_effectiveness([0., 1., 1., 1.])
    b, _, yaw_error = control.computeControl(**arguments(target_rpy=np.array([0., 0., 1.])))
    np.testing.assert_allclose(a, b, atol=.05, rtol=1e-5)
    assert yaw_error == pytest.approx(1.)


def test_indi_compensates_known_external_roll_torque(control):
    thrust = np.ones(4)*control._hover_force
    control._update_indi(np.zeros(3), thrust, 1/48)
    disturbance = np.array([.05*control.wrench_scale[1], 0., 0.])
    omega = np.linalg.solve(control.J, disturbance)/48
    control._update_indi(omega, thrust, 1/48)
    nominal = np.full(4, (control.HOVER_RPM/control._command_max)**2)
    result, diagnostics = control._apply_indi(nominal, True)
    assert diagnostics["indi_disturbance_torque"][0] > 0.
    assert diagnostics["estimated_wrench"][1] < 0.
    assert diagnostics["estimated_wrench"][0] == pytest.approx(control.GRAVITY, rel=1e-5)
    assert np.all((result >= 0.) & (result <= 1.))


def test_modelled_drag_is_removed_from_indi_disturbance():
    c = NMPCINDIControl(DroneModel.CF2X, horizon=2, angular_drag=(.001, 0., 0.))
    omega = np.array([1., 0., 0.])
    desired = np.r_[c.GRAVITY, c.angular_drag*omega]
    thrust = np.linalg.solve(c._force_geometry, desired)
    c._update_indi(omega, thrust, 1/48)
    c._update_indi(omega, thrust, 1/48)
    np.testing.assert_allclose(c._disturbance_torque, 0., atol=1e-12)

def test_solver_failure_retains_bounded_command_and_failed_motor_zero(control, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("deliberate solver failure")
    monkeypatch.setattr(control, "_solver", broken)
    control._command_min = 1000.
    try:
        control.set_motor_effectiveness([1., 0., 1., 1.])
        rpm, _, _ = control.computeControl(**arguments())
        assert rpm[1] == 0.
        assert np.all((rpm[[0, 2, 3]] >= 1000.) & (rpm[[0, 2, 3]] <= 21700.))
        assert control.last_diagnostics["fallback_used"]
        assert not control.last_diagnostics["solver_success"]
    finally:
        control._command_min = 0.


def test_nonfinite_optimizer_result_is_not_applied(control, monkeypatch):
    class BadSolver:
        def __call__(self, **kwargs):
            return {"x": np.full(control._state_variables+4*control.horizon, np.nan)}
        def stats(self):
            return dict(success=False, return_status="Invalid_Number_Detected")
    monkeypatch.setattr(control, "_solver", BadSolver())
    rpm, _, _ = control.computeControl(**arguments())
    assert np.isfinite(rpm).all()
    assert control.last_diagnostics["fallback_used"]
    assert not control.last_diagnostics["solution_accepted"]


@pytest.mark.parametrize("kwargs", [
    dict(control_timestep=0.), dict(control_timestep=np.nan),
    dict(cur_quat=np.zeros(4)), dict(cur_quat=np.ones(3)),
    dict(cur_pos=np.array([np.nan, 0., 1.])), dict(cur_ang_vel=np.ones(4)),
    dict(target_pos=np.ones(4)), dict(target_vel=np.full(3, np.inf))])
def test_invalid_measurements_leave_controller_unadvanced(control, kwargs):
    with pytest.raises(ValueError):
        control.computeControl(**arguments(**kwargs))
    assert control.control_counter == 0
    assert control._motor_thrust is None


def test_state_interface_and_quaternion_sign_scaling_match(control):
    state = np.zeros(20); state[2] = 1.; state[6] = -5.
    a = control.computeControlFromState(1/48, state, state[:3])[0]
    control.reset()
    b = control.computeControl(**arguments())[0]
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)


def test_reset_preserves_compiled_solver_and_clears_episode_state(control):
    solver = control._solver
    control.computeControl(**arguments())
    control.set_motor_effectiveness([.8, 1., 1., 1.])
    control.reset()
    assert control._solver is solver
    assert control._warm_controls is None and control._filtered_omega is None
    assert control.control_counter == 0 and control.last_diagnostics == {}
    np.testing.assert_array_equal(control.motor_rpm_factors, np.ones(4))
    with pytest.raises(NotImplementedError):
        control.setPIDCoefficients()


def test_plus_geometry_and_default_horizon_are_supported():
    c = NMPCINDIControl(DroneModel.CF2P, horizon=2, solver_time_limit=.25)
    np.testing.assert_allclose(c._force_geometry[1], c.ARM*np.array([0, 1, 0, -1]))
    rpm, _, _ = c.computeControl(**arguments())
    np.testing.assert_allclose(rpm, c.HOVER_RPM, rtol=2e-5)
    default = NMPCINDIControl(DroneModel.CF2X)
    assert default.horizon*default.prediction_timestep == pytest.approx(1.)
    rpm, _, _ = default.computeControl(**arguments())
    assert np.isfinite(rpm).all() and np.all((rpm >= 0.) & (rpm <= 21700.))
    np.testing.assert_allclose(rpm, default.HOVER_RPM, rtol=1e-3)


def test_short_vertical_closed_loop_uses_only_small_physical_model(control):
    position = np.array([0., 0., 1.]); velocity = np.zeros(3)
    dt = 1/48
    for _ in range(8):
        rpm, _, _ = control.computeControl(**arguments(cur_pos=position, cur_vel=velocity,
                                                       target_pos=np.array([0., 0., 1.05])))
        acceleration = np.array([0., 0., control.KF*np.sum(rpm**2)/control.MASS-9.8])
        position = position + dt*velocity + .5*dt**2*acceleration
        velocity = velocity + dt*acceleration
        control.set_motor_speeds(rpm)
    assert 1. < position[2] < 1.08
    assert np.isfinite(velocity).all()


@pytest.mark.parametrize("spin", [-20., 20.])
def test_faulted_spinning_state_has_finite_optimized_command(control, spin):
    control.set_motor_effectiveness([0., 1., 1., 1.])
    rpm, _, _ = control.computeControl(**arguments(cur_ang_vel=np.array([0., 0., spin])))
    assert rpm[0] == 0. and np.isfinite(rpm).all()
    assert control.last_diagnostics["yaw_relaxed"]
    assert control.last_diagnostics["solution_accepted"]
    assert control.last_diagnostics["objective"] < control.last_diagnostics["seed_objective"]


@pytest.fixture(scope="module", params=[DroneModel.CF2X, DroneModel.CF2P])
def physical_control(request):
    return NMPCINDIControl(request.param, horizon=2)


@pytest.mark.parametrize("motor", range(4))
def test_wrench_geometry_agrees_with_one_physics_step(physical_control, motor):
    import pybullet as p
    from importlib.resources import files
    c = physical_control
    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(files("gym_pybullet_drones").joinpath("assets", c.DRONE_MODEL.value+".urdf")),
                          [0, 0, 1], flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=client)
        p.setTimeStep(1e-5, physicsClientId=client)
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.changeDynamics(body, -1, linearDamping=0, angularDamping=0, physicsClientId=client)
        force = .1
        p.applyExternalForce(body, motor, [0, 0, force], [0, 0, 0], p.LINK_FRAME, physicsClientId=client)
        p.applyExternalTorque(body, 4, [0, 0, force*c._force_geometry[3, motor]],
                              p.LINK_FRAME, physicsClientId=client)
        p.stepSimulation(physicsClientId=client)
        velocity, omega = p.getBaseVelocity(body, physicsClientId=client)
        np.testing.assert_allclose(c.J@np.asarray(omega)/1e-5,
                                   force*c._force_geometry[1:, motor], rtol=1e-5, atol=1e-10)
        assert velocity[2] == pytest.approx(force/c.MASS*1e-5, rel=1e-6)
    finally:
        p.disconnect(physicsClientId=client)


@pytest.mark.parametrize("fraction", [-.1, 1.1, np.nan, np.inf])
def test_fault_yaw_rate_fraction_rejects_invalid_values(fraction):
    with pytest.raises(ValueError):
        NMPCINDIControl(DroneModel.CF2X, fault_yaw_rate_fraction=fraction)


def test_yaw_rate_weight_can_remain_active_with_free_heading(control, monkeypatch):
    control.set_motor_effectiveness([.705, 1., 1., 1.])
    state = np.r_[0., 0., 1., 0., 0., 0., 1., np.zeros(3),
                  0., 0., 3., np.ones(4)]
    zeros = np.zeros(3)
    off, seed, heading = control._reference_parameters(state, state[:3], zeros, zeros, zeros)
    assert heading == 0. and off[-2] == 0. and off[-1] == 0.
    monkeypatch.setattr(control, "fault_yaw_rate_fraction", 1.)
    on, _, heading = control._reference_parameters(state, state[:3], zeros, zeros, zeros)
    assert heading == 0. and on[-2] == 0. and on[-1] == 1.
    np.testing.assert_array_equal(on[:-1], off[:-1])
    controls = np.repeat(seed[:, None], control.horizon, axis=1)
    trajectory = np.repeat(state[:, None], control.horizon+1, axis=1)
    decision = np.r_[trajectory.ravel(order="F"), controls.ravel(order="F")]
    assert float(control._objective(decision, on)) > float(control._objective(decision, off))
    control.set_motor_effectiveness(np.ones(4))
    healthy, _, heading = control._reference_parameters(state, state[:3], zeros, zeros, zeros)
    assert heading == 1. and healthy[-2] == 1. and healthy[-1] == 1.
