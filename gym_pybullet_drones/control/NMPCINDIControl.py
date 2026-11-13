"""Fault-aware nonlinear MPC followed by incremental nonlinear dynamic inversion.

Public control calls follow BaseControl/DSLPIDControl: SI state measurements,
xyzw quaternions, WORLD angular velocity, and commanded motor RPM as output.
set_motor_effectiveness takes RPM factors, not thrust factors. Fault detection
is external. set_motor_speeds optionally supplies measured PHYSICAL rotor RPM;
without feedback a first-order motor model estimates actual thrust.

Reference: Nan, Sun, Foehn, Scaramuzza, IEEE RA-L 2022,
doi:10.1109/LRA.2022.3154033, equations (1)-(12).
This is a project adaptation using CasADi/IPOPT multiple shooting and bounded
INDI allocation, rather than the paper's ACADO/qpOASES and unconstrained inverse.
It is not a reproduction of the paper's measured flight performance.

CasADi is optional for the rest of the repository; see requirements-nmpc.txt.
The default prediction horizon is 1 s (20 x 0.05 s). The IPOPT CPU budget is
not a hard real-time guarantee. Inspect last_diagnostics for solver acceptance,
fallback, saturation and solve time. A fallback is a bounded warm-start command,
not a stability or recovery guarantee. Physical parameters must match the plant,
especially motor lag and rotational drag during spinning flight.
"""
import time
from importlib.resources import files
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class NMPCINDIControl(BaseControl):
    """NMPC position/reduced-attitude control with bounded INDI compensation.

    Supported plants are CF2X and CF2P, using their URDF mass, inertia, arm,
    thrust and reaction-torque coefficients. A motor with eta=0 receives zero
    commanded RPM, including when min_rpm>0 or the optimizer fails. Yaw tracking
    is relinquished when any supplied RPM factor is below 0.999; roll/pitch
    body-rate penalties and thrust-direction tracking remain active.

    First call without RPM feedback assumes nominal hover RPM (with supplied
    effectiveness). Use set_motor_speeds for a different initial motor state.
    Reset between independent flights; optimizer construction is reused.

    The weight vectors for position, velocity and body rates have length three.
    angular_drag is a nonnegative, body-frame linear drag coefficient (N m s);
    its zero default does not invent an aerodynamic equilibrium.
    """

    def __init__(self, drone_model, g=9.8, *, horizon=20,
                 prediction_timestep=0.05, integration_substeps=4,
                 motor_time_constant=0.03, min_rpm=0., max_rpm=21700.,
                 position_weights=(80., 80., 800.),
                 velocity_weights=(1., 1., 1.),
                 angular_rate_weights=(0.5, 0.5, 0.1),
                 attitude_weight=60., yaw_weight=6.,
                 thrust_weight=0.03, input_weight=0.02,
                 terminal_weight=3., angular_drag=(0., 0., 0.),
                 indi_gain=1., indi_filter_time_constant=0.04,
                 indi_torque_limit_fraction=0.5,
                 solver_maxiter=40, solver_time_limit=0.05,
                 max_position_error=3., fault_yaw_rate_fraction=0.):
        if drone_model not in (DroneModel.CF2X, DroneModel.CF2P):
            raise ValueError("NMPCINDIControl supports CF2X and CF2P")
        gravity = _positive(g, "g")
        for name, value in (("horizon", horizon),
                            ("integration_substeps", integration_substeps),
                            ("solver_maxiter", solver_maxiter)):
            if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.horizon = int(horizon)
        self.prediction_timestep = _positive(prediction_timestep, "prediction_timestep")
        self.integration_substeps = int(integration_substeps)
        self.motor_time_constant = _positive(motor_time_constant, "motor_time_constant")
        if self.prediction_timestep / self.integration_substeps > self.motor_time_constant:
            raise ValueError("Integration substep must not exceed the motor time constant")
        self._command_min = float(min_rpm)
        self._command_max = _positive(max_rpm, "max_rpm")
        if not np.isfinite(self._command_min) or not 0 <= self._command_min < self._command_max:
            raise ValueError("Require 0 <= min_rpm < max_rpm")
        self._wp = _vector(position_weights, 3, "position_weights")
        self._wv = _vector(velocity_weights, 3, "velocity_weights")
        self._ww = _vector(angular_rate_weights, 3, "angular_rate_weights")
        self.fault_yaw_rate_fraction = float(fault_yaw_rate_fraction)
        if not 0 <= self.fault_yaw_rate_fraction <= 1:
            raise ValueError("fault_yaw_rate_fraction must lie in [0,1]")
        if np.any(self._wp <= 0) or np.any(self._wv < 0) or np.any(self._ww < 0):
            raise ValueError("Position weights must be positive; other weights nonnegative")
        self._attitude_weight = _positive(attitude_weight, "attitude_weight")
        self._yaw_weight = _positive(yaw_weight, "yaw_weight")
        self._thrust_weight = _positive(thrust_weight, "thrust_weight")
        self._input_weight = _positive(input_weight, "input_weight")
        self._terminal_weight = _positive(terminal_weight, "terminal_weight")
        self.angular_drag = _vector(angular_drag, 3, "angular_drag")
        if np.any(self.angular_drag < 0):
            raise ValueError("angular_drag must be nonnegative")
        self.indi_gain = float(indi_gain)
        if not np.isfinite(self.indi_gain) or not 0 <= self.indi_gain <= 1:
            raise ValueError("indi_gain must lie in [0,1]")
        self._filter_time_constant = _positive(indi_filter_time_constant, "indi_filter_time_constant")
        self._torque_limit_fraction = _positive(indi_torque_limit_fraction, "indi_torque_limit_fraction")
        self._maxiter = int(solver_maxiter)
        self._cpu_limit = _positive(solver_time_limit, "solver_time_limit")
        self._max_position_error = _positive(max_position_error, "max_position_error")
        try:
            import casadi as ca
        except ImportError as exc:
            raise ImportError("NMPCINDIControl requires the optional control/requirements-nmpc.txt dependency") from exc
        self._ca = ca
        super().__init__(drone_model, gravity)
        self.MASS = self.GRAVITY / gravity
        self.ARM = self._getURDFParameter("arm")
        self.J = np.diag([self._getURDFParameter(k) for k in ("ixx", "iyy", "izz")])
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self._gravity_acceleration = gravity
        self._hover_force = self.GRAVITY / 4
        self._force_geometry = self._geometry(drone_model)
        self.wrench_scale = np.array([self.GRAVITY,
                                     self.GRAVITY * self.ARM / np.sqrt(2),
                                     self.GRAVITY * self.ARM / np.sqrt(2),
                                     self.GRAVITY * self.KM / self.KF])
        self._build_solver()
        self.reset()

    def _geometry(self, model):
        # Use actual rotor offsets, including URDF rounding of the X geometry.
        tree = ET.fromstring(files("gym_pybullet_drones").joinpath("assets", model.value+".urdf").read_bytes())
        offsets = []
        for motor in range(4):
            origin = tree.find(f"./link[@name='prop{motor}_link']/inertial/origin")
            if origin is None:
                raise ValueError("The plant URDF must define all four rotor positions")
            offsets.append(_vector([float(v) for v in origin.attrib["xyz"].split()], 3, "rotor position"))
        offsets = np.asarray(offsets)
        return np.vstack((np.ones(4), offsets[:, 1], -offsets[:, 0],
                          self.KM / self.KF * np.array([-1., 1., -1., 1.])))
    def reset(self):
        super().reset()
        self.motor_rpm_factors = np.ones(4)
        self._warm_controls = None
        self._motor_thrust = None
        self._last_thrust_command = np.zeros(4)
        self._pending_motor_rpm = None
        self._filtered_omega = None
        self._filtered_thrust = None
        self._disturbance_torque = np.zeros(3)
        self.last_diagnostics = {}

    def set_motor_effectiveness(self, rpm_factors):
        factors = _vector(rpm_factors, 4, "rpm_factors")
        if np.any(factors < 0) or np.any(factors > 1):
            raise ValueError("RPM factors must lie in [0,1]")
        previous = self.motor_rpm_factors.copy()
        if np.any((previous == 0) != (factors == 0)) or np.max(np.abs(previous - factors)) > .1:
            self._warm_controls = None
            self._filtered_omega = None
            self._filtered_thrust = None
            self._disturbance_torque[:] = 0.
        if self._motor_thrust is not None:
            ratio = np.divide(factors, previous, out=np.zeros(4), where=previous > 0)
            self._motor_thrust *= ratio ** 2
            self._last_thrust_command *= ratio ** 2
        self.motor_rpm_factors = factors

    def set_motor_speeds(self, measured_rpm):
        """Supply actual rotor RPM for the next observation; eta is NOT reapplied."""
        rpm = _vector(measured_rpm, 4, "measured_rpm")
        if np.any(rpm < 0) or np.any(rpm > self._command_max):
            raise ValueError("Measured RPM must lie in [0,max_rpm]")
        self._pending_motor_rpm = rpm

    def setPIDCoefficients(self, *args, **kwargs):
        raise NotImplementedError("Configure NMPC cost weights and INDI parameters at construction")

    def _rotation_symbolic(self, q):
        ca = self._ca
        q = q / ca.sqrt(ca.dot(q, q) + 1e-24)
        x, y, z, w = (q[i] for i in range(4))
        return ca.vertcat(
            ca.horzcat(1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            ca.horzcat(2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            ca.horzcat(2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))

    def _build_solver(self):
        ca = self._ca
        x = ca.SX.sym("x", 17)
        u = ca.SX.sym("u", 4)
        cap = ca.SX.sym("capacity", 4)
        inertia = ca.DM(self.J)
        geometry = ca.DM(self._force_geometry)

        def derivative(s, command):
            q, omega = s[3:7], s[10:13]
            rotation = self._rotation_symbolic(q)
            wrench = geometry @ (self._hover_force * s[13:17])
            dq = .5 * ca.vertcat(q[3]*omega + ca.cross(q[:3], omega), -ca.dot(q[:3], omega))
            acceleration = rotation[:, 2] * wrench[0] / self.MASS - ca.DM([0., 0., self._gravity_acceleration])
            angular = ca.solve(inertia, wrench[1:4] - ca.cross(omega, inertia @ omega)
                               - ca.DM(self.angular_drag) * omega)
            return ca.vertcat(s[7:10], dq, acceleration, angular,
                              (cap * command - s[13:17]) / self.motor_time_constant)

        step = x
        h = self.prediction_timestep / self.integration_substeps
        for _ in range(self.integration_substeps):
            k1 = derivative(step, u)
            k2 = derivative(step + h*k1/2, u)
            k3 = derivative(step + h*k2/2, u)
            k4 = derivative(step + h*k3, u)
            step = step + h*(k1 + 2*k2 + 2*k3 + k4)/6
            step = ca.vertcat(step[:3], step[3:7]/ca.sqrt(ca.dot(step[3:7], step[3:7])+1e-24), step[7:])
        self._step = ca.Function("nmpc_indi_step", [x, u, cap], [step])

        n = self.horizon
        states = ca.SX.sym("states", 17, n+1)
        controls = ca.SX.sym("controls", 4, n)
        parameters = ca.SX.sym("parameters", 17 + 13*(n+1) + 10)
        references = ca.reshape(parameters[17:17+13*(n+1)], 13, n+1)
        capacity = parameters[-10:-6]
        force_reference = parameters[-6:-2]
        yaw_enabled = parameters[-2]
        rate_enabled = parameters[-1]
        rate_weights = ca.vertcat(self._ww[0], self._ww[1], self._ww[2]*rate_enabled)

        def state_cost(s, reference):
            rotation = self._rotation_symbolic(s[3:7])
            desired_rotation = self._rotation_symbolic(reference[3:7])
            rate_reference = rotation.T @ desired_rotation @ reference[10:13]
            return (ca.dot(ca.DM(self._wp), (s[:3]-reference[:3])**2)
                    + ca.dot(ca.DM(self._wv), (s[7:10]-reference[7:10])**2)
                    + ca.dot(rate_weights, (s[10:13]-rate_reference)**2)
                    + self._attitude_weight*(1-ca.dot(rotation[:, 2], desired_rotation[:, 2]))
                    + self._yaw_weight*yaw_enabled*(1-ca.dot(rotation[:, 0], desired_rotation[:, 0]))
                    + self._thrust_weight*ca.sumsqr(s[13:17]-force_reference))

        cost = 0
        constraints = [states[:, 0] - parameters[:17]]
        for k in range(n):
            cost += self.prediction_timestep * (
                state_cost(states[:, k], references[:, k])
                + self._input_weight*ca.sumsqr(capacity*controls[:, k]-force_reference))
            constraints.append(states[:, k+1] - self._step(states[:, k], controls[:, k], capacity))
        cost += self._terminal_weight*state_cost(states[:, n], references[:, n])
        decision = ca.vertcat(ca.vec(states), ca.vec(controls))
        self._state_variables = 17*(n+1)
        self._objective = ca.Function("nmpc_indi_objective", [decision, parameters], [cost])
        self._solver = ca.nlpsol("nmpc_indi", "ipopt",
                                dict(x=decision, p=parameters, f=cost, g=ca.vertcat(*constraints)),
                                {"print_time": False, "error_on_fail": False,
                                 "ipopt.print_level": 0, "ipopt.sb": "yes",
                                 "ipopt.max_iter": self._maxiter,
                                 "ipopt.max_cpu_time": self._cpu_limit,
                                 "ipopt.tol": 1e-5, "ipopt.acceptable_tol": 1e-4,
                                 "ipopt.bound_relax_factor": 0.})

    def _capacity(self):
        return self.KF * (self._command_max*self.motor_rpm_factors)**2

    def _observe_motors(self, dt):
        capacity = self._capacity()
        if self._pending_motor_rpm is not None:
            self._motor_thrust = self.KF*self._pending_motor_rpm**2
            self._pending_motor_rpm = None
        elif self._motor_thrust is None:
            rpm = np.clip(np.full(4, self.HOVER_RPM), self._command_min, self._command_max)
            self._motor_thrust = self.KF*(rpm*self.motor_rpm_factors)**2
        else:
            alpha = -np.expm1(-dt/self.motor_time_constant)
            self._motor_thrust += alpha*(self._last_thrust_command-self._motor_thrust)
        self._motor_thrust = np.clip(self._motor_thrust, 0., capacity)
        return self._motor_thrust.copy()

    def _update_indi(self, omega, thrust, dt):
        if self._filtered_omega is None:
            self._filtered_omega = omega.copy()
            self._filtered_thrust = thrust.copy()
            self._disturbance_torque[:] = 0.
            return
        alpha = -np.expm1(-dt/self._filter_time_constant)
        old = self._filtered_omega.copy()
        self._filtered_omega += alpha*(omega-self._filtered_omega)
        self._filtered_thrust += alpha*(thrust-self._filtered_thrust)
        acceleration = (self._filtered_omega-old)/dt
        known_torque = (self._force_geometry @ self._filtered_thrust)[1:]
        wf = self._filtered_omega
        disturbance = self.J@acceleration + np.cross(wf, self.J@wf) - known_torque + self.angular_drag*wf
        limit = self._torque_limit_fraction*self.wrench_scale[1:]
        self._disturbance_torque = np.clip(disturbance, -limit, limit)

    def _rollout(self, state, controls, capacity_ratio):
        predicted = np.empty((17, self.horizon+1))
        predicted[:, 0] = state
        for k in range(self.horizon):
            predicted[:, k+1] = np.asarray(self._step(predicted[:, k], controls[:, k], capacity_ratio)).ravel()
        return predicted

    def _bounds(self):
        active = self.motor_rpm_factors > 0
        low = np.where(active, (self._command_min/self._command_max)**2, 0.)
        high = active.astype(float)
        return low, high

    def _reference_parameters(self, state, position, rpy, velocity, rates):
        cap = self._capacity()
        active = cap > 0
        force = np.zeros(4)
        # Water filling supplies a bounded collective-force reference.
        remainder = self.GRAVITY
        remaining = active.copy()
        for _ in range(4):
            if not remaining.any():
                break
            share = remainder / remaining.sum()
            saturated = remaining & (cap <= share)
            if not saturated.any():
                force[remaining] = share
                break
            force[saturated] = cap[saturated]
            remainder -= float(cap[saturated].sum())
            remaining[saturated] = False
        error = position-state[:3]
        distance = np.linalg.norm(error)
        if distance > self._max_position_error:
            error *= self._max_position_error/distance
        reference = np.empty((13, self.horizon+1))
        for k in range(self.horizon+1):
            elapsed = k*self.prediction_timestep
            angles = rpy + rates*elapsed
            roll, pitch, _ = angles
            rd, pd, yd = rates
            body_rate = np.array([rd-yd*np.sin(pitch),
                                 pd*np.cos(roll)+yd*np.sin(roll)*np.cos(pitch),
                                 -pd*np.sin(roll)+yd*np.cos(roll)*np.cos(pitch)])
            reference[:, k] = np.r_[state[:3]+error+elapsed*velocity,
                                    Rotation.from_euler("xyz", angles).as_quat(), velocity, body_rate]
        yaw_enabled = float(np.all(self.motor_rpm_factors >= .999))
        params = np.r_[state, reference.ravel(order="F"), cap/self._hover_force,
                       force/self._hover_force, yaw_enabled,
                       yaw_enabled+(1-yaw_enabled)*self.fault_yaw_rate_fraction]
        seed = np.divide(force, cap, out=np.zeros(4), where=cap > 0)
        return params, seed, yaw_enabled

    def _nominal_command(self, state, params, seed, dt):
        low, high = self._bounds()
        cap = self._capacity()/self._hover_force
        if self._warm_controls is None:
            initial = np.repeat(seed[:, None], self.horizon, axis=1)
        else:
            indices = np.arange(self.horizon) + dt/self.prediction_timestep
            left = np.minimum(indices.astype(int), self.horizon-1)
            right = np.minimum(left+1, self.horizon-1)
            fraction = np.minimum(indices-left, 1.)
            initial = self._warm_controls[:, left]*(1-fraction) + self._warm_controls[:, right]*fraction
        initial = np.clip(initial, low[:, None], high[:, None])
        trajectory = self._rollout(state, initial, cap)
        guess = np.r_[trajectory.ravel(order="F"), initial.ravel(order="F")]
        seed_cost = float(self._objective(guess, params))
        stats = dict(return_status="not_run", success=False, iter_count=0)
        accepted = False
        candidate_cost = seed_cost
        selected, predicted = initial, trajectory
        start = time.perf_counter()
        try:
            solution = self._solver(x0=guess, p=params,
                                    lbx=np.r_[np.full(self._state_variables, -np.inf), np.tile(low, self.horizon)],
                                    ubx=np.r_[np.full(self._state_variables, np.inf), np.tile(high, self.horizon)],
                                    lbg=0., ubg=0.)
            stats = self._solver.stats()
            vector = np.asarray(solution["x"]).ravel()
            proposed = vector[self._state_variables:].reshape(4, self.horizon, order="F")
            if np.isfinite(proposed).all():
                proposed = np.clip(proposed, low[:, None], high[:, None])
                # Re-integrate controls instead of trusting an infeasible IPOPT iterate.
                candidate_trajectory = self._rollout(state, proposed, cap)
                decision = np.r_[candidate_trajectory.ravel(order="F"), proposed.ravel(order="F")]
                candidate_cost = float(self._objective(decision, params))
                accepted = (np.isfinite(candidate_trajectory).all() and np.isfinite(candidate_cost)
                            and candidate_cost <= seed_cost + 1e-8*(1+abs(seed_cost)))
                if accepted:
                    selected, predicted = proposed, candidate_trajectory
        except RuntimeError as exc:
            stats = dict(return_status=str(exc)[:300], success=False, iter_count=0)
        elapsed = time.perf_counter()-start
        if not np.isfinite(selected).all() or not np.isfinite(predicted).all() or not np.isfinite(seed_cost):
            raise RuntimeError("NMPC prediction is nonfinite; no valid bounded prediction is available")
        self._warm_controls = selected.copy()
        info = dict(solver_status=stats.get("return_status", "unknown"),
                    solver_success=bool(stats.get("success", False)),
                    solver_iterations=int(stats.get("iter_count", 0)),
                    solution_accepted=bool(accepted), fallback_used=not accepted,
                    solve_seconds=elapsed, seed_objective=seed_cost,
                    objective=candidate_cost if accepted else seed_cost,
                    predicted_terminal_position=predicted[:3, -1].copy())
        return selected[:, 0].copy(), info

    def _apply_indi(self, nominal, yaw_enabled):
        capacity = self._capacity()
        matrix = self._force_geometry*capacity[None, :]
        desired = matrix@nominal
        correction = self.indi_gain*self._disturbance_torque
        if not yaw_enabled:
            correction = correction.copy()
            correction[2] = 0.
        desired[1:] -= correction
        low, high = self._bounds()
        result = nominal.copy()
        success = True
        if np.any(np.abs(correction) > 1e-15):
            active = high > 0
            weights = np.array([8., 8., 8., float(yaw_enabled)])/self.wrench_scale
            weighted = weights[:, None]*matrix[:, active]
            target = weights*desired
            regularization = 1e-4
            augmented = np.vstack((weighted, np.sqrt(regularization)*np.eye(active.sum())))
            target = np.r_[target, np.sqrt(regularization)*nominal[active]]
            fit = lsq_linear(augmented, target, bounds=(low[active], high[active]),
                             method="bvls", tol=1e-9, max_iter=30)
            success = bool(fit.success and np.isfinite(fit.x).all())
            if success:
                result[active] = fit.x
        result = np.clip(result, low, high)
        return result, dict(indi_allocation_success=success, desired_wrench=desired,
                           estimated_wrench=matrix@result,
                           indi_disturbance_torque=self._disturbance_torque.copy())

    def computeControl(self, control_timestep, cur_pos, cur_quat, cur_vel, cur_ang_vel,
                       target_pos, target_rpy=np.zeros(3), target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        dt = _positive(control_timestep, "control_timestep")
        position = _vector(cur_pos, 3, "cur_pos")
        quaternion = _vector(cur_quat, 4, "cur_quat")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError("cur_quat must be nonzero")
        quaternion /= norm
        velocity = _vector(cur_vel, 3, "cur_vel")
        world_rate = _vector(cur_ang_vel, 3, "cur_ang_vel")
        goal = _vector(target_pos, 3, "target_pos")
        rpy = _vector(target_rpy, 3, "target_rpy")
        desired_velocity = _vector(target_vel, 3, "target_vel")
        rates = _vector(target_rpy_rates, 3, "target_rpy_rates")
        rotation = Rotation.from_quat(quaternion).as_matrix()
        omega = rotation.T@world_rate
        thrust = self._observe_motors(dt)
        self._update_indi(omega, thrust, dt)
        state = np.r_[position, quaternion, velocity, omega, thrust/self._hover_force]
        params, seed, yaw_enabled = self._reference_parameters(state, goal, rpy, desired_velocity, rates)
        if np.any(self.motor_rpm_factors > 0):
            nominal, solver_info = self._nominal_command(state, params, seed, dt)
            command, indi_info = self._apply_indi(nominal, yaw_enabled)
        else:
            nominal = command = np.zeros(4)
            solver_info = dict(solver_status="all_motors_unavailable", solver_success=False,
                               solution_accepted=False, fallback_used=True, solve_seconds=0.)
            indi_info = dict(indi_allocation_success=True, desired_wrench=np.zeros(4),
                             estimated_wrench=np.zeros(4), indi_disturbance_torque=np.zeros(3))
        rpm = np.clip(self._command_max*np.sqrt(command), 0., self._command_max)
        rpm[self.motor_rpm_factors == 0] = 0.
        self._last_thrust_command = self.KF*(rpm*self.motor_rpm_factors)**2
        self.control_counter += 1
        error = goal-position
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        yaw_error = float((rpy[2]-yaw+np.pi) % (2*np.pi)-np.pi)
        low, high = self._bounds()
        self.last_diagnostics = dict(solver_info, **indi_info, commanded_rpm=rpm.copy(),
                                     nominal_rpm=self._command_max*np.sqrt(nominal),
                                     motor_rpm_factors=self.motor_rpm_factors.copy(),
                                     motor_thrust_estimate=thrust, yaw_relaxed=not bool(yaw_enabled),
                                     saturated_low=command <= low+1e-7,
                                     saturated_high=command >= high-1e-7,
                                     unavailable_motors=self.motor_rpm_factors == 0,
                                     position_error=error.copy(), body_angular_velocity=omega)
        return rpm, error, yaw_error
