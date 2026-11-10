import numpy as np
import pybullet as p
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.LegacyDSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
class RobustDSLPIDControl(DSLPIDControl):
  
    def __init__(
        self,
        drone_model: DroneModel,
        g: float = 9.8,
        *,
        fault_aware: bool = False,
        allocation_weights=(2.0, 3.0, 3.0, 0.25),
        allocation_regularization: float = 1e-4,
        disable_yaw_integral_on_fault: bool = True,
    ):
        self.fault_aware = bool(fault_aware)
        self.allocation_weights = np.asarray(
            allocation_weights, dtype=np.float64
        )
        if (self.allocation_weights.shape != (4,)
                or not np.all(np.isfinite(self.allocation_weights))
                or np.any(self.allocation_weights <= 0)):
            raise ValueError("allocation_weights must contain 4 positive finite values")
        self.allocation_regularization = float(allocation_regularization)
        if (not np.isfinite(self.allocation_regularization)
                or self.allocation_regularization < 0):
            raise ValueError("allocation_regularization must be finite and non-negative")
        self.disable_yaw_integral_on_fault = bool(disable_yaw_integral_on_fault)
        super().__init__(drone_model=drone_model, g=g)
        self.ARM = self._getURDFParameter("arm")
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4.0 * self.KF))
        self._wrench_matrix = self._build_wrench_matrix()
        self._rpm_min = self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST
        self._rpm_max = self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST
        # The parent's __init__ calls reset(); these attributes are also reset
        # on every explicit reset() call.
        self.reset()
    def reset(self):
        super().reset()
        self.motor_rpm_factors = np.ones(4, dtype=np.float64)
        self.last_diagnostics = {}
    def set_motor_effectiveness(self, rpm_factors):
        """Set external per-motor RPM multipliers, in motor order [0,1,2,3].
        Example: 10% RPM loss in motor 0 --> [0.90, 1.0, 1.0, 1.0].
        This uses the RPM convention of the project's ``aplicar_falla``.
        It is NOT a thrust-efficiency factor (which would be rpm_factors**2).
        In fault_aware=True mode, these factors are assumed available to the
        controller from an oracle or an estimator. The external simulation
        must still multiply commanded RPM by these factors once.
        """
        value = np.asarray(rpm_factors, dtype=np.float64)
        if (value.shape != (4,) or not np.all(np.isfinite(value))
                or np.any(value < 0.0) or np.any(value > 1.0)):
            raise ValueError("rpm_factors must have shape (4,) and lie in [0, 1]")
        self.motor_rpm_factors = value.copy()
    def computeControl(
        self,
        control_timestep,
        cur_pos,
        cur_quat,
        cur_vel,
        cur_ang_vel,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
    ):
        """Same call signature and three return values as DSLPIDControl."""
        dt = float(control_timestep)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("control_timestep must be positive and finite")
        self.control_counter += 1
        thrust, target_euler, pos_error = self._dslPIDPositionControl(
            dt,
            np.asarray(cur_pos, dtype=np.float64),
            np.asarray(cur_quat, dtype=np.float64),
            np.asarray(cur_vel, dtype=np.float64),
            np.asarray(target_pos, dtype=np.float64),
            np.asarray(target_rpy, dtype=np.float64),
            np.asarray(target_vel, dtype=np.float64),
        )
        rpm = self._dslPIDAttitudeControl(
            dt, thrust, cur_quat, target_euler,
            target_rpy_rates, cur_ang_vel_world=cur_ang_vel,
        )
        # Preserve the original third return-value convention.
        cur_yaw = p.getEulerFromQuaternion(cur_quat)[2]
        return rpm, pos_error, float(target_euler[2] - cur_yaw)
    def _dslPIDAttitudeControl(
        self,
        control_timestep,
        thrust,
        cur_quat,
        target_euler,
        target_rpy_rates,
        cur_ang_vel_world=None,
    ):
        """PID attitude feedback using body-frame rates, not Euler differences."""
        cur_rotation = np.array(
            p.getMatrixFromQuaternion(cur_quat), dtype=np.float64
        ).reshape(3, 3)
        target_euler = np.asarray(target_euler, dtype=np.float64)
        target_rotation = Rotation.from_euler(
            "XYZ", target_euler, degrees=False
        ).as_matrix()
        # Rotation error, with the same convention and gains as DSLPIDControl.
        rot_matrix_e = (
            target_rotation.T @ cur_rotation - cur_rotation.T @ target_rotation
        )
        rot_e = np.array(
            [rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]
        )
        cur_rpy = np.asarray(p.getEulerFromQuaternion(cur_quat))
        if cur_ang_vel_world is None:
            # Compatibility fallback for external calls to this protected method.
            # Normal calls via computeControl always use measured angular velocity.
            rpy_delta = (cur_rpy - self.last_rpy + np.pi) % (2*np.pi) - np.pi
            cur_omega_body = rpy_delta / control_timestep
        else:
            omega_world = np.asarray(cur_ang_vel_world, dtype=np.float64)
            if omega_world.shape != (3,) or not np.all(np.isfinite(omega_world)):
                raise ValueError("cur_ang_vel must contain 3 finite values")
            # PyBullet getBaseVelocity returns angular velocity in world frame.
            cur_omega_body = cur_rotation.T @ omega_world
        # Target Euler rates follow the intrinsic XYZ convention used by the
        # original target_euler/Rotation.from_euler('XYZ') implementation.
        # Map desired rates to a desired world-frame angular velocity, then
        # express it in the current body frame for comparable rate feedback.
        roll, pitch, _ = target_euler
        roll_dot, pitch_dot, yaw_dot = np.asarray(
            target_rpy_rates, dtype=np.float64
        )
        target_omega_world = np.array([
            roll_dot + np.sin(pitch) * yaw_dot,
            np.cos(roll) * pitch_dot - np.sin(roll) * np.cos(pitch) * yaw_dot,
            np.sin(roll) * pitch_dot + np.cos(roll) * np.cos(pitch) * yaw_dot,
        ])
        rate_error = cur_rotation.T @ target_omega_world - cur_omega_body
        self.last_rpy = cur_rpy.copy()
        self.integral_rpy_e -= rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500.0, 1500.0)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1.0, 1.0)
        fault_present = np.any(self.motor_rpm_factors < 1.0 - 1e-12)
        if (self.fault_aware and fault_present
                and self.disable_yaw_integral_on_fault):
            self.integral_rpy_e[2] = 0.0
        target_torques = (
            -self.P_COEFF_TOR * rot_e
            + self.D_COEFF_TOR * rate_error
            + self.I_COEFF_TOR * self.integral_rpy_e
        )
        target_torques = np.clip(target_torques, -3200.0, 3200.0)
        # Reuse the original mixer to form a nominal command. This keeps
        # tuning, sign conventions and saturation behavior consistent.
        nominal_pwm = np.clip(
            thrust + self.MIXER_MATRIX @ target_torques,
            self.MIN_PWM, self.MAX_PWM,
        )
        nominal_rpm = (
            self.PWM2RPM_SCALE * nominal_pwm + self.PWM2RPM_CONST
        )
        if self.fault_aware and fault_present:
            rpm = self._allocate_faults(nominal_rpm)
        else:
            rpm = nominal_rpm.copy()
        factors = self.motor_rpm_factors
        nominal_wrench = self._wrench_from_effective_rpm(nominal_rpm)
        achieved_wrench = self._wrench_from_effective_rpm(rpm * factors)
        self.last_diagnostics = {
            "position_target_thrust_pwm": float(thrust),
            "rotation_error": rot_e.copy(),
            "body_rate_error": rate_error.copy(),
            "body_angular_velocity": cur_omega_body.copy(),
            "nominal_rpm": nominal_rpm.copy(),
            "commanded_rpm": rpm.copy(),
            "estimated_effective_rpm": rpm * factors,
            "motor_rpm_factors": factors.copy(),
            "desired_wrench": nominal_wrench.copy(),
            "estimated_wrench": achieved_wrench.copy(),
            "rpm_at_limit": bool(np.any(
                (rpm <= self._rpm_min + 1.0)
                | (rpm >= self._rpm_max - 1.0)
            )),
        }
        return rpm
    def _build_wrench_matrix(self):
        """Maps individual motor forces [N] to [F, tau_x, tau_y, tau_z]."""
        L = self.ARM
        yaw_to_thrust = self.KM / self.KF
        if self.DRONE_MODEL == DroneModel.CF2X:
            roll_row = np.array([-1, -1, 1, 1]) * (L / np.sqrt(2.0))
            pitch_row = np.array([-1, 1, 1, -1]) * (L / np.sqrt(2.0))
        elif self.DRONE_MODEL == DroneModel.CF2P:
            roll_row = np.array([0, 1, 0, -1]) * L
            pitch_row = np.array([-1, 0, 1, 0]) * L
        else:
            raise ValueError("Only CF2X and CF2P are supported")
        return np.vstack([
            np.ones(4), roll_row, pitch_row,
            np.array([-1, 1, -1, 1]) * yaw_to_thrust,
        ])
    def _wrench_from_effective_rpm(self, effective_rpm):
        return self._wrench_matrix @ (
            self.KF * np.asarray(effective_rpm, dtype=np.float64)**2
        )
    def _allocate_faults(self, nominal_rpm):
        """Bounded, weighted least squares in normalized squared RPM.
        Objective:
            min_q || W S^-1 (A(eta)q - wrench_nominal) ||_2^2
                  + lambda ||q-q_nominal||_2^2,
            q_i = (rpm_command_i / HOVER_RPM)^2.
        S normalizes force/torque units. This prioritizes thrust and roll/pitch
        before yaw when the requested wrench becomes infeasible.
        """
        factors = self.motor_rpm_factors
        desired_wrench = self._wrench_from_effective_rpm(nominal_rpm)
        A = self._wrench_matrix * (
            self.KF * self.HOVER_RPM**2 * factors**2
        )[np.newaxis, :]
        scale = np.array([
            self.GRAVITY,
            self.GRAVITY * self.ARM / np.sqrt(2.0),
            self.GRAVITY * self.ARM / np.sqrt(2.0),
            self.GRAVITY * self.KM / self.KF,
        ], dtype=np.float64)
        weight = self.allocation_weights / scale
        A_weighted = weight[:, None] * A
        b_weighted = weight * desired_wrench
        q_nominal = (nominal_rpm / self.HOVER_RPM)**2
        lam = self.allocation_regularization
        if lam > 0:
            A_weighted = np.vstack([
                A_weighted, np.sqrt(lam) * np.eye(4),
            ])
            b_weighted = np.concatenate([
                b_weighted, np.sqrt(lam) * q_nominal,
            ])
        lo = (self._rpm_min / self.HOVER_RPM)**2
        hi = (self._rpm_max / self.HOVER_RPM)**2
        result = lsq_linear(
            A_weighted, b_weighted,
            bounds=(np.full(4, lo), np.full(4, hi)),
            method="bvls", tol=1e-10, max_iter=100,
        )
        if not np.all(np.isfinite(result.x)):
            # Safe fallback when numerical allocation fails.
            return nominal_rpm.copy()
        return np.clip(
            self.HOVER_RPM * np.sqrt(np.maximum(result.x, 0.0)),
            self._rpm_min, self._rpm_max,
        )