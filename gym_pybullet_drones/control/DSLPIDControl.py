"""Fault-aware evolution of DSLPIDControl in SI units.
The original implementation is retained in LegacyDSLPIDControl for reproduction.
A bounded allocation follows the unsaturated force/torque demand. No motor limit
is increased: commands remain in [0,21700] RPM in the Crazyflie simulation.
"""
import numpy as np
import pybullet as p
from scipy.optimize import lsq_linear
from .LegacyDSLPIDControl import DSLPIDControl as OriginalDSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel


def _vee(matrix):
    return np.array([matrix[2,1],matrix[0,2],matrix[1,0]])


class DSLPIDControl(OriginalDSLPIDControl):
    def __init__(self,drone_model,g=9.8,*,fault_aware=True,max_rpm=21700.,
                 min_rpm=0.,tilt_limit_deg=25.,allocation_weights=(4.,8.,8.,.25),
                 allocation_regularization=1e-6,antiwindup_gain=2.,
                 attitude_frequency=(25.,25.,12.),attitude_damping=1.,
                 thrust_reserve_fraction=.05,emergency_thrust_ratio=1.02,
                 emergency_policy="legacy"):
        self._modern_ready=False
        super().__init__(drone_model,g)
        self._modern_fault_aware=bool(fault_aware)
        self.ARM=self._getURDFParameter('arm')
        self.MASS=self.GRAVITY/float(g)
        self.J=np.diag([self._getURDFParameter(k) for k in ['ixx','iyy','izz']])
        self._command_min=float(min_rpm);self._command_max=float(max_rpm)
        if not 0<=self._command_min<self._command_max:raise ValueError('Invalid RPM bounds')
        self._tilt_limit=np.deg2rad(tilt_limit_deg)
        self._weights=np.asarray(allocation_weights,float)
        if self._weights.shape!=(4,) or np.any(self._weights<=0):raise ValueError('Invalid allocation weights')
        if not np.isfinite(self._weights).all():raise ValueError('Invalid allocation weights')
        self._allocation_regularization=float(allocation_regularization)
        if not np.isfinite(self._allocation_regularization) or self._allocation_regularization<0:raise ValueError('Invalid regularization')
        self._thrust_reserve=float(thrust_reserve_fraction)
        self._emergency_ratio=float(emergency_thrust_ratio)
        if emergency_policy not in ["legacy","damped","balanced"]:raise ValueError("Invalid emergency policy")
        self._emergency_policy=emergency_policy
        if not np.isfinite(self._thrust_reserve) or not 0<=self._thrust_reserve<.5:raise ValueError('Invalid thrust reserve')
        if not np.isfinite(self._emergency_ratio) or not 1<=self._emergency_ratio<=1.2:raise ValueError('Invalid emergency ratio')
        self._aw_gain=float(antiwindup_gain)
        if not np.isfinite(self._aw_gain) or self._aw_gain<0:raise ValueError('Invalid antiwindup gain')
        if not 0<tilt_limit_deg<90:raise ValueError('Tilt limit must lie between0 and90 degrees')
        frequency=np.asarray(attitude_frequency,float)
        if frequency.shape!=(3,) or not np.isfinite(frequency).all() or np.any(frequency<=0):raise ValueError('Invalid attitude frequencies')
        if not np.isfinite(attitude_damping) or attitude_damping<=0:raise ValueError('Invalid damping')
        self.KR=np.diag(self.J)*frequency**2
        self.KW=np.diag(self.J)*2*float(attitude_damping)*frequency
        self.KI_R=np.diag(self.J)*np.array([1.,1.,.5])
        self.HOVER_RPM=np.sqrt(self.GRAVITY/(4*self.KF))
        a=self.ARM/np.sqrt(2)
        if drone_model==DroneModel.CF2X:
            roll=np.array([-1,-1,1,1])*a;pitch=np.array([-1,1,1,-1])*a
        else:
            roll=np.array([0,1,0,-1])*self.ARM;pitch=np.array([-1,0,1,0])*self.ARM
        self._force_geometry=np.vstack([np.ones(4),roll,pitch,
                     np.array([-1,1,-1,1])*self.KM/self.KF])
        self.wrench_scale=np.array([self.GRAVITY,self.GRAVITY*self.ARM/np.sqrt(2),
                    self.GRAVITY*self.ARM/np.sqrt(2),self.GRAVITY*self.KM/self.KF])
        self._modern_ready=True
        self.reset()

    # Keep the public PID coefficient interface meaningful in SI units.
    @property
    def KR(self):return self.P_COEFF_TOR
    @KR.setter
    def KR(self,value):self.P_COEFF_TOR=np.asarray(value,float).copy()
    @property
    def KW(self):return self.D_COEFF_TOR
    @KW.setter
    def KW(self,value):self.D_COEFF_TOR=np.asarray(value,float).copy()
    @property
    def KI_R(self):return self.I_COEFF_TOR
    @KI_R.setter
    def KI_R(self,value):self.I_COEFF_TOR=np.asarray(value,float).copy()

    def setPIDCoefficients(self,p_coeff_pos=None,i_coeff_pos=None,d_coeff_pos=None,
                           p_coeff_att=None,i_coeff_att=None,d_coeff_att=None):
        """Retune position in N and attitude in N m (not legacy PWM gains).

        Attitude P/D/I multiply rotation error, body-rate error and its integral.
        Constructor attitude_frequency/damping is the simpler tuning interface.
        """
        values=[p_coeff_pos,i_coeff_pos,d_coeff_pos,p_coeff_att,i_coeff_att,d_coeff_att]
        clean=[]
        for value in values:
            if value is None:clean.append(None);continue
            value=np.asarray(value,float)
            if value.shape!=(3,) or not np.isfinite(value).all() or np.any(value<0):
                raise ValueError('PID gains must be three non-negative finite SI values')
            clean.append(value.copy())
        super().setPIDCoefficients(*clean)

    def reset(self):
        super().reset()
        self.motor_rpm_factors=np.ones(4)
        self.last_diagnostics={}
        self._previous_q=np.ones(4)
        self._prepared=None
        self._residual_wrench=np.zeros(4)

    def set_motor_effectiveness(self,rpm_factors):
        """Set estimated RPM factors eta; allocation uses eta squared.

        For10% RPM loss use[0.9,1,1,1]. The physical plant must still apply
        eta to the commanded RPM once. This method does not estimate eta.
        """
        factors=np.asarray(rpm_factors,float)
        if factors.shape!=(4,) or not np.isfinite(factors).all() or np.any((factors<0)|(factors>1)):
            raise ValueError('Motor effectiveness must be four RPM factors in [0,1]')
        self.motor_rpm_factors=factors.copy()

    def set_residual_wrench(self,delta):
        delta=np.asarray(delta,float)
        if delta.shape!=(4,) or not np.isfinite(delta).all():raise ValueError('Invalid residual wrench')
        self._residual_wrench=delta.copy()

    def _allocation_matrix(self):
        factors=self.motor_rpm_factors if self._modern_fault_aware else np.ones(4)
        return self._force_geometry*(self.KF*self.HOVER_RPM**2*factors**2)[None,:]

    def allocate_wrench(self,desired_wrench,*,commit=True):
        desired=np.asarray(desired_wrench,float)
        if desired.shape!=(4,) or not np.isfinite(desired).all():raise ValueError('Invalid wrench')
        A=self._allocation_matrix()
        active=self.motor_rpm_factors>1e-8 if self._modern_fault_aware else np.ones(4,bool)
        weighted=self._weights/self.wrench_scale
        matrix=weighted[:,None]*A[:,active];target=weighted*desired
        lo=(self._command_min/self.HOVER_RPM)**2;hi=(self._command_max/self.HOVER_RPM)**2
        q=np.zeros(4);solver_ok=True
        if np.any(active):
            lam=self._allocation_regularization
            if lam:
                matrix=np.vstack([matrix,np.sqrt(lam)*np.eye(int(active.sum()))])
                target=np.r_[target,np.sqrt(lam)*self._previous_q[active]]
            # Interior feasible requests have a unique exact solution. Avoid
            # solving a constrained problem unnecessarily in every control tick.
            candidate=np.linalg.lstsq(matrix,target,rcond=None)[0]
            if np.all(candidate>=lo) and np.all(candidate<=hi):solution=candidate
            else:
                fit=lsq_linear(matrix,target,bounds=(np.full(active.sum(),lo),np.full(active.sum(),hi)),
                               method='bvls',tol=1e-9,max_iter=40)
                solution=fit.x;solver_ok=bool(fit.success and np.isfinite(solution).all())
            if not solver_ok:solution=np.clip(self._previous_q[active],lo,hi)
            q[active]=np.clip(solution,lo,hi)
        achieved=A@q
        normalized_error=(achieved-desired)/self.wrench_scale
        diagnostics=dict(desired_wrench=desired.copy(),estimated_wrench=achieved.copy(),
            allocation_error=normalized_error,allocation_error_norm=float(np.linalg.norm(normalized_error)),
            motor_rpm_factors=self.motor_rpm_factors.copy(),solver_ok=solver_ok,
            saturated_low=(q<=lo+1e-7),saturated_high=(q>=hi-1e-7),
            unavailable_motors=~active)
        if commit:self._previous_q=q.copy()
        return self.HOVER_RPM*np.sqrt(q),diagnostics

    def prepare_control(self,control_timestep,cur_pos,cur_quat,cur_vel,cur_ang_vel,
                        target_pos,target_rpy=np.zeros(3),target_vel=np.zeros(3),
                        target_rpy_rates=np.zeros(3)):
        dt=float(control_timestep)
        if not np.isfinite(dt) or dt<=0:raise ValueError('control_timestep must be positive')
        pos=np.asarray(cur_pos,float);vel=np.asarray(cur_vel,float)
        R=np.asarray(p.getMatrixFromQuaternion(cur_quat)).reshape(3,3)
        omega=R.T@np.asarray(cur_ang_vel,float)
        ep=np.asarray(target_pos)-pos;ev=np.asarray(target_vel)-vel
        self.integral_pos_e=np.clip(self.integral_pos_e+ep*dt,-2.,2.)
        self.integral_pos_e[2]=np.clip(self.integral_pos_e[2],-.15,.15)
        raw_force=self.P_COEFF_FOR*ep+self.D_COEFF_FOR*ev+self.I_COEFF_FOR*self.integral_pos_e
        raw_force=raw_force+np.array([0.,0.,self.GRAVITY])
        factors=self.motor_rpm_factors if self._modern_fault_aware else np.ones(4)
        balanced_max=4*self.KF*(self._command_max*factors.min())**2
        total_max=self.KF*self._command_max**2*np.sum(factors**2)
        emergency=bool(balanced_max<self.GRAVITY*self._emergency_ratio)
        force=raw_force.copy()
        force[2]=max(force[2],.15*self.GRAVITY)
        tilt=self._tilt_limit
        if not emergency or self._emergency_policy=="balanced":
            available=max(self.GRAVITY,balanced_max*(1-self._thrust_reserve))
            tilt=min(tilt,np.arccos(np.clip(self.GRAVITY/available,0,1)))
        else:available=total_max*.9
        xy=np.linalg.norm(force[:2]);maxxy=max(0.,force[2]*np.tan(tilt))
        if xy>maxxy:force[:2]*=maxxy/xy
        norm=np.linalg.norm(force)
        if norm>available and norm>0:force*=available/norm
        zdesired=force/max(np.linalg.norm(force),1e-9)
        yaw=float(np.asarray(target_rpy)[2]);heading=np.array([np.cos(yaw),np.sin(yaw),0.])
        yd=np.cross(zdesired,heading);yd/=max(np.linalg.norm(yd),1e-9)
        Rd=np.column_stack([np.cross(yd,zdesired),yd,zdesired])
        error=.5*_vee(Rd.T@R-R.T@Rd)
        if emergency and self._emergency_policy!="balanced":
            # Best-effort reduced-attitude request, not a validated spinning
            # equilibrium / emergency landing controller.
            error=R.T@np.cross(zdesired,R[:,2]);error[2]=0.
        self.integral_rpy_e=np.clip(self.integral_rpy_e+error*dt,-.5,.5)
        if np.any(factors<.999):self.integral_rpy_e[2]=0.
        rates=np.asarray(target_rpy_rates,float)
        if rates.shape!=(3,) or not np.isfinite(rates).all():raise ValueError('Invalid target Euler rates')
        desired_omega=np.zeros(3)
        if np.any(rates):
            from scipy.spatial.transform import Rotation
            roll,pitch,_=Rotation.from_matrix(Rd).as_euler('xyz')
            rd,pd,ydot=rates
            omega_desired_body=np.array([rd-ydot*np.sin(pitch),
                pd*np.cos(roll)+ydot*np.sin(roll)*np.cos(pitch),
                -pd*np.sin(roll)+ydot*np.cos(roll)*np.cos(pitch)])
            desired_omega=Rd@omega_desired_body
        rate_error=omega-R.T@desired_omega
        torque=-self.KR*error-self.KW*rate_error-self.KI_R*self.integral_rpy_e+np.cross(omega,self.J@omega)
        if emergency and self._emergency_policy=="legacy":torque[2]=0.
        desired=np.r_[max(0.,float(force@R[:,2])),torque]
        self._prepared=dict(dt=dt,R=R,raw_force=raw_force,force=force,position_error=ep,
            rotation_error=error,omega=omega,base_wrench=desired.copy(),emergency=emergency,
            balanced_max=balanced_max,yaw_error=float(yaw-p.getEulerFromQuaternion(cur_quat)[2]))
        self.control_counter+=1
        return desired.copy()

    def apply_prepared(self,residual_wrench=None):
        if self._prepared is None:raise RuntimeError('Call prepare_control once before applying action')
        data=self._prepared
        delta=self._residual_wrench if residual_wrench is None else np.asarray(residual_wrench,float)
        if delta.shape!=(4,) or not np.isfinite(delta).all():raise ValueError('Residual wrench must contain four finite SI values')
        rpm,allocation=self.allocate_wrench(data['base_wrench']+delta)
        return self._finish_prepared(rpm,allocation,delta)

    def apply_prepared_q(self,q):
        """Commit feasible squared normalized motor speeds, retaining anti-windup.

        Zero intervention must pass the exact q returned by allocate_wrench.
        The desired residual is measured relative to that allocated base, not
        the infeasible raw wrench; this avoids projection-induced teacher labels.
        """
        if self._prepared is None:raise RuntimeError('Call prepare_control first')
        q=np.asarray(q,float)
        if q.shape!=(4,) or not np.isfinite(q).all():raise ValueError('Invalid motor q')
        lo=(self._command_min/self.HOVER_RPM)**2;hi=(self._command_max/self.HOVER_RPM)**2
        q=np.clip(q,lo,hi);q[self.motor_rpm_factors<=1e-8]=0.
        base_rpm,_=self.allocate_wrench(self._prepared['base_wrench'],commit=False)
        baseq=(base_rpm/self.HOVER_RPM)**2
        A=self._allocation_matrix();delta=A@(q-baseq)
        desired=self._prepared['base_wrench']+delta;achieved=A@q
        allocation=dict(desired_wrench=desired.copy(),estimated_wrench=achieved.copy(),
            allocation_error=(achieved-desired)/self.wrench_scale,
            allocation_error_norm=float(np.linalg.norm((achieved-desired)/self.wrench_scale)),
            motor_rpm_factors=self.motor_rpm_factors.copy(),solver_ok=True,
            saturated_low=q<=lo+1e-7,saturated_high=q>=hi-1e-7,
            unavailable_motors=self.motor_rpm_factors<=1e-8)
        self._previous_q=q.copy()
        return self._finish_prepared(self.HOVER_RPM*np.sqrt(q),allocation,delta)

    def _finish_prepared(self,rpm,allocation,delta):
        data=self._prepared
        dt=data['dt'];achieved=allocation['estimated_wrench']
        # Back-calculation in force units, including the reference governor.
        force_error=data['R'][:,2]*achieved[0]-data['raw_force']
        self.integral_pos_e+=self._aw_gain*dt*force_error/np.maximum(self.I_COEFF_FOR,.01)
        self.integral_pos_e=np.clip(self.integral_pos_e,-2.,2.)
        self.integral_pos_e[2]=np.clip(self.integral_pos_e[2],-.15,.15)
        tau_error=achieved[1:]-(data['base_wrench'][1:]+delta[1:])
        self.integral_rpy_e-=self._aw_gain*dt*tau_error/np.maximum(self.KI_R,1e-6)
        self.integral_rpy_e=np.clip(self.integral_rpy_e,-.5,.5)
        if np.any(self.motor_rpm_factors<.999):self.integral_rpy_e[2]=0.
        self.last_diagnostics=dict(allocation,base_wrench=data['base_wrench'],residual_wrench=delta.copy(),
            requested_force_world=data['raw_force'],governed_force_world=data['force'],
            position_error=data['position_error'],body_angular_velocity=data['omega'],
            rotation_error=data['rotation_error'],emergency_required=data['emergency'],
            balanced_max_force=data['balanced_max'],commanded_rpm=rpm.copy(),
            rpm_at_limit=bool(allocation['saturated_high'].any()))
        self._prepared=None
        return rpm,data['position_error'].copy(),data['yaw_error']

    def computeControl(self,control_timestep,cur_pos,cur_quat,cur_vel,cur_ang_vel,
                       target_pos,target_rpy=np.zeros(3),target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        self.prepare_control(control_timestep,cur_pos,cur_quat,cur_vel,cur_ang_vel,
                             target_pos,target_rpy,target_vel,target_rpy_rates)
        return self.apply_prepared()