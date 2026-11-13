"""Compatible control adapters; fault heading release retains physical rate braking."""
import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class CoordinatedDSLPID(DSLPIDControl):
    def __init__(self,*a,rate_braking=True,filter_tau=0.,yaw_priority=1.,yaw_acceleration_limit=None,coordinate_thrust=False,landing_only=False,**kw):
        self.rate_braking=rate_braking; self.filter_tau=filter_tau; self.yaw_priority=yaw_priority
        self.yaw_acceleration_limit=yaw_acceleration_limit; self.coordinate_thrust=coordinate_thrust;self.landing_only=landing_only
        super().__init__(*a,**kw)
    def reset(self):
        super().reset(); self._measure_filter=None;self.landing_active=False;self.landing_braking_allowed=False
    def prepare_control(self,*positional,**args):
        names=['control_timestep','cur_pos','cur_quat','cur_vel','cur_ang_vel','target_pos','target_rpy','target_vel','target_rpy_rates']
        if len(positional)>len(names):raise TypeError('Too many positional arguments')
        for name,value in zip(names,positional):
            if name in args:raise TypeError('Duplicate argument '+name)
            args[name]=value
        args=dict(args)
        if self.filter_tau:
            dt=args['control_timestep']; x=np.r_[args['cur_vel'],args['cur_ang_vel']]
            self._filter_before=None if self._measure_filter is None else self._measure_filter.copy()
            if self._measure_filter is None:self._measure_filter=x.copy()
            else:self._measure_filter+=(-np.expm1(-dt/self.filter_tau))*(x-self._measure_filter)
            args['cur_vel']=self._measure_filter[:3];args['cur_ang_vel']=self._measure_filter[3:]
        desired=super().prepare_control(**args)
        if self.rate_braking and self.motor_rpm_factors.min()<.995 and (not self.landing_only or (self.landing_active and self.landing_braking_allowed)):
            d=self._prepared; omega=d['omega']
            # Release heading only, never the damping of actual yaw rate.
            rate_torque=-self.KW[2]*omega[2]
            if self.yaw_acceleration_limit is not None:
                limit=self.J[2,2]*self.yaw_acceleration_limit
                rate_torque=float(np.clip(rate_torque,-limit,limit))
            desired[3]=rate_torque+np.cross(omega,self.J@omega)[2]
            if self.coordinate_thrust:
                # Given torques, q(F)=A^-1[:,0]*F + A^-1[:,1:]*tau.
                # Intersect all four motor bounds to choose feasible collective.
                A=self._allocation_matrix()
                if np.linalg.matrix_rank(A)==4:
                    inverse=np.linalg.inv(A);b=inverse[:,1:]@desired[1:];v=inverse[:,0]
                    lo=(self._command_min/self.HOVER_RPM)**2;hi=(self._command_max/self.HOVER_RPM)**2
                    low=0.;high=np.inf
                    for aa,bb in zip(v,b):
                        if abs(aa)<1e-12:
                            if not lo<=bb<=hi:high=-1.
                        else:
                            bounds=sorted([(lo-bb)/aa,(hi-bb)/aa]);low=max(low,bounds[0]);high=min(high,bounds[1])
                    if high>=low and np.isfinite(high):desired[0]=np.clip(desired[0],low,high)
            self.integral_rpy_e[2]=0.
            self._weights[3]=self.yaw_priority
            self._prepared['base_wrench']=desired.copy()
        else:self._weights[3]=.25
        return desired

class NMPCAdapter(CoordinatedDSLPID):
    def __init__(self,model,*,nmpc_options=None,fault_braking=False,mpc_stride=1,**kw):
        from gym_pybullet_drones.control.NMPCINDIControl import NMPCINDIControl
        self.nmpc=None; self.fault_braking=fault_braking; self.mpc_stride=mpc_stride; self._mpc_ticks=0
        super().__init__(model,**kw)
        self.nmpc=NMPCINDIControl(model,**(nmpc_options or {})); self._nmpc_q=None
        if fault_braking:self.nmpc.fault_yaw_rate_fraction=1.
    def reset(self):
        super().reset(); self._mpc_ticks=0; self._nmpc_q=None
        if self.nmpc is not None:self.nmpc.reset()
    def prepare_control(self,*positional,**args):
        names=['control_timestep','cur_pos','cur_quat','cur_vel','cur_ang_vel','target_pos','target_rpy','target_vel','target_rpy_rates']
        if len(positional)>len(names):raise TypeError('Too many positional arguments')
        for name,value in zip(names,positional):
            if name in args:raise TypeError('Duplicate argument '+name)
            args[name]=value
        nominal=super().prepare_control(**args)
        self.nmpc.set_motor_effectiveness(self.motor_rpm_factors)
        # Current plant has instantaneous rotor actuation. Supply observed-model
        # rotor state from the last commanded RPM and observable eta, not true eta.
        if self._nmpc_q is not None:
            self.nmpc.set_motor_speeds(self.HOVER_RPM*np.sqrt(self._previous_q)*self.motor_rpm_factors)
        if self._mpc_ticks%self.mpc_stride==0 or self._nmpc_q is None:
            rpm,_,_=self.nmpc.computeControl(**args)
            self._nmpc_q=(rpm/self.HOVER_RPM)**2
        else:
            from scipy.spatial.transform import Rotation
            omega=Rotation.from_quat(args['cur_quat']).as_matrix().T@args['cur_ang_vel']
            thrust=self.nmpc._observe_motors(args['control_timestep'])
            self.nmpc._update_indi(omega,thrust,args['control_timestep'])
        self._mpc_ticks+=1
        desired=self._allocation_matrix()@self._nmpc_q
        self._prepared['base_wrench']=desired.copy()
        self._prepared['balanced_max']=4*self.KF*(self._command_max*self.motor_rpm_factors.min())**2
        return desired
    def allocate_wrench(self,desired_wrench,*,commit=True):
        if self._nmpc_q is None:return super().allocate_wrench(desired_wrench,commit=commit)
        q=self._nmpc_q.copy(); A=self._allocation_matrix(); achieved=A@q
        if np.linalg.norm(np.asarray(desired_wrench)-achieved)>1e-10:return super().allocate_wrench(desired_wrench,commit=commit)
        hi=(self._command_max/self.HOVER_RPM)**2
        d=dict(desired_wrench=np.asarray(desired_wrench).copy(),estimated_wrench=achieved,
            allocation_error=(achieved-desired_wrench)/self.wrench_scale,
            allocation_error_norm=float(np.linalg.norm((achieved-desired_wrench)/self.wrench_scale)),
            motor_rpm_factors=self.motor_rpm_factors.copy(),solver_ok=True,
            saturated_low=q<=1e-7,saturated_high=q>=hi-1e-7,unavailable_motors=self.motor_rpm_factors<=1e-8)
        if commit:self._previous_q=q.copy()
        return self.HOVER_RPM*np.sqrt(q),d
