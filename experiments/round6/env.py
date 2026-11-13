"""Round6 environment: same plant/contact contract, new observable supervisor."""
import copy
import numpy as np
from gymnasium import spaces
from ..round5.env import LandingEnv as PreviousEnv, ZERO
from ..round5.estimator import make_estimator
from .controllers import CoordinatedDSLPID,NMPCAdapter
from .supervisor import RecoveryDescent
from gym_pybullet_drones.utils.enums import DroneModel
DIM=88
PHASES=['approach','recover','descend','brake','terminal','recovery']

class LandingEnv(PreviousEnv):
    def __init__(self,**kw):
        cfg=dict(kw.get('config') or {})
        self.manager=cfg.get('manager','coordinated')
        self.control_kind=cfg.get('control_kind','coordinated')
        if self.manager=='coordinated':cfg['readiness']=True
        kw['config']=cfg;super().__init__(**kw)
        self.native_solve_times=[];self.native_fallbacks=[]
        if self.control_kind in ['coordinated','nmpc']:
            options=dict(thrust_reserve_fraction=cfg.get('reserve',.005),emergency_thrust_ratio=1.0001,
                emergency_policy=cfg.get('emergency','damped'),rate_braking=cfg.get('rate_braking',True),
                filter_tau=cfg.get('filter_tau',.04),yaw_priority=cfg.get('yaw_priority',1.),
                yaw_acceleration_limit=cfg.get('yaw_acceleration_limit'),coordinate_thrust=cfg.get('coordinate_thrust',False),landing_only=cfg.get('landing_only',False))
            if self.control_kind=='nmpc':
                self.c=NMPCAdapter(DroneModel.CF2X,nmpc_options=cfg.get('nmpc_options'),
                    fault_braking=cfg.get('fault_braking',False),mpc_stride=cfg.get('mpc_stride',1),**options)
            else:self.c=CoordinatedDSLPID(DroneModel.CF2X,**options)
            self.core.pid_controller=self.c;self.physics_model=self.c
            self.estimator=make_estimator(self.c,self.config['estimator'],velocity_noise=0.,gyro_noise=0.)
        if self.config['estimator']=='structured':
            from .estimator import StructuredEstimator
            self.estimator=StructuredEstimator(self.c,velocity_noise=0.,gyro_noise=0.)
        if self.manager=='coordinated':
            self.descent=RecoveryDescent(**cfg.get('supervisor_options',{}))
            self.descent.bind(self.c,lambda:self.estimated_eta)
        self.observation_space=spaces.Box(-10.,10.,(self.history,DIM),dtype=np.float32)
    def reset(self,**kw):
        self.native_solve_times=[];self.native_fallbacks=[];self.trace=[]
        o,i=super().reset(**kw);self.case_seed=kw.get('seed');return o,i
    def _state(self):
        x=super()._state();d=self.descent
        phase=getattr(d,'phase','approach');onehot=np.array([phase==p for p in PHASES],float)
        return np.r_[x,onehot,min(getattr(d,'spin_time',0.),30.)/30,getattr(d,'margin',0.)].astype(np.float32)
    def teacher_action(self):
        if self.contact or self.condition=='exact':return ZERO.copy()
        teacher=copy.deepcopy(self.c)
        teacher.integral_pos_e=self._integrals[0].copy();teacher.integral_rpy_e=self._integrals[1].copy()
        if getattr(teacher,'filter_tau',0.):teacher._measure_filter=copy.deepcopy(getattr(self.c,'_filter_before',None))
        teacher.set_motor_effectiveness(self.true_eta)
        w=teacher.prepare_control(**self._arguments);rpm,_=teacher.allocate_wrench(w,commit=False)
        delta=np.clip(((rpm/teacher.HOVER_RPM)**2-self.baseq)/self.qscale,-1.,1.)
        amount=float(np.max(np.abs(delta)))
        return ZERO.copy() if amount<.002 else np.r_[delta/amount,2*amount-1,1.].astype(np.float32)
    def _command(self,action):
        if not self.touchdown_seen:return super()._command(action)
        # Once contact initiated motor shutdown, a brief bounce cannot reactivate
        # an intervention that the plant will immediately discard.
        physical_contact=self.contact;self.contact=True
        try:return super()._command(action)
        finally:self.contact=physical_contact
    def step(self,a):
        before_q=self.baseq.copy();before_scale=self.qscale;before_contact=self.touchdown_seen
        if self.fault_step is None and np.min(self.estimated_eta)<.98:self.false_alarm_ticks+=1
        requested_action=np.asarray(a,np.float32).copy()
        if before_contact:self.requested_on_ticks+=int(requested_action[5]>0)
        o,r,done,truncated,info=super().step(ZERO if before_contact else a)
        info['requested_action']=requested_action
        if before_contact:
            self.last_requested=requested_action.copy();self.requested_gates[-1]=float((requested_action[4]+1)/2)
            if done:info['requested_gate']=float(np.mean(self.requested_gates))
        command=np.clip(np.asarray(self.c.last_diagnostics.get('commanded_rpm',np.zeros(4)),float),0.,self.c._command_max)
        if before_contact:command=np.zeros(4)
        info['motor_command_rpm']=command.copy()
        info['projected_motor_delta']=np.zeros(4) if before_contact else ((command/self.c.HOVER_RPM)**2-before_q)/before_scale
        native=getattr(self.c,'nmpc',None)
        if native is not None:
            dd=native.last_diagnostics
            self.native_solve_times.append(dd.get('solve_seconds',0.));self.native_fallbacks.append(dd.get('fallback_used',False))
        info['executed_action']=self.last_applied.copy() if info['gate_active'] else ZERO.copy()
        info['reward_unscaled']=r
        if done or truncated:
            d=self.descent
            info.update(effectiveness_condition=self.condition,route_reached=self.route_reached,descent_actually_started=bool(d.started),
                recovery_phase=getattr(d,'phase','legacy'),phase_ticks=getattr(d,'phase_ticks',{}),
                recovery_events=getattr(d,'recovery_events',0),recovery_feasible=getattr(d,'recovery_feasible',True),
                vertical_braking_acceleration=getattr(d,'braking_acceleration',0.),predicted_recovery_height=getattr(d,'predicted_recovery_height',0.),
                solve_seconds_mean=float(np.mean(self.native_solve_times)) if self.native_solve_times else 0.,
                solve_seconds_p95=float(np.quantile(self.native_solve_times,.95)) if self.native_solve_times else 0.,
                nmpc_fallback_fraction=float(np.mean(self.native_fallbacks)) if self.native_fallbacks else 0.)
        return o,r,done,truncated,info
