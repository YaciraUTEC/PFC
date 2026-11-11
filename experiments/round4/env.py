"""Round4: common route/descent, strict contact metric, observable estimator.
Old rounds are preserved. The same landing manager is used by every controller.
"""
from collections import deque
import copy
import numpy as np
import pybullet as p
from gymnasium import spaces
from ..round3.env import TrialEnv
from .estimator import EffectivenessEstimator,rotation
from .landing import LandingMonitor

LEVELS=[0.,.02,.05,.10,.15,.20,.25,.28,.29]
STRESS=[.295,.30,.35,.50]
ZERO=np.array([0.,0.,0.,0.,-1.],np.float32)
DIM=64
DT=1/48

class LandingEnv(TrialEnv):
    def __init__(self,history=1,controller='updated',condition='estimated',fixed_level=None,
                 landing_profile='descent',guard=True,sensor_noise=True):
        old_governor=controller=='updated_round3'
        if old_governor:controller='updated'
        super().__init__(history=1,controller=controller,condition=condition,fixed_level=fixed_level)
        if controller=='updated' and not old_governor:
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.utils.enums import DroneModel
            self.c=DSLPIDControl(DroneModel.CF2X,thrust_reserve_fraction=.005,emergency_thrust_ratio=1.0001)
            self.core.pid_controller=self.c
        self.history=history;self.frames=deque(maxlen=history)
        self.observation_space=spaces.Box(-10.,10.,(history,DIM),dtype=np.float32)
        self.landing_profile=landing_profile;self.guard=guard;self.sensor_noise=sensor_noise
        # Geometry/inertia is a known calibrated nominal model, not fault truth.
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel
        self.physics_model=self.c if controller=='updated' else DSLPIDControl(DroneModel.CF2X)
        self.estimator=EffectivenessEstimator(self.physics_model,
            velocity_noise=.002 if sensor_noise else 0.,gyro_noise=.002 if sensor_noise else 0.)
    def reset(self,seed=None,options=None):
        import gymnasium as gym
        gym.Env.reset(self,seed=seed)
        self.core.reset(seed=seed)
        if seed is not None or not hasattr(self,'rng'):
            self.rng=np.random.default_rng(None if seed is None else seed+99173)
        self.level=float((options or {}).get('level',self.fixed_level if self.fixed_level is not None else self.rng.choice(LEVELS)))
        self.motor=int((options or {}).get('motor',self.rng.integers(4)))
        self.bias=float(self.rng.uniform(.45,.9));self.delay=int(self.rng.integers(2,7))
        self.t=0;self.fault_step=None;self.true_eta=np.ones(4);self.estimated_eta=np.ones(4)
        self.estimator.reset(int(self.rng.integers(1,2**30)) if seed is None else seed)
        self.previous_action=ZERO.copy();self.tracking=[];self.effort=[];self.gates=[];self.saturation=[]
        self.max_tilt=0.;self.route_reached=False;self.descent_start=None;self.contact=False
        self.monitor=LandingMonitor();self.estimation_errors=[];self.false_alarm_ticks=0
        self.guard_off_ticks=0;self.requested_gates=[];self.gate_variations=[];self.previous_gate=0.
        self.touchdown_seen=False;self.reference=None;self.legacy_waypoint_success=False
        self.failed_motor_errors=[];self.gates_prefault=[];self.gates_postfault=[];self.estimate_recovery_tick=None
        self.frames.clear()
        for _ in range(self.history-1):self.frames.append(np.zeros(DIM,np.float32))
        self._prepare()
        return self._obs(),{}
    def _contacts(self):
        e=self.core._env
        return any(c[8]<=.001 for c in p.getContactPoints(int(e.DRONE_IDS[0]),e.PLANE_ID,physicsClientId=e.CLIENT))
    def _prepare(self):
        e=self.core;s=e.obs_raw[0]
        if self.level>0 and self.fault_step is None and e.wp_idx>=e.wp_medio_crucero:
            self.fault_step=self.t
        if self.fault_step is not None:self.true_eta[self.motor]=1-self.level
        if self.condition=='exact':self.estimated_eta=self.true_eta.copy()
        elif self.condition=='biased':
            self.estimated_eta=np.ones(4)
            if self.fault_step is not None and self.t-self.fault_step>=self.delay:
                self.estimated_eta[self.motor]=1-self.level*self.bias
        else:self.estimated_eta=self.estimator.eta.copy()
        final=e.wp_idx>=len(e.waypoints)-1
        target=e.waypoints[min(e.wp_idx,len(e.waypoints)-1)].copy()
        velocity=np.zeros(3)
        if final:
            self.route_reached=True
            if self.descent_start is None:self.descent_start=(self.t,float(s[2]))
            if self.landing_profile=='descent':
                # Same continuous descent reference for baseline and all learned policies.
                elapsed=(self.t-self.descent_start[0])*DT
                target[2]=max(-.025,self.descent_start[1]-.18*elapsed)
                velocity[2]=-.18 if target[2]>-.025 else 0.
        self.reference=target;self.target_velocity=velocity
        self._arguments=dict(control_timestep=DT,cur_pos=s[:3],cur_quat=s[3:7],cur_vel=s[10:13],
            cur_ang_vel=s[13:16],target_pos=target,target_vel=velocity)
        if self.kind=='updated':
            self.c.set_motor_effectiveness(self.estimated_eta)
            self._integrals=(self.c.integral_pos_e.copy(),self.c.integral_rpy_e.copy())
            self.base=self.c.prepare_control(**self._arguments)
            _,self.predicted=self.c.allocate_wrench(self.base,commit=False)
        else:self.base=np.zeros(4);self.predicted={}
    def _state(self):
        # Preserve the original 48 input meanings for historical round3 policies.
        e=self.core;idx=min(e.wp_idx,len(e.waypoints)-1);old=e.waypoints[idx].copy()
        e.waypoints[idx]=self.reference
        state=super()._state()
        e.waypoints[idx]=old
        extra=np.r_[self.estimator.uncertainty/.1,self.estimator.innovation,
            float(self.route_reached),float(self.contact),self.monitor.stable_ticks/48,
            self.previous_gate,self.target_velocity, min(self.estimator.updates,100)/100]
        assert extra.shape==(16,)
        return np.clip(np.r_[state,extra],-10,10).astype(np.float32)
    def _obs(self):
        self.frames.append(self._state());return np.asarray(self.frames,np.float32)
    def need(self):
        s=self.core.obs_raw[0];R=rotation(s);omega=R.T@s[13:16]
        # Applied to estimated/exact identically. It uses no simulated fault label.
        error=np.linalg.norm(self.reference-s[:3])
        rate=np.linalg.norm(omega[:2]);tilt=np.max(np.abs(s[7:9]))
        uncertainty=float(np.max(self.estimator.uncertainty)) if self.condition!='exact' else 0.
        allocation=float(np.linalg.norm(self.predicted.get('allocation_error',np.zeros(4))[:3]))
        healthy=error<.22 and rate<.5 and tilt<.10 and allocation<.015
        if self.route_reached:
            healthy=np.linalg.norm(s[:2]-self.core.punto_B[:2])<.12 and rate<.4 and tilt<.08 and abs(s[12]-.18*-1)<.16 and allocation<.015
        return 0. if healthy and (uncertainty<.05 or self.condition=='exact') else 1.
    def teacher_action(self):
        if self.kind!='updated' or self.contact:return ZERO.copy()
        teacher=copy.deepcopy(self.c)
        teacher.integral_pos_e=self._integrals[0].copy();teacher.integral_rpy_e=self._integrals[1].copy()
        teacher.set_motor_effectiveness(self.true_eta)
        w=teacher.prepare_control(**self._arguments)
        rpm,_=teacher.allocate_wrench(w,commit=False)
        q=(rpm/self.c.HOVER_RPM)**2
        equivalent=self.c._allocation_matrix()@q
        normalized=np.clip((equivalent-self.base)/(self.c.wrench_scale*np.array([.35,.5,.5,.5])),-.76,.76)
        amplitude=float(np.max(np.abs(normalized)))
        if amplitude<.005:return ZERO.copy()
        gate=min(amplitude/.8,.95)
        return np.r_[normalized/max(gate,1e-9),2*gate-1].astype(np.float32)
    def _impact_step(self,rpm):
        # Read pre-contact velocity at physics frequency; post-impact velocity is
        # insufficient because the contact impulse can make a hard impact look slow.
        env=self.core._env;original=env._physics
        incoming=None;last_free=self.core.obs_raw[0].copy();was_contact=self.contact
        def observe(command,index):
            nonlocal incoming,last_free,was_contact
            pos,quat=p.getBasePositionAndOrientation(int(env.DRONE_IDS[0]),physicsClientId=env.CLIENT)
            vel,ang=p.getBaseVelocity(int(env.DRONE_IDS[0]),physicsClientId=env.CLIENT)
            state=np.zeros(20);state[:3]=pos;state[3:7]=quat;state[7:10]=p.getEulerFromQuaternion(quat)
            state[10:13]=vel;state[13:16]=ang
            contact=self._contacts()
            if contact and not was_contact and incoming is None:incoming=last_free.copy()
            if not contact:last_free=state
            was_contact=contact
            original(command,index)
        env._physics=observe
        try:obs,_,_,_,_=env.step(rpm.reshape(1,4))
        finally:env._physics=original
        contact=self._contacts()
        if contact and not was_contact and incoming is None:incoming=last_free.copy()
        return obs,contact,incoming
    def step(self,action):
        action=np.clip(np.asarray(action,np.float32),-1,1)
        e=self.core;before=e.obs_raw[0].copy();frame=self.frames[-1].copy()
        requested=float((action[4]+1)/2);gate=requested
        if self.guard and self.need()==0.:gate=0.;self.guard_off_ticks+=1
        if self.contact:gate=0.
        if self.kind=='updated':
            residual=gate*action[:4]*self.c.wrench_scale*np.array([.35,.5,.5,.5])
            command,_,_=self.c.apply_prepared(residual)
            saturated=float(self.c.last_diagnostics['saturated_high'].mean())
        else:
            if self.kind=='robust':self.c.set_motor_effectiveness(self.estimated_eta)
            command,_,_=self.c.computeControlFromState(control_timestep=DT,state=before,
                target_pos=self.reference,target_vel=self.target_velocity)
            gate=0.;saturated=float(np.mean(command>=21665))
        if self.touchdown_seen:command=np.zeros(4)
        obs,contact,impact=self._impact_step(np.clip(command,0,21700)*self.true_eta)
        e.obs_raw=obs;self.contact=contact;self.t+=1;e.paso=self.t
        self.estimator.update(before,obs[0],command,DT,contact=contact or self.touchdown_seen)
        self.estimation_errors.append(float(np.mean(np.abs(self.estimated_eta-self.true_eta))))
        if self.fault_step is None and np.min(self.estimator.eta)<.95:self.false_alarm_ticks+=1
        s=obs[0];distance=float(np.linalg.norm(self.reference-s[:3]))
        self.max_tilt=max(self.max_tilt,float(np.max(np.abs(s[7:9]))))
        self.tracking.append(distance);self.gates.append(gate);self.requested_gates.append(requested)
        if self.route_reached and np.linalg.norm(s[:3]-e.punto_B)<.25:self.legacy_waypoint_success=True
        if self.fault_step is None:self.gates_prefault.append(gate)
        else:
            self.gates_postfault.append(gate)
            err=abs(float(self.estimated_eta[self.motor]-self.true_eta[self.motor]))
            self.failed_motor_errors.append(err)
            if err<.02 and self.estimate_recovery_tick is None:self.estimate_recovery_tick=self.t-self.fault_step
        self.gate_variations.append(abs(gate-self.previous_gate));self.previous_gate=gate
        self.effort.append(float(gate**2*np.mean(action[:4]**2)));self.saturation.append(saturated)
        success,bad=self.monitor.update(s,e.punto_B,contact,self.route_reached,impact_state=impact)
        if contact and self.route_reached:self.touchdown_seen=True
        outcome=None
        if success:outcome='stable_landing'
        elif bad:outcome='hard_landing'
        elif self.t>12 and (s[2]<.055 and not self.route_reached and
                (np.linalg.norm(s[:2]-e.punto_A[:2])>.2 or self.t>72)):outcome='early_contact'
        elif s[2]<-.03 or s[2]>5 or np.linalg.norm(s[:2]-e.punto_A[:2])>12 or np.max(np.abs(s[7:9]))>1.25:outcome='crash'
        elif not self.route_reached and distance<.25:
            e.wp_idx=min(e.wp_idx+1,len(e.waypoints)-1)
        if self.t>=1440 and outcome is None:outcome='timeout'
        done=outcome is not None;terminated=done and outcome!='timeout';truncated=outcome=='timeout'
        effort=self.effort[-1];variation=self.gate_variations[-1]
        cost=min(distance**2,4)+.03*np.dot(s[13:16],s[13:16])+.15*effort+.04*variation
        task=30. if success else -30. if done else 0.
        task+=.12*(float(np.linalg.norm(self.reference-before[:3]))-distance)-.004
        task-=.01*effort+.005*variation
        if self.route_reached:task-=.01*np.linalg.norm(s[:2]-e.punto_B[:2])+.01*abs(s[12]-self.target_velocity[2])
        self.previous_action=action.copy()
        if not done:self._prepare();after=self._obs()
        else:after=np.asarray(self.frames,np.float32)
        info=dict(state=frame,action=action.copy(),next_state=after[-1].copy(),task_reward=task,
            manual_style=-cost,outcome=outcome)
        if done:
            info.update(mission_success=success,stable_landing=success,waypoint_reached=self.legacy_waypoint_success,descent_started=self.route_reached,
                exposed=self.fault_step is not None,level=self.level,motor=self.motor,
                episode_steps=self.t,tracking_rmse=float(np.sqrt(np.mean(np.square(self.tracking)))),
                mean_gate=float(np.mean(self.gates)),requested_gate=float(np.mean(self.requested_gates)),
                effort=float(np.mean(self.effort)),gate_variation=float(np.mean(self.gate_variations)),
                guard_off_fraction=self.guard_off_ticks/self.t,saturation=float(np.mean(self.saturation)),
                max_tilt=self.max_tilt,distance_to_goal=float(np.linalg.norm(s[:2]-e.punto_B[:2])),
                first_contact=self.monitor.first,stable_seconds=self.monitor.peak_stable_ticks*DT,
                terminal_position=s[:3].tolist(),terminal_velocity=s[10:13].tolist(),
                terminal_angular_velocity=s[13:16].tolist(),eta_mae=float(np.mean(self.estimation_errors)),
                estimated_eta=self.estimated_eta.tolist(),true_eta=self.true_eta.tolist(),
                failed_motor_eta_mae=None if not self.failed_motor_errors else float(np.mean(self.failed_motor_errors)),
                first_estimate_within_2pct_seconds=None if self.estimate_recovery_tick is None else self.estimate_recovery_tick*DT,
                gate_prefault=float(np.mean(self.gates_prefault)) if self.gates_prefault else 0.,
                gate_postfault=float(np.mean(self.gates_postfault)) if self.gates_postfault else 0.,
                pre_fault_false_alarm_ticks=self.false_alarm_ticks,seed=getattr(self,'case_seed',None))
        return after,float(task+.02*-cost),bool(terminated),bool(truncated),info
