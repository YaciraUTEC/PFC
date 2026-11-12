"""Round5: paired gated motor-space corrections, impulse observer and ready descent.
Old rounds are preserved. The same landing manager is used by every controller.
"""
from collections import deque
import copy
import numpy as np
import pybullet as p
from gymnasium import spaces
from ..round3.env import TrialEnv
from .estimator import make_estimator,rotation
from .supervisor import DescentSupervisor
from ..round4.landing import LandingMonitor

LEVELS=[0.,.02,.05,.10,.20,.25,.28,.29,.293,.295]
STRESS=[.296,.30,.35,.50]
ZERO=np.array([0.,0.,0.,0.,-1.,-1.],np.float32)
DIM=80
DT=1/48

class LandingEnv(TrialEnv):
    def __init__(self,history=1,controller='updated',condition='estimated',fixed_level=None,
                 landing_profile='descent',guard=True,sensor_noise=True,config=None):
        self.config=dict(estimator='original',readiness=True,emergency='legacy',reserve=.005,
            intervention='residual',gate_mode='learned',gate_threshold=0.,noise_multiplier=1.,fault_clock=None,
            physics_hz=240,contact_hz=960,solver_iterations=200,solver_residual=1e-12,sensor_model='shared_velocity_gyro')
        self.config.update(config or {})
        super().__init__(history=1,controller=controller,condition=condition,fixed_level=fixed_level)
        if controller=='updated':
            from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
            from gym_pybullet_drones.utils.enums import DroneModel
            self.c=DSLPIDControl(DroneModel.CF2X,thrust_reserve_fraction=self.config['reserve'],
                emergency_thrust_ratio=1.0001,emergency_policy=self.config['emergency'])
            self.core.pid_controller=self.c
        self.history=history;self.frames=deque(maxlen=history)
        self.observation_space=spaces.Box(-10.,10.,(history,DIM),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,(6,),dtype=np.float32)
        self.landing_profile=landing_profile;self.guard=guard;self.sensor_noise=sensor_noise
        self.physics_model=self.c
        noise=.002*self.config['noise_multiplier'] if sensor_noise else 0.
        self._sensor_sigma=noise
        self.estimator=make_estimator(self.physics_model,self.config['estimator'],velocity_noise=0.,gyro_noise=0.)
        self.descent=DescentSupervisor(enabled=self.config['readiness']);self.descent.reset()
        self.judge=None
        if self.config.get('judge_path'):
            from .judge import BenefitJudge
            self.judge=BenefitJudge.load(self.config['judge_path'])
    def reset(self,seed=None,options=None):
        import gymnasium as gym
        gym.Env.reset(self,seed=seed)
        self.descent.reset()
        self.last_requested=np.zeros(6);self.last_applied=ZERO.copy();self.applied_eta=np.ones(4)
        self.gate_rejections=0;self.q_delta_energy=[];self.benefit_predictions=[]
        self.core.reset(seed=seed)
        sim=self.core._env;hz=int(self.config['physics_hz'])
        if hz%48:raise ValueError('Physics frequency must be divisible by control frequency48')
        sim.PYB_FREQ=hz;sim.PYB_TIMESTEP=1/hz;sim.PYB_STEPS_PER_CTRL=hz//48
        p.setTimeStep(1/hz,physicsClientId=sim.CLIENT)
        p.setPhysicsEngineParameter(numSolverIterations=int(self.config['solver_iterations']),
            solverResidualThreshold=float(self.config['solver_residual']),physicsClientId=sim.CLIENT)
        if seed is not None or not hasattr(self,'rng'):
            self.rng=np.random.default_rng(None if seed is None else seed+99173)
        self.level=float((options or {}).get('level',self.fixed_level if self.fixed_level is not None else self.rng.choice(LEVELS)))
        self.motor=int((options or {}).get('motor',self.rng.integers(4)))
        self.bias=float(self.rng.uniform(.45,.9));self.delay=int(self.rng.integers(2,7))
        self.t=0;self.fault_step=None;self.true_eta=np.ones(4);self.estimated_eta=np.ones(4)
        sensor_seed=int(self.rng.integers(1,2**30)) if seed is None else int(seed)
        self.sensor_rng=np.random.default_rng(sensor_seed+7711039)
        self.measured=self._measurement(self.core.obs_raw[0])
        self.estimator.reset(sensor_seed)
        self.applied_estimation_errors=[];self.applied_postfault_errors=[];self.estimator_coverages=[]
        self.requested_on_ticks=0;self.applied_on_ticks=0;self.effectual_ticks=0
        self.previous_action=np.r_[np.zeros(4),-1.].astype(np.float32);self.tracking=[];self.effort=[];self.gates=[];self.saturation=[]
        self.max_tilt=0.;self.route_reached=False;self.descent_start=None;self.contact=False
        self.monitor=LandingMonitor();self.estimation_errors=[];self.false_alarm_ticks=0
        self.guard_off_ticks=0;self.requested_gates=[];self.gate_variations=[];self.previous_gate=0.
        self.touchdown_seen=False;self.reference=None;self.legacy_waypoint_success=False
        self.failed_motor_errors=[];self.gates_prefault=[];self.gates_postfault=[];self.estimate_recovery_tick=None
        self.frames.clear()
        for _ in range(self.history-1):self.frames.append(np.zeros(DIM,np.float32))
        self._prepare()
        return self._obs(),{}
    def _measurement(self,physical):
        result=physical.copy()
        result[10:13]+=self.sensor_rng.normal(0,self._sensor_sigma,3)
        result[13:16]+=self.sensor_rng.normal(0,self._sensor_sigma,3)
        return result
    def _contacts(self):
        e=self.core._env
        return any(c[8]<=.001 for c in p.getContactPoints(int(e.DRONE_IDS[0]),e.PLANE_ID,physicsClientId=e.CLIENT))
    def _prepare(self):
        e=self.core;s=self.measured;clock=self.config['fault_clock']
        triggered=e.wp_idx>=e.wp_medio_crucero if clock is None else self.t>=round(48*clock)
        if self.level>0 and self.fault_step is None and triggered:self.fault_step=self.t
        if self.fault_step is not None:self.true_eta[self.motor]=1-self.level
        if self.condition=='exact':self.estimated_eta=self.true_eta.copy()
        elif self.condition=='biased':
            self.estimated_eta=np.ones(4)
            if self.fault_step is not None and self.t-self.fault_step>=self.delay:self.estimated_eta[self.motor]=1-self.level*self.bias
        else:self.estimated_eta=self.estimator.eta.copy()
        self.control_sigma=np.zeros(4) if self.condition=='exact' else self.estimator.uncertainty.copy()
        final=e.wp_idx>=len(e.waypoints)-1
        target=e.waypoints[min(e.wp_idx,len(e.waypoints)-1)].copy();velocity=np.zeros(3)
        if final:
            self.route_reached=True
            if self.descent_start is None:self.descent_start=(self.t,float(s[2]))
            if self.landing_profile=='descent':
                if self.config['readiness']:target,velocity=self.descent.reference(s,e.punto_B,DT)
                else:
                    target[2]=max(-.025,self.descent_start[1]-.18*(self.t-self.descent_start[0])*DT)
                    velocity[2]=-.18 if target[2]>-.025 else 0.
        self.reference=target;self.target_velocity=velocity
        self._arguments=dict(control_timestep=DT,cur_pos=s[:3],cur_quat=s[3:7],cur_vel=s[10:13],
            cur_ang_vel=s[13:16],target_pos=target,target_vel=velocity)
        self.c.set_motor_effectiveness(self.estimated_eta)
        self._integrals=(self.c.integral_pos_e.copy(),self.c.integral_rpy_e.copy())
        self.base=self.c.prepare_control(**self._arguments)
        rpm,self.predicted=self.c.allocate_wrench(self.base,commit=False)
        self.baseq=(rpm/self.c.HOVER_RPM)**2
        self.qmax=(self.c._command_max/self.c.HOVER_RPM)**2
        self.qscale=.20*self.qmax
    def _state(self):
        e=self.core;s=self.measured;R=rotation(s);omega=R.T@s[13:16]
        prepared=self.c._prepared or dict(balanced_max=self.c.last_diagnostics.get('balanced_max_force',self.c.GRAVITY))
        state=np.r_[(self.reference-s[:3]),s[10:13]/2,R.flatten(),omega/10,
            (e.punto_B-s[:3])/4,self.base/self.c.wrench_scale,self.c.integral_pos_e/2,
            self.c.integral_rpy_e/.5,(1-self.estimated_eta)/.25,self.previous_action,s[2]/1.2,
            e.wp_idx/(len(e.waypoints)-1),self.predicted['allocation_error'],
            prepared['balanced_max']/self.c.GRAVITY-1.,self.t/960]
        assert state.shape==(48,)
        uncertainty=self.estimator.uncertainty
        if self.condition=='exact':uncertainty=np.zeros(4)
        oldextra=np.r_[uncertainty/.1,self.estimator.innovation,
            float(self.route_reached),float(self.contact),self.monitor.stable_ticks/48,
            self.previous_gate,self.target_velocity,min(self.estimator.updates,100)/100]
        s=self.measured
        extra=np.r_[self.baseq/self.qmax,self.predicted['allocation_error'],
            (1440-self.t)/1440,self.descent.ready_ticks/12,
            0. if self.descent.height is None else self.descent.height,
            float(self.descent.started),np.linalg.norm(s[13:16])/10,
            self.last_applied[-1],self.last_applied[-2],float(self.config['intervention']=='effectiveness')]
        result=np.r_[state,oldextra,extra]
        assert result.shape==(DIM,),result.shape
        return np.clip(result,-10,10).astype(np.float32)
    def need(self):
        # This compatibility method never substitutes state error for benefit.
        return 1.
    def teacher_action(self):
        if self.contact:return ZERO.copy()
        if self.config['intervention']=='effectiveness':
            delta=(self.true_eta**2-self.estimated_eta**2)/.10
        else:
            teacher=copy.deepcopy(self.c)
            teacher.integral_pos_e=self._integrals[0].copy();teacher.integral_rpy_e=self._integrals[1].copy()
            teacher.set_motor_effectiveness(self.true_eta)
            w=teacher.prepare_control(**self._arguments);rpm,_=teacher.allocate_wrench(w,commit=False)
            oracleq=(rpm/teacher.HOVER_RPM)**2
            delta=(oracleq-self.baseq)/self.qscale
        delta=np.clip(delta,-1.,1.);amplitude=float(np.max(np.abs(delta)))
        if amplitude<.002:return ZERO.copy()
        return np.r_[delta/amplitude,2*amplitude-1,1.].astype(np.float32)
    def candidate_features(self,action):
        a=np.asarray(action,float);effective=((a[4]+1)/2)*a[:4]
        mode=self.config['intervention'];return np.r_[self._state(),effective,float(mode=='residual'),float(mode=='effectiveness')].astype(np.float32)
    def _command(self,action):
        a=np.clip(np.asarray(action,np.float32),-1,1);self.last_requested=a.copy()
        amplitude=float((a[4]+1)/2);active=bool(a[5]>self.config['gate_threshold'])
        if self.contact:active=False
        if self.config['gate_mode']=='off':active=False
        if active and self.judge is not None and self.config['gate_mode']=='conservative':
            decision,lower=self.judge.accept(self.candidate_features(a))
            self.benefit_predictions.append(float(lower))
            if not decision:active=False;self.gate_rejections+=1
        gate=amplitude if active else 0.
        self.requested_on_ticks+=int(a[5]>0);self.applied_on_ticks+=int(gate>0)
        self.last_applied=a.copy();self.last_applied[4]=2*gate-1;self.last_applied[5]=1. if active else -1.
        self.applied_eta=self.estimated_eta.copy()
        if self.config['intervention']=='effectiveness' and active:
            beta=np.clip(self.estimated_eta**2+.10*gate*a[:4],.04,1.)
            self.applied_eta=np.sqrt(beta);self.c.set_motor_effectiveness(self.applied_eta)
            self.c.integral_pos_e=self._integrals[0].copy();self.c.integral_rpy_e=self._integrals[1].copy()
            self.base=self.c.prepare_control(**self._arguments)
            rpm,_,_=self.c.apply_prepared(np.zeros(4))
        else:
            q=np.clip(self.baseq+self.qscale*gate*a[:4],0.,self.qmax)
            rpm,_,_=self.c.apply_prepared_q(q)
        energy=float(np.mean(((rpm/self.c.HOVER_RPM)**2-self.baseq)**2))
        self.q_delta_energy.append(energy);self.effectual_ticks+=int(energy>1e-16)
        return rpm,gate
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
        e=self.core;before=e.obs_raw[0].copy();measured_before=self.measured.copy();frame=self.frames[-1].copy()
        requested=float((action[4]+1)/2);gate=requested
        command,gate=self._command(action)
        saturated=float(self.c.last_diagnostics['saturated_high'].mean())
        if gate==0.:self.guard_off_ticks+=1
        if self.touchdown_seen:
            command=np.zeros(4)
            sim=e._env;hz=int(self.config.get('contact_hz',self.config['physics_hz']))
            if sim.PYB_FREQ!=hz:
                if hz%48:raise ValueError('Contact frequency must be divisible by48')
                sim.PYB_FREQ=hz;sim.PYB_TIMESTEP=1/hz;sim.PYB_STEPS_PER_CTRL=hz//48
                p.setTimeStep(1/hz,physicsClientId=sim.CLIENT)
        obs,contact,impact=self._impact_step(np.clip(command,0,21700)*self.true_eta)
        e.obs_raw=obs;self.contact=contact;self.t+=1;e.paso=self.t
        self.measured=self._measurement(obs[0])
        self.estimator.update(measured_before,self.measured,command,DT,contact=contact or self.touchdown_seen)
        self.applied_estimation_errors.append(np.abs(self.applied_eta-self.true_eta))
        if self.fault_step is not None:
            self.applied_postfault_errors.append(np.abs(self.applied_eta-self.true_eta))
            sigma=self.control_sigma[self.motor]
            self.estimator_coverages.append(abs(self.estimated_eta[self.motor]-self.true_eta[self.motor])<=1.96*sigma+1e-12)
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
        done=outcome is not None;terminated=done;truncated=False
        effort=self.effort[-1];variation=self.gate_variations[-1]
        cost=min(distance**2,4)+.03*np.dot(s[13:16],s[13:16])+.15*effort+.04*variation
        terminal_reward=100. if success else -100. if done else 0.
        dense=.12*(float(np.linalg.norm(self.reference-before[:3]))-distance)-.004
        dense-=.01*effort+.005*variation
        if self.route_reached:dense-=.01*np.linalg.norm(s[:2]-e.punto_B[:2])+.01*abs(s[12]-self.target_velocity[2])
        task=terminal_reward+float(np.clip(dense,-.005,.005))
        self.previous_action=np.r_[action[:4],2*gate-1].astype(np.float32)
        if not done:self._prepare();after=self._obs()
        else:
            after=self._obs()
        info=dict(state=frame,action=action.copy(),next_state=after[-1].copy(),task_reward=task,
            manual_style=-cost,outcome=outcome,deadline_terminal=outcome=='timeout',
            gate_active=bool(self.last_applied[-1]>0),applied_amplitude=gate,applied_effectiveness=self.applied_eta.tolist())
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
                pre_fault_false_alarm_ticks=self.false_alarm_ticks,seed=getattr(self,'case_seed',None),
                descent_paused_seconds=self.descent.paused_ticks*DT,
                descent_readiness_enabled=self.descent.enabled,explicit_gate_rejections=self.gate_rejections,
                q_delta_energy=float(np.mean(self.q_delta_energy)),configuration=self.config.copy(),
                applied_eta_mae=float(np.mean(self.applied_estimation_errors)),
                applied_failed_motor_eta_mae=None if not self.applied_postfault_errors else float(np.mean(np.asarray(self.applied_postfault_errors)[:,self.motor])),
                requested_on_fraction=self.requested_on_ticks/self.t,applied_on_fraction=self.applied_on_ticks/self.t,effectual_fraction=self.effectual_ticks/self.t,
                estimated_95pct_coverage=None if not self.estimator_coverages else float(np.mean(self.estimator_coverages)))
        return after,float(task+np.clip(-.001*cost,-.005,.005)),bool(terminated),bool(truncated),info
