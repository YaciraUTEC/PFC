"""Round-three task: physical DSLPID + gated residual wrench, same information.
Original route-completion criterion is retained and labelled; not touchdown.
"""
from pathlib import Path
from collections import deque
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..env import ROOT
from nominal_flight_env import NominalFlightEnv
from comparar_base import es_caida,es_aterrizaje,UMBRAL_WAYPOINT,CTRL_FREQ
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.LegacyDSLPIDControl import DSLPIDControl as Legacy
from gym_pybullet_drones.control.RobustDSLPIDControl import RobustDSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

LEVELS=[0.,.02,.05,.10,.15,.20,.25]
class TrialEnv(gym.Env):
    def __init__(self,history=32,controller='updated',condition='uncertain',fixed_level=None,
                 reward_model=None,oracle=False):
        self.core=NominalFlightEnv(niveles_falla_pct=LEVELS[1:])
        self.kind=controller;self.condition=condition;self.fixed_level=fixed_level
        self.oracle=oracle;self.reward_model=reward_model;self.history=history
        self.frames=deque(maxlen=history)
        self.c=DSLPIDControl(DroneModel.CF2X)
        if controller=='legacy':self.c=Legacy(DroneModel.CF2X)
        elif controller=='robust':self.c=RobustDSLPIDControl(DroneModel.CF2X,fault_aware=True)
        self.core.pid_controller=self.c
        self.observation_space=spaces.Box(-10.,10.,shape=(history,48),dtype=np.float32)
        self.action_space=spaces.Box(-1.,1.,shape=(5,),dtype=np.float32)
        self._manual=True
    def set_fixed_fault(self,value):self.fixed_level=value
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        _,_=self.core.reset(seed=seed)
        if seed is not None or not hasattr(self,'rng'):
            self.rng=np.random.default_rng(None if seed is None else seed+99173)
        self.level=float((options or {}).get('level',self.fixed_level if self.fixed_level is not None else self.rng.choice(LEVELS)))
        self.motor=int((options or {}).get('motor',self.rng.integers(4)))
        self.bias=float(self.rng.uniform(.45,.9));self.delay=int(self.rng.integers(2,7))
        if self.condition=='exact':self.bias=1.;self.delay=0
        self.t=0;self.fault_step=None;self.true_eta=np.ones(4);self.estimated_eta=np.ones(4)
        self.previous_action=np.r_[np.zeros(4),-1.].astype(np.float32)
        self.tracking=[];self.effort=[];self.gates=[];self.saturation=[];self.max_tilt=0.
        self.frames.clear()
        for _ in range(self.history-1):self.frames.append(np.zeros(48,np.float32))
        self._prepare()
        return self._obs(),{}
    def _prepare(self):
        e=self.core
        if self.level>0 and self.fault_step is None and e.wp_idx>=e.wp_medio_crucero:
            self.fault_step=self.t
        if self.fault_step is not None:
            self.true_eta[self.motor]=1-self.level
            if self.t-self.fault_step>=self.delay:self.estimated_eta[self.motor]=1-self.level*self.bias
        c=self.c
        if self.kind=='updated':
            c.set_motor_effectiveness(self.true_eta if self.oracle else self.estimated_eta)
            self._integrals=(c.integral_pos_e.copy(),c.integral_rpy_e.copy())
            state=e.obs_raw[0];wp=e.waypoints[min(e.wp_idx,len(e.waypoints)-1)]
            self._arguments=dict(control_timestep=1/48,cur_pos=state[:3],cur_quat=state[3:7],
                cur_vel=state[10:13],cur_ang_vel=state[13:16],target_pos=wp)
            self.base=c.prepare_control(**self._arguments)
            _,self.predicted=c.allocate_wrench(self.base,commit=False)
        else:self.base=np.zeros(4);self.predicted={}
    def _state(self):
        e=self.core;s=e.obs_raw[0];wp=e.waypoints[min(e.wp_idx,len(e.waypoints)-1)]
        import pybullet as p
        R=np.asarray(p.getMatrixFromQuaternion(s[3:7])).reshape(3,3)
        omega=R.T@s[13:16]
        if self.kind=='updated':
            scale=self.c.wrench_scale;err=self.predicted['allocation_error']
            margin=self.c._prepared['balanced_max']/self.c.GRAVITY-1.
        else:
            scale=np.array([.294,.00825,.00825,.00739]);err=np.zeros(4);margin=0.
        state=np.r_[(wp-s[:3]),s[10:13]/2,R.flatten(),omega/10,(e.punto_B-s[:3])/4,
            self.base/scale,self.c.integral_pos_e/2,self.c.integral_rpy_e/.5,
            (1-self.estimated_eta)/.25,self.previous_action,s[2]/1.2,
            e.wp_idx/(len(e.waypoints)-1),err,margin,self.t/960]
        assert state.shape==(48,),state.shape
        return np.clip(state,-10,10).astype(np.float32)
    def _obs(self):
        self.frames.append(self._state());return np.asarray(self.frames,dtype=np.float32)
    def teacher_action(self):
        if self.kind!='updated':return np.r_[np.zeros(4),-1.].astype(np.float32)
        # Teacher shares the delay; true effectiveness is privileged only in demonstrations.
        if np.all(self.estimated_eta==1.):return np.r_[np.zeros(4),-1.].astype(np.float32)
        teacher=copy.deepcopy(self.c)
        teacher.integral_pos_e=self._integrals[0].copy();teacher.integral_rpy_e=self._integrals[1].copy()
        teacher.set_motor_effectiveness(self.true_eta)
        w=teacher.prepare_control(**self._arguments)
        rpm,_=teacher.allocate_wrench(w,commit=False)
        q=(rpm/self.c.HOVER_RPM)**2
        equivalent=self.c._allocation_matrix()@q
        normalized=(equivalent-self.base)/(self.c.wrench_scale*np.array([.35,.5,.5,.5]))
        normalized=np.clip(normalized,-.76,.76)
        amplitude=float(np.max(np.abs(normalized)))
        gate=min(amplitude/.8,.95)
        direction=normalized/max(gate,1e-9)
        return np.r_[direction,2*gate-1].astype(np.float32)
    def step(self,action):
        action=np.clip(np.asarray(action,np.float32),-1,1)
        e=self.core;before=self.frames[-1].copy();wp=e.waypoints[min(e.wp_idx,len(e.waypoints)-1)]
        if self.kind=='updated':
            gate=float((action[4]+1)/2)
            residual=gate*action[:4]*self.c.wrench_scale*np.array([.35,.5,.5,.5])
            command,_,_=self.c.apply_prepared(residual)
            saturated=float(self.c.last_diagnostics['saturated_high'].mean())
        else:
            if self.kind=='robust':self.c.set_motor_effectiveness(self.true_eta if self.oracle else self.estimated_eta)
            command,_,_=self.c.computeControlFromState(control_timestep=1/48,state=e.obs_raw[0],target_pos=wp)
            gate=0.;saturated=float(np.mean(command>=21665.))
        rpm=np.clip(command,0,21700)*self.true_eta
        obs,_,_,_,_=e._env.step(rpm.reshape(1,4));e.obs_raw=obs
        self.t+=1;e.paso=self.t
        pos=obs[0,:3];velocity=obs[0,10:13]
        distance=float(np.linalg.norm(wp-pos));tilt=float(np.max(np.abs(obs[0,7:9])))
        self.max_tilt=max(self.max_tilt,tilt)
        self.tracking.append(distance);self.gates.append(gate)
        self.effort.append(float(gate**2*np.mean(action[:4]**2)));self.saturation.append(saturated)
        outcome=None
        if es_caida(obs,pos,self.t,e.wp_idx,len(e.waypoints)):outcome='crash'
        elif es_aterrizaje(obs,pos,self.t):outcome='landing'
        elif distance<UMBRAL_WAYPOINT:
            e.wp_idx+=1
            if e.wp_idx>=len(e.waypoints):outcome='mission'
        if self.t>=960 and outcome is None:outcome='timeout'
        terminated=outcome is not None and outcome!='timeout';truncated=outcome=='timeout'
        # Common explicit task anchors. Learned style is evaluated separately
        # and against the manual-style ablation with identical anchors.
        task=10. if outcome=='mission' else -10. if outcome=='crash' else -5. if outcome else 0.
        task-=.002*self.effort[-1]
        omega=obs[0,13:15]
        style=-float(min(distance**2,4)+.02*np.dot(omega,omega)+.1*self.effort[-1])
        self.previous_action=action.copy()
        done=terminated or truncated
        if not done:self._prepare()
        else:
            self.base=np.zeros(4)
            if self.kind=='updated':self.predicted={'allocation_error':np.zeros(4)}
        # Last observation has no newly prepared controller; use last frame on terminal.
        after=self._obs() if not done else np.asarray(self.frames,dtype=np.float32)
        reward=task+.02*style
        info=dict(state=before,action=action.copy(),next_state=after[-1].copy(),task_reward=task,
            manual_style=style,actual_level=self.level,estimation_bias=self.bias,motor=self.motor,
            terminated=bool(terminated),outcome=outcome)
        if done:
            info.update(mission_success=outcome=='mission',exposed=self.fault_step is not None,
                pasos_post_falla=0 if self.fault_step is None else self.t-self.fault_step,
                tracking_rmse=float(np.sqrt(np.mean(np.square(self.tracking)))),
                mean_gate=float(np.mean(self.gates)),effort=float(np.mean(self.effort)),
                saturation=float(np.mean(self.saturation)),max_tilt=self.max_tilt,
                episode_steps=self.t,distance_to_goal=float(np.linalg.norm(pos-e.punto_B)))
        if self.reward_model is not None:reward=task+.02*float(self.reward_model.predict(before,action,after[-1],done))
        return after,float(reward),terminated,truncated,info
    def close(self):self.core.close()