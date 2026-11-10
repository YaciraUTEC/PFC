"""One DSLPIDControl residual task for IRL, all six policies, and evaluation."""
from pathlib import Path
from collections import deque
import sys,json
import numpy as np
from gymnasium import spaces
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'comparacion'))
sys.path.insert(0,str(ROOT/'fallas'))
from nominal_flight_env import NominalFlightEnv
from irl_features import N_FEATURES,FEATURE_NAMES
LEVELS=[.02,.05,.10,.15,.20]
EXPERT_LEVELS=[.02,.04,.06,.08,.10]

class FlightEnv(NominalFlightEnv):
    def __init__(self,reward_path=None,levels=None,history=1,fixed_fault=None,
                 shaping_coef=.01,task_reward=True,gui=False,warm_start=False):
        super().__init__(gui=gui,controlador="pid",fault_aware=False,
                         niveles_falla_pct=LEVELS if levels is None else levels)
        self.history_len=int(history);self.fixed_fault=fixed_fault
        self.warm_start=warm_start;self._pending_done=None
        self.task_reward=task_reward;self.shaping_coef=shaping_coef
        self.feature_scale=np.ones(N_FEATURES)
        self.theta=np.r_[np.ones(9)/3.,np.zeros(2)]
        if reward_path:
            data=json.loads(Path(reward_path).read_text(encoding='utf-8-sig'))
            if data["feature_names"]!=FEATURE_NAMES:raise ValueError("Reward feature order mismatch")
            self.theta=np.asarray(data["weights"],float)
            self.feature_scale=np.asarray(data["feature_scale"],float)
            if np.any(self.theta<0) or np.any(self.feature_scale<=0):
                raise ValueError("Invalid constrained reward")
            if np.any(self.theta[9:] != 0):
                raise ValueError("Policy-relative features must be excluded; regenerate the IRL reward")
            self.shaping_coef=float(data.get("shaping_coef",shaping_coef))
        self.set_reward_weights(self.theta/self.feature_scale)
        self.frames=deque(maxlen=self.history_len)
        self.observation_space=spaces.Box(-10.,10.,shape=(24,) if history==1 else (history,24),dtype=np.float32)
        self.paso_falla=None
    def set_fixed_fault(self,value):
        self.fixed_fault=None if value is None else float(value)
    def configure_reward(self,theta,scale):
        self.theta=np.asarray(theta,float);self.feature_scale=np.asarray(scale,float)
        self.set_reward_weights(self.theta/self.feature_scale)
    def _history(self,obs):
        if self.history_len==1:return np.clip(obs,-10,10).astype(np.float32)
        self.frames.append(np.clip(obs,-10,10).astype(np.float32))
        return np.asarray(self.frames,dtype=np.float32)
    def reset(self,seed=None,options=None):
        self._pending_done=None
        obs,info=super().reset(seed=seed,options=options)
        fixed=(options or {}).get("fault_pct",self.fixed_fault)
        if fixed is not None:self.severidad=1.-float(fixed)
        if self.warm_start and self.fixed_fault is None and self.severidad == 1.:
            choices=[v for v in self.niveles if v<=self.max_fault_pct+1e-9]
            self.severidad=1.-float(self.np_random.choice(choices))
        self.paso_falla=None;self.max_tilt=0.;self.min_altitude=10.
        self.frames.clear()
        for _ in range(self.history_len-1):self.frames.append(np.zeros(24,np.float32))
        obs=self._history(self._build_obs(self.obs_raw,self.waypoints[0]))
        if self.warm_start and self.severidad<1.:
            # The base controller alone owns the prefix in every experiment.
            # Advance physics and the observation history, not neural inference.
            # Preserve prefix failures as terminal samples; never retry silently.
            while self.wp_idx<self.wp_medio_crucero:
                result=self.step(np.zeros(4,np.float32))
                obs,reward,terminated,truncated,step_info=result
                if terminated or truncated:
                    step_info["pre_exposure_failure"]=True
                    self._pending_done=result
                    break
        return obs,info
    def step(self,action):
        if self._pending_done is not None:
            result=self._pending_done;self._pending_done=None
            return result
        previously_exposed=self.falla_activada
        obs,reward,done,truncated,info=super().step(np.asarray(action,np.float32))
        if self.falla_activada and not previously_exposed:self.paso_falla=self.paso
        self.max_tilt=max(self.max_tilt,float(np.abs(self.obs_raw[0,7:9]).max()))
        if self.falla_activada:self.min_altitude=min(self.min_altitude,float(self.obs_raw[0,2]))
        reward*=self.shaping_coef
        if self.task_reward:
            # Explicit task/safety prior, kept identical for all architectures.
            if self.falla_activada:reward+=.02-.02*float(np.mean(np.square(np.clip(action,-1,1))))
            if done:
                outcome=info.get("outcome","tiempo")
                dist=float(np.linalg.norm(self.obs_raw[0,:2]-self.punto_B[:2]))
                reward+=100. if outcome=="llego" else (-120. if outcome=="cayo" else
                         (40. if dist<.3 else -60.) if outcome=="aterrizo" else -30.)
        if done:
            dist=float(np.linalg.norm(self.obs_raw[0,:2]-self.punto_B[:2]))
            info.update(t_falla_seg=None if self.paso_falla is None else self.paso_falla/48.,
                        pasos_post_falla=0 if self.paso_falla is None else self.paso-self.paso_falla+1,
                        exposed=bool(self.falla_activada and self.severidad<1),
                        planned_fault_pct=1-self.severidad,
                        mission_success=info.get("outcome")=="llego",
                        landed_near=info.get("outcome")=="aterrizo" and dist<.3,
                        distance_to_goal=dist,max_tilt=self.max_tilt,
                        min_altitude_post_fault=self.min_altitude if self.falla_activada else None,
                        base_controller=type(self.pid_controller).__name__)
        timeout=done and info.get("outcome")=="tiempo"
        return self._history(obs),float(reward),bool(done and not timeout),bool(timeout),info
