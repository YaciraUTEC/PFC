"""Historical 24-input/RPM-residual adapter under the new common landing task.
No retraining. The gate/fault fields now use the provided exact/estimated eta.
The old policies were trained with motor0 only; varying motors is a shift.
"""
import numpy as np
from gymnasium import spaces
from .env import LandingEnv,ZERO

class HistoricalRPMEnv(LandingEnv):
    def __init__(self,**kw):
        super().__init__(controller='legacy',guard=False,**kw)
        self.observation_space=spaces.Box(-10.,10.,(1,24),dtype=np.float32)
        self.direct=np.zeros(4)
        original=self.c.computeControlFromState
        def compensated(**arguments):
            rpm,e,y=original(**arguments)
            active=bool(np.min(self.estimated_eta)<.99)
            self.direct_active=active
            rpm=np.clip(rpm+3000*self.direct*active,9440.,21700.)
            return rpm,e,y
        self.c.computeControlFromState=compensated
    def _legacy(self):
        self.core.falla_activa=bool(np.min(self.estimated_eta)<.99)
        self.core.severidad=float(np.min(self.estimated_eta))
        return np.clip(self.core._build_obs(self.core.obs_raw,self.reference),-10,10)[None].astype(np.float32)
    def reset(self,seed=None,options=None):
        self.direct=np.zeros(4);self.direct_active=False;self.legacy_efforts=[];self.legacy_active=[]
        _,i=super().reset(seed=seed,options=options)
        return self._legacy(),i
    def step(self,action):
        self.direct=np.asarray(action,np.float32);cutoff=self.touchdown_seen
        _,r,d,t,i=super().step(ZERO)
        active=float(self.direct_active and not cutoff)
        self.legacy_active.append(active);self.legacy_efforts.append(active*float(np.mean(self.direct**2)))
        if d or t:i.update(mean_gate=float(np.mean(self.legacy_active)),effort=float(np.mean(self.legacy_efforts)),
            intervention_units='normalized RPM residual,not wrench')
        return self._legacy(),r,d,t,i
