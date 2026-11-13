"""Single-rotor, motion-only impulse estimator for the registered single-fault task.
Healthy rotors remain nominal rather than acquiring clipping-induced deficits.
The assumption is explicit: at most one faulty rotor; not a multi-fault observer.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from ..round4.estimator import EffectivenessEstimator,rotation

class StructuredEstimator(EffectivenessEstimator):
    def __init__(self,controller,**kw):
        self.window_size=8;super().__init__(controller,**kw)
    def reset(self,seed=0):
        super().reset(seed);self.eta=np.ones(4);self.detector=None;self.samples=deque(maxlen=self.window_size)
        self.candidate=None;self.candidate_ticks=0;self.filtered_beta=1.;self.scalar_variance=.01
        self._raw=EffectivenessEstimator.__new__(EffectivenessEstimator)
        for name in ['geometry','kf','mass','J','scale','forgetting','velocity_noise','gyro_noise']:setattr(self._raw,name,getattr(self,name))
        EffectivenessEstimator.reset(self._raw,seed)
    def update(self,before,after,command,dt,contact=False):
        if contact or min(before[2],after[2])<.08:return
        self._raw.update(before,after,command,dt,contact)
        raw=self._raw.eta;candidate=int(np.argmin(raw)) if raw.min()<.98 else None
        if candidate!=self.candidate:
            self.candidate_ticks=self.candidate_ticks+1 if candidate==self.detector else 1;self.detector=candidate
            if self.candidate_ticks>=2:
                self.candidate=candidate;self.samples.clear()
                self.filtered_beta=1. if candidate is None else float(raw[candidate]**2)
        R0,R1=rotation(before),rotation(after)
        rotvec=Rotation.from_matrix(R0.T@R1).as_rotvec();f=np.array([0.,.25,.5,.75,1.])
        avg=(R0[None]@Rotation.from_rotvec(f[:,None]*rotvec).as_matrix()*np.array([1,4,2,4,1])[:,None,None]).sum(0)/12
        thrust=self.kf*np.asarray(command)**2
        H=np.vstack([avg[:,2,None]*thrust[None],(avg@self.geometry[1:])*thrust[None]])*dt
        L0=R0@self.J@R0.T@before[13:16];L1=R1@self.J@R1.T@after[13:16]
        y=np.r_[self.mass*(after[10:13]-before[10:13]+np.array([0.,0.,9.8*dt])),L1-L0]
        self.samples.append((H,y));self.eta=np.ones(4)
        if self.candidate is not None:
            if len(self.samples)>=4:
                Hs=sum(a for a,b in self.samples);ys=sum(b for a,b in self.samples)
                scale=np.r_[np.repeat(self.scale[0],3),self.scale[1:]]*dt*len(self.samples)
                # Gyro impulse channels have smaller normalized endpoint noise.
                weight=np.r_[np.repeat(1/.03,3),np.repeat(1/.003,3)]/scale
                h=Hs[:,self.candidate]*weight;residual=(ys-Hs@np.ones(4))*weight
                beta=1+float(h@residual)/max(float(h@h),1e-12)
                self.filtered_beta=.5*self.filtered_beta+.5*np.clip(beta,.04,1.05)
                self.scalar_variance=max(1e-7,float(np.dot(residual-h*(beta-1),residual-h*(beta-1)))/max(float(h@h),1e-9))
            else:self.filtered_beta=float(raw[self.candidate]**2)
            self.eta[self.candidate]=np.sqrt(np.clip(self.filtered_beta,.04,1.))
        self.beta=self.eta**2;self.innovation=self._raw.innovation.copy();self.updates=self._raw.updates
    @property
    def uncertainty(self):
        sigma=np.full(4,.001)
        if self.candidate is not None:sigma[self.candidate]=np.sqrt(self.scalar_variance)/max(2*self.eta[self.candidate],.1)
        return sigma
