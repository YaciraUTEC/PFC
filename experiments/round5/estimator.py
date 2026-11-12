"""Impulse/momentum RLS. Runtime inputs are measured states and motor commands.
The orientation integral replaces midpoint body acceleration approximations.
No fault identity, true effectiveness, or fault clock is accepted by update().
"""
import numpy as np
from scipy.spatial.transform import Rotation
from ..round4.estimator import EffectivenessEstimator as OriginalEstimator,rotation

class ImpulseEstimator(OriginalEstimator):
    def __init__(self,controller,variant='impulse',**kw):
        self.variant=variant
        super().__init__(controller,forgetting=.995,**kw)
    def reset(self,seed=0):
        super().reset(seed);self.change_count=0;self.change_detections=0
        self.innovation_score=0.;self.last_interval=(np.ones(4),np.ones(4))
    def update(self,before,after,command,dt,contact=False):
        if contact or min(before[2],after[2])<.08 or not np.isfinite(after).all():return
        R0,R1=rotation(before),rotation(after)
        relative=Rotation.from_matrix(R0.T@R1).as_rotvec()
        fractions=np.array([0.,.25,.5,.75,1.])
        matrices=R0[None]@Rotation.from_rotvec(fractions[:,None]*relative).as_matrix()
        avg=(matrices*np.array([1,4,2,4,1])[:,None,None]).sum(0)/12
        dv=after[10:13]-before[10:13]+self.rng.normal(0,self.velocity_noise*np.sqrt(2),3)
        omega0=before[13:16]+self.rng.normal(0,self.gyro_noise,3)
        omega1=after[13:16]+self.rng.normal(0,self.gyro_noise,3)
        L0=R0@self.J@R0.T@omega0;L1=R1@self.J@R1.T@omega1
        force=self.mass*(dv/dt+np.array([0.,0.,9.8]))
        torque=(L1-L0)/dt
        thrust=self.kf*np.asarray(command)**2
        H=np.vstack([avg[:,2,None]*thrust[None],(avg@self.geometry[1:])*thrust[None]])
        scale=np.r_[np.repeat(self.scale[0],3),self.scale[1:]]
        H=H/scale[:,None];y=np.r_[force,torque]/scale
        noise=np.diag(np.r_[np.repeat(.012,3),[.018,.018,.025]]**2)
        innovation=y-H@self.beta
        score=float(innovation@np.linalg.solve(noise+H@self.P@H.T,innovation))
        self.innovation_score=.7*self.innovation_score+.3*score
        # A surprise resets adaptation uncertainty, not the observed fault value.
        adaptive=self.variant=='adaptive'
        self.change_count=self.change_count+1 if self.innovation_score>12 else 0
        if adaptive and self.change_count>=2:
            self.P+=np.eye(4)*.03;self.change_count=0;self.change_detections+=1
        prior=self.P/self.forgetting+np.eye(4)*(2e-7 if adaptive else 2e-6)
        K=np.linalg.solve(H@prior@H.T+noise,H@prior).T
        self.beta=np.clip(self.beta+K@np.clip(innovation,-1.,1.),.04,1.)
        I=np.eye(4)-K@H
        self.P=I@prior@I.T+K@noise@K.T;self.P=(self.P+self.P.T)/2
        self.eta=np.sqrt(self.beta)
        # Body-frame innovation is descriptive, never a privileged target.
        self.innovation=.8*self.innovation+.2*np.r_[float(force@avg[:,2]),avg.T@torque]/self.scale
        self.innovation-=.2*(self.geometry*(thrust*self.beta)[None,:]).sum(1)/self.scale
        self.updates+=1
    @property
    def uncertainty(self):
        # Covariance is exposed as an estimate; calibration is evaluated separately.
        return np.sqrt(np.maximum(np.diag(self.P),0))/np.maximum(2*self.eta,.1)

def make_estimator(controller,variant='original',**kw):
    if variant=='hybrid':return HybridEstimator(controller,**kw)
    return OriginalEstimator(controller,**kw) if variant=='original' else ImpulseEstimator(controller,variant=variant,**kw)

class HybridEstimator:
    """Original observer away from the boundary, impulse observer near it.

    Both filters use the SAME noisy velocity/rate increments. No independent
    sensor samples are averaged. The original path remains bit-identical until
    its measured eta enters [.72,.74]; the decision sees no fault metadata.
    """
    def __init__(self,controller,velocity_noise=.002,gyro_noise=.002):
        self.original=OriginalEstimator(controller,velocity_noise=velocity_noise,gyro_noise=gyro_noise)
        self.impulse=ImpulseEstimator(controller,variant='adaptive',velocity_noise=0.,gyro_noise=0.)
        self.velocity_noise=velocity_noise;self.gyro_noise=gyro_noise;self.reset()
    def reset(self,seed=0):
        self.original.reset(seed);self.impulse.reset(seed)
        self.shared_rng=np.random.default_rng(seed+18731);self.eta=np.ones(4);self.beta=np.ones(4)
        self.innovation=np.zeros(4);self.updates=0;self.weight=0.
    def update(self,before,after,command,dt,contact=False):
        old=self.original.updates;self.original.update(before,after,command,dt,contact)
        if self.original.updates==old:return
        dv=self.shared_rng.normal(0,self.velocity_noise*np.sqrt(2),3)
        dw=self.shared_rng.normal(0,self.gyro_noise*np.sqrt(2),3)
        noisy=after.copy();noisy[10:13]+=dv;noisy[13:16]+=rotation(after)@dw
        self.impulse.update(before,noisy,command,dt,contact)
        self.weight=float(np.clip((.74-float(np.min(self.original.eta)))/.02,0,1))
        if self.weight==0:self.eta=self.original.eta.copy()
        else:self.eta=(1-self.weight)*self.original.eta+self.weight*self.impulse.eta
        self.beta=self.eta**2;self.innovation=(1-self.weight)*self.original.innovation+self.weight*self.impulse.innovation
        self.updates=self.original.updates
    @property
    def uncertainty(self):
        return (1-self.weight)*self.original.uncertainty+self.weight*self.impulse.uncertainty+abs(self.original.eta-self.impulse.eta)*self.weight*(1-self.weight)
