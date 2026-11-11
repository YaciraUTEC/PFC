"""Motion/command-only online thrust-effectiveness RLS (beta = eta RPM squared)."""
import numpy as np
import pybullet as p

def rotation(s):
    return np.asarray(p.getMatrixFromQuaternion(s[3:7])).reshape(3,3)

class EffectivenessEstimator:
    def __init__(self,controller,forgetting=.96,velocity_noise=.002,gyro_noise=.002):
        self.geometry=controller._force_geometry.copy()
        self.kf=controller.KF;self.mass=controller.MASS;self.J=controller.J.copy()
        self.scale=controller.wrench_scale.copy()
        self.forgetting=forgetting;self.velocity_noise=velocity_noise;self.gyro_noise=gyro_noise
        self.reset()
    def reset(self,seed=0):
        self.beta=np.ones(4);self.P=np.eye(4)*.2
        self.eta=np.ones(4);self.innovation=np.zeros(4);self.updates=0
        self.rng=np.random.default_rng(seed+18731)
    def update(self,before,after,command,dt,contact=False):
        # These are the only runtime inputs: measured states, commanded RPM, dt/contact.
        if contact or min(before[2],after[2])<.08 or not np.isfinite(after).all():return
        R0,R1=rotation(before),rotation(after);z=(R0[:,2]+R1[:,2])/2
        dv=after[10:13]-before[10:13]+self.rng.normal(0,self.velocity_noise*np.sqrt(2),3)
        w0=R0.T@before[13:16];w1=R1.T@after[13:16]
        dw=w1-w0+self.rng.normal(0,self.gyro_noise*np.sqrt(2),3)
        w=(w0+w1)/2
        force=self.mass*np.dot(dv/dt+np.array([0,0,9.8]),z)
        torque=self.J@(dw/dt)+np.cross(w,self.J@w)
        y=np.r_[force,torque]/self.scale
        H=self.geometry*(self.kf*np.asarray(command)**2)[None,:]/self.scale[:,None]
        # Measurement covariance is conservative about finite differences/model error.
        noise=np.diag(np.array([.025,.035,.035,.05])**2)
        innovation=y-H@self.beta
        self.innovation=.8*self.innovation+.2*innovation
        prior=self.P/self.forgetting
        K=np.linalg.solve(H@prior@H.T+noise,H@prior).T
        self.beta=np.clip(self.beta+K@np.clip(innovation,-.8,.8),.04,1.)
        I=np.eye(4)-K@H
        self.P=I@prior@I.T+K@noise@K.T+np.eye(4)*2e-5
        self.P=(self.P+self.P.T)/2
        self.eta=.5*self.eta+.5*np.sqrt(self.beta)
        self.updates+=1
    @property
    def uncertainty(self):
        return np.sqrt(np.maximum(np.diag(self.P),0))/np.maximum(2*self.eta,.1)
