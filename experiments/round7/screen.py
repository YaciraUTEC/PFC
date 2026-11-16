"""Actuator-aware comparative landing screen, NOT a reachability certificate.
The short held-input prediction and a coupled force/torque braking envelope
screen adverse proposals relative to OFF. A validated terminal backup is not
assumed from this approximation. Full flights decide whether it is useful.
"""
import torch
from torch import nn
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel
from .common import silent

class LandingScreen(nn.Module):
    def __init__(self,allowance=0.,uncertainty_multiplier=1.,braking_time=.6):
        super().__init__();self.allowance=float(allowance);self.uncertainty_multiplier=float(uncertainty_multiplier);self.braking_time=float(braking_time)
        with silent():c=DSLPIDControl(DroneModel.CF2X)
        self.mass=c.MASS;self.gravity=9.8;self.qmax=(c._command_max/c.HOVER_RPM)**2;self.qscale=.2*self.qmax
        self.register_buffer('geometry',torch.as_tensor(c._force_geometry*c.KF*c.HOVER_RPM**2,dtype=torch.float64))
        self.register_buffer('inertia',torch.as_tensor(c.J.diagonal().copy(),dtype=torch.float64))
    def penalty(self,obs,delta,duration):
        n=len(obs);dtype=obs.dtype;device=obs.device
        q=(obs[:,64:68]*self.qmax+self.qscale*delta).clamp(0,self.qmax)
        eta=(1-.25*obs[:,31:35]).clamp(.2,1.)
        uncertainty=(obs[:,48:52]*.1).clamp(0,.15)*self.uncertainty_multiplier
        scenarios=torch.stack([(eta-uncertainty).clamp(.2,1.),eta,(eta+uncertainty).clamp(.2,1.)],1)
        A=self.geometry[None,None]*scenarios[:,:,None,:].square()
        force_torque=(A@q[:,None,:,None]).squeeze(-1)
        rotation=obs[:,6:15].reshape(n,3,3);cos=rotation[:,2,2].clamp(.05,1.)
        omega=obs[:,15:18]*10;velocity=obs[:,3:6]*2;height=obs[:,40]*1.2
        dt=duration.clamp(1,12)/48
        alpha=(force_torque[:,:,1:]-torch.cross(omega,omega*self.inertia,dim=-1)[:,None,:])/self.inertia
        future_omega=omega[:,None,:]+dt[:,None,None]*alpha
        future_cos=(cos[:,None]-omega[:,:2].norm(dim=-1)[:,None]*dt[:,None]-.5*alpha[:,:,:2].norm(dim=-1)*dt[:,None].square()).clamp(.05,1.)
        acceleration=force_torque[:,:,0]*cos[:,None]/self.mass-self.gravity
        vz=velocity[:,2,None]+acceleration*dt[:,None]
        h=height[:,None]+velocity[:,2,None]*dt[:,None]+.5*acceleration*dt[:,None].square()
        vdown=(-vz).clamp_min(0)
        # Find achievable collective while reserving the torque required to brake.
        desired=-self.inertia*future_omega/self.braking_time+torch.cross(future_omega,future_omega*self.inertia,dim=-1)
        inverse=torch.linalg.inv(A);v=inverse[:,:,:,0];b=(inverse[:,:,:,1:]@desired[:,:,:,None]).squeeze(-1)
        upper=((self.qmax-b)/v.clamp_min(1e-9)).amin(-1);lower=(-b/v.clamp_min(1e-9)).amax(-1).clamp_min(0)
        collective=torch.where(upper>=lower,upper.clamp_min(0),torch.zeros_like(upper))
        brake_acc=(collective*future_cos/self.mass-self.gravity).clamp_min(.005)
        stop_height=(vdown.square()-.20**2).clamp_min(0)/(2*brake_acc)
        clearance=(stop_height+.035-h).clamp_min(0)
        tilt=future_cos.acos();angular=future_omega.norm(dim=-1)
        horizontal=velocity[:,:2].norm(dim=-1)[:,None]
        near=(h<.075).to(dtype)
        contact=near*((vdown-.23).clamp_min(0)*5+(horizontal-.23).clamp_min(0)*3+(tilt-.19).clamp_min(0)*2+(angular-.9).clamp_min(0))
        travel=((h-.035).clamp_min(0)/.18)+(angular-.5).clamp_min(0)/3+1.+vz.clamp_min(0)*2
        remaining=obs[:,72,None]*30-dt[:,None]
        late=(travel-remaining).clamp_min(0)/5
        early=near*(1-obs[:,56,None])*2
        penalty=10*clearance+contact+late+early
        return penalty.amax(1)
    def forward(self,obs,delta,duration):
        off=torch.zeros_like(delta)
        if not bool(torch.any(delta.abs().amax(-1)>.002)):return off
        candidate=self.penalty(obs,delta,duration);base=self.penalty(obs,off,duration)
        permitted=torch.isfinite(candidate)&(candidate<=base+self.allowance+1e-10)
        return torch.where(permitted[:,None],delta,off)
