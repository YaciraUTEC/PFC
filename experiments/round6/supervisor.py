"""Observable, continuous reference governor with a terminal braking envelope.
No true fault or simulator derivatives enter decisions. This is a conservative
kinematic governor, not a proof of nonlinear recursive feasibility.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class RecoveryDescent:
    enabled: bool = True
    max_speed: float = .18
    braking_height: float = .22
    rate_limit: float = .65
    acceleration: float = .25
    capacity_envelope: bool = False
    planned_yaw_acceleration: float = 3.
    def __post_init__(self): self.reset()
    def reset(self):
        self.height=None; self.ready_ticks=0; self.paused_ticks=0; self.started=False
        self.phase='approach'; self.elapsed=0.; self.no_progress=0.; self.velocity=0.
        self.filtered=None; self.phase_ticks={}; self.recovery_events=0
        self.last_height=None; self.yaw_acceleration=.2; self.margin=0.; self.spin_time=0.
        self.braking_acceleration=0.;self.predicted_recovery_height=0.;self.recovery_feasible=True
    def bind(self,controller,get_eta): self.controller=controller; self.get_eta=get_eta
    def reference(self,state,goal,dt):
        if self.height is None: self.height=float(state[2]); self.last_height=float(state[2])
        self.elapsed+=dt
        measured=np.r_[state[10:13],state[13:16]]
        alpha=-np.expm1(-dt/.06)
        self.filtered=measured.copy() if self.filtered is None else self.filtered+alpha*(measured-self.filtered)
        lateral=float(np.linalg.norm(state[:2]-goal[:2])); horizontal=float(np.linalg.norm(self.filtered[:2]))
        spin=float(np.linalg.norm(self.filtered[3:])); tilt=float(np.max(np.abs(state[7:9])))
        c=getattr(self,'controller',None); eta=np.asarray(self.get_eta()) if hasattr(self,'get_eta') else np.ones(4)
        if c is not None:
            capacity=c.KF*(c._command_max*eta)**2
            self.margin=4*float(capacity.min())/c.GRAVITY-1.
            # Static balanced hover budget is deliberately not called a reachability guarantee.
            yaw_capacity=max(0.,4*capacity.min()-c.GRAVITY)*c.KM/c.KF
            self.yaw_acceleration=max(.08,min(6.,yaw_capacity/max(c.J[2,2],1e-12)))
        self.spin_time=max(0.,spin-self.rate_limit)/self.yaw_acceleration
        altitude=max(0.,float(state[2])-.035)
        current_down=max(0.,-float(self.filtered[2]))
        stop_height=current_down**2/(2*self.acceleration)+.035
        # Retain enough time/height to brake yaw before terminal descent.
        envelope=max(0.,altitude-self.braking_height-stop_height)
        yaw_limited=envelope/max(self.spin_time+.8,.8)
        lateral_ready=lateral<.16 and horizontal<.18 and tilt<.16
        if c is not None:
            c.landing_active=True;c.landing_braking_allowed=lateral_ready
        aligned=lateral_ready and spin<self.rate_limit
        self.ready_ticks=self.ready_ticks+1 if aligned else 0
        if not lateral_ready:
            wanted=0.; self.phase='recover'
        elif altitude>self.braking_height:
            wanted=-self.max_speed if spin<self.rate_limit else -min(self.max_speed,max(.0,yaw_limited))
            self.phase='descend' if wanted<-.005 else 'brake'
        elif spin<self.rate_limit and horizontal<.15 and tilt<.14:
            wanted=-min(.10,max(.045,np.sqrt(max(0.,2*.18*altitude))))
            self.phase='terminal'
        else:
            wanted=0.; self.phase='brake'
        if self.capacity_envelope and c is not None:
            # Piecewise braking envelope: angular braking may reduce collective;
            # after the spin is arrested, balanced thrust brakes vertical speed.
            from scipy.spatial.transform import Rotation
            body_rotation=Rotation.from_quat(state[3:7]).as_matrix()
            omega=body_rotation.T@self.filtered[3:]
            alpha=self.planned_yaw_acceleration
            torque=np.array([0.,0.,-c.J[2,2]*np.clip(12*omega[2],-alpha,alpha)])
            A=c._force_geometry*(c.KF*c.HOVER_RPM**2*eta**2)[None,:]
            force_max=4*float(capacity.min())
            if np.linalg.matrix_rank(A)==4:
                inv=np.linalg.inv(A);b=inv[:,1:]@torque;v=inv[:,0]
                lower=0.;upper=np.inf;hi=(c._command_max/c.HOVER_RPM)**2
                for aa,bb in zip(v,b):
                    if abs(aa)>1e-12:
                        limits=sorted([-bb/aa,(hi-bb)/aa]);lower=max(lower,limits[0]);upper=min(upper,limits[1])
                force_max=max(0.,upper) if upper>=lower else 0.
            cosine=max(.0,float(body_rotation[2,2]))
            reserve=getattr(c,'_thrust_reserve',0.)
            force_max*=1-reserve
            down_acceleration=max(0.,9.8-force_max*cosine/c.MASS)
            after_acceleration=max(0.,(4*float(capacity.min())*(1-reserve)*cosine/c.MASS)-9.8)
            self.braking_acceleration=after_acceleration
            angular_time=max(0.,spin-self.rate_limit)/alpha
            max_allowed=0.;self.recovery_feasible=False
            for proposed in np.linspace(self.max_speed,0.,19):
                initial=max(current_down,float(proposed))
                final_down=initial+down_acceleration*angular_time
                angular_height=initial*angular_time+.5*down_acceleration*angular_time**2
                if final_down<=.20:vertical_height=0.
                elif after_acceleration>1e-6:vertical_height=(final_down**2-.20**2)/(2*after_acceleration)
                else:vertical_height=np.inf
                required=angular_height+vertical_height
                if required<=altitude:
                    self.recovery_feasible=True;max_allowed=float(proposed);self.predicted_recovery_height=float(required);break
            if not self.recovery_feasible:self.phase='recovery';wanted=0.
            elif wanted<0:wanted=-min(-wanted,max_allowed)
        if self.last_height-float(state[2])>.01:
            self.no_progress=0.; self.last_height=float(state[2])
        elif self.elapsed>2.:self.no_progress+=dt
        if self.no_progress>2.:
            # Explicit recovery phase; never force unsafe touchdown on a timer.
            self.phase='recovery'; self.recovery_events+=int(self.no_progress-dt<=2.)
        self.velocity+=np.clip(wanted-self.velocity,-self.acceleration*dt,self.acceleration*dt)
        if not lateral_ready and current_down>.03:
            # Stop the descending reference within a bounded braking distance.
            self.height=min(self.height,float(state[2])+stop_height)
        old_height=self.height
        self.height=max(-.025,self.height+self.velocity*dt)
        # Bound tracking lag without ratcheting the reference upward every pause.
        self.height=min(old_height,max(self.height,float(state[2])-.12))
        if self.velocity<-.005:self.started=True
        else:self.paused_ticks+=1
        self.phase_ticks[self.phase]=self.phase_ticks.get(self.phase,0)+1
        target=goal.copy(); target[2]=self.height
        return target,np.array([0.,0.,self.velocity])
