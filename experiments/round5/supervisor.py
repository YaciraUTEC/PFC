"""Observable landing readiness, with braking and no change to touchdown contract."""
from dataclasses import dataclass
import numpy as np

@dataclass
class DescentSupervisor:
    enabled:bool=True
    hold_ticks:int=3
    def reset(self):
        self.ready_ticks=0;self.height=None;self.paused_ticks=0;self.started=False
    def reference(self,state,goal,dt):
        if self.height is None:self.height=float(state[2])
        lateral=float(np.linalg.norm(state[:2]-goal[:2]))
        speed=float(np.linalg.norm(state[10:12]));spin=float(np.linalg.norm(state[13:16]))
        tilt=float(np.max(np.abs(state[7:9])))
        ready=lateral<.18 and speed<.20 and spin<.80 and tilt<.17
        self.ready_ticks=self.ready_ticks+1 if ready else 0
        permitted=not self.enabled or self.ready_ticks>=self.hold_ticks
        vertical=-.18 if permitted and self.height>-.025 else 0.
        if vertical:self.started=True
        else:
            self.paused_ticks+=1
            # Stop lowering the reference while angular/lateral motion is unsafe.
            # Do not leave an old reference below the aircraft during braking.
            if self.enabled and not ready:self.height=max(self.height,float(state[2]))
        self.height=max(-.025,self.height+dt*vertical)
        target=goal.copy();target[2]=self.height
        return target,np.array([0.,0.,vertical])
