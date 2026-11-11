"""Versioned strict touchdown/settling metric, independent of controller/reward.
Angle limits apply separately to each roll/pitch Euler coordinate."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class LandingCriteria:
    radius:float=.20
    first_vertical_speed:float=.25
    first_horizontal_speed:float=.25
    first_tilt:float=float(np.deg2rad(12))
    first_angular_speed:float=1.
    settled_speed:float=.10
    settled_tilt:float=float(np.deg2rad(10))
    settled_angular_speed:float=.5
    settled_height:float=.055
    hold_seconds:float=1.
    def dictionary(self):return asdict(self)

class LandingMonitor:
    def __init__(self,criteria=None,dt=1/48):
        self.criteria=criteria or LandingCriteria();self.dt=dt
        self.first=None;self.stable_ticks=0;self.peak_stable_ticks=0
    def update(self,state,goal,contact,route_complete,impact_state=None):
        c=self.criteria;v=state[10:13];omega=state[13:16]
        radius=float(np.linalg.norm(state[:2]-goal[:2]))
        tilt=float(np.max(np.abs(state[7:9])))
        if contact and route_complete and self.first is None:
            impact=state if impact_state is None else impact_state
            self.first=dict(vertical_speed=abs(float(impact[12])),horizontal_speed=float(np.linalg.norm(impact[10:12])),
                tilt=float(np.max(np.abs(impact[7:9]))),angular_speed=float(np.linalg.norm(impact[13:16])),
                radius=float(np.linalg.norm(impact[:2]-goal[:2])))
        gentle=self.first is not None and (
            self.first['vertical_speed']<=c.first_vertical_speed and
            self.first['horizontal_speed']<=c.first_horizontal_speed and
            self.first['tilt']<=c.first_tilt and self.first['angular_speed']<=c.first_angular_speed and
            self.first['radius']<=c.radius)
        stable=bool(contact and route_complete and gentle and radius<=c.radius and
            state[2]<=c.settled_height and np.linalg.norm(v)<=c.settled_speed and
            tilt<=c.settled_tilt and np.linalg.norm(omega)<=c.settled_angular_speed)
        self.stable_ticks=self.stable_ticks+1 if stable else 0
        self.peak_stable_ticks=max(self.peak_stable_ticks,self.stable_ticks)
        return bool(self.stable_ticks*self.dt>=c.hold_seconds-1e-9),bool(self.first is not None and not gentle)
