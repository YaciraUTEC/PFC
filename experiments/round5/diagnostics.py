"""Read-only evaluation instrumentation; never exposed in policy observations."""
from collections import deque
import numpy as np
from .estimator import rotation

def attach_impact_diagnostics(env):
    monitor=env.monitor;original=monitor.update
    def update(state,goal,contact,route_complete,impact_state=None):
        was_empty=monitor.first is None
        result=original(state,goal,contact,route_complete,impact_state=impact_state)
        if was_empty and monitor.first is not None:
            incoming=state if impact_state is None else impact_state
            monitor.first.update(angular_velocity=incoming[13:16].tolist(),
                body_angular_velocity=(rotation(incoming).T@incoming[13:16]).tolist(),
                velocity=incoming[10:13].tolist(),position=incoming[:3].tolist())
        return result
    monitor.update=update

class InterventionTrace:
    def __init__(self):self.recent=deque(maxlen=48);self.before_contact=None
    def update(self,env,info,requested):
        state=env.core.obs_raw[0];body=rotation(state).T@state[13:16]
        self.recent.append(dict(requested=bool(requested[5]>0),applied=bool(info['gate_active']),amplitude=float(info['applied_amplitude']),
            yaw_rate=float(abs(body[2])),eta_min=float(min(info['applied_effectiveness']))))
        if self.before_contact is None and env.monitor.first is not None:self.before_contact=list(self.recent)
    def summary(self):
        rows=self.before_contact
        if not rows:return dict(precontact_trace_available=False)
        return dict(precontact_trace_available=True,precontact_requested_on_fraction=float(np.mean([r['requested'] for r in rows])),
            precontact_applied_on_fraction=float(np.mean([r['applied'] for r in rows])),
            precontact_mean_amplitude=float(np.mean([r['amplitude'] for r in rows])),
            precontact_max_body_yaw_rate=float(max(r['yaw_rate'] for r in rows)),
            precontact_min_applied_eta=float(min(r['eta_min'] for r in rows)))
