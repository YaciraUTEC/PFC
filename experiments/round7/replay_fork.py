"""Natural-history simulator forks by deterministic prefix replay.
PyBullet save/restore did not preserve the next natural physical step. This fork
never changes the caller and never uses restoreState. Each branch resets its own
physics world and replays the exact requested actuator history from the reset.
"""
import copy
import numpy as np
from ..round6.env import LandingEnv
from .common import silent


def checked_state(e):
    return dict(t=e.t,motor=e.motor,level=e.level,fault_step=e.fault_step,route_reached=e.route_reached,touchdown_seen=e.touchdown_seen,
        observation=e.frames[-1].copy(),physical=e.core.obs_raw[0].copy(),measured=e.measured.copy(),estimated_eta=e.estimated_eta.copy(),true_eta=e.true_eta.copy(),
        baseq=e.baseq.copy(),integral_position=e.c.integral_pos_e.copy(),integral_attitude=e.c.integral_rpy_e.copy(),sensor_rng=copy.deepcopy(e.sensor_rng.bit_generator.state),phase=e.descent.phase)

class ReplayFork:
    def __init__(self,e,history):
        assert len(history)==e.t,(len(history),e.t)
        self.prefix=[np.asarray(a,np.float32).copy() for a in history];self.seed=int(e.case_seed);self.level=float(e.level);self.expected=checked_state(e);self.checks=0;self.maximum_error=0.
        with silent():self.env=LandingEnv(history=e.history,controller=e.kind,condition=e.condition,landing_profile=e.landing_profile,guard=e.guard,sensor_noise=e.sensor_noise,config=copy.deepcopy(e.config))
    def restore(self):
        with silent():obs,_=self.env.reset(seed=self.seed,options=dict(level=self.level))
        for action in self.prefix:
            obs,_,done,truncated,_=self.env.step(action)
            if done or truncated:raise AssertionError('Replay prefix terminated before the query state')
        actual=checked_state(self.env)
        for key,value in self.expected.items():
            if isinstance(value,np.ndarray):
                error=float(np.abs(actual[key]-value).max());self.maximum_error=max(self.maximum_error,error)
                assert error==0.,('Natural replay state mismatch',key,error)
            else:assert actual[key]==value,('Natural replay state mismatch',key,actual[key],value)
        self.checks+=1;return self.env
    def close(self):self.env.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
