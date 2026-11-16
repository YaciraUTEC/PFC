"""Exact simulator branching, preserving shared Python graph and observer RNG."""
import copy
import pybullet as p

class Snapshot:
    def __init__(self,e):
        self.e=e;sim=e.core._env
        self.client=sim.CLIENT;self.sid=p.saveState(physicsClientId=self.client)
        # One deepcopy operation preserves aliases between controller and observers.
        self.payload=copy.deepcopy(dict(outer={k:v for k,v in e.__dict__.items() if k not in ['core','c','physics_model']},
            core={k:v for k,v in e.core.__dict__.items() if k not in ['_env','pid_controller']},
            sim={k:v for k,v in sim.__dict__.items() if not callable(v)},controller=e.c))
    def restore(self):
        e=self.e;core=e.core;sim=core._env;d=copy.deepcopy(self.payload)
        p.restoreState(self.sid,physicsClientId=self.client)
        e.__dict__.clear();e.__dict__.update(d['outer']);e.core=core;e.c=d['controller'];e.physics_model=e.c
        core.__dict__.clear();core.__dict__.update(d['core']);core._env=sim;core.pid_controller=e.c
        sim.__dict__.clear();sim.__dict__.update(d['sim'])
        e.descent.bind(e.c,lambda:e.estimated_eta)
        p.setTimeStep(sim.PYB_TIMESTEP,physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=int(e.config['solver_iterations']),solverResidualThreshold=float(e.config['solver_residual']),physicsClientId=self.client)
        return e
    def close(self):p.removeState(self.sid,physicsClientId=self.client)
    def __enter__(self):return self
    def __exit__(self,*exc):
        try:self.restore()
        finally:self.close()
