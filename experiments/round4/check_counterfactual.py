import numpy as np
from .env import LandingEnv
from .counterfactual import snapshot,restore,choose
e=LandingEnv(condition='biased')
try:
 o,_=e.reset(seed=410101,options=dict(level=.25))
 while e.fault_step is None or e.t-e.fault_step<8:o,*_=e.step(e.teacher_action())
 snap=snapshot(e);action=e.teacher_action();restore(e,snap);one=e.step(action);restore(e,snap);two=e.step(action)
 assert np.array_equal(one[0],two[0]),np.max(np.abs(one[0]-two[0]))
 restore(e,snap);before=e._state().copy();t=e.t;chosen,detail=choose(e,8)
 assert e.t==t and np.array_equal(e._state(),before)
 import pybullet as p
 p.removeState(snap[0],physicsClientId=e.core._env.CLIENT)
 print('Full-state replay and counterfactual restoration passed',detail,flush=True)
finally:e.close()
