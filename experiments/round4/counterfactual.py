"""Short real-simulator branches with complete Python/Bullet state restoration."""
import copy
import numpy as np
import pybullet as p
from .env import ZERO

def snapshot(e):
    bullet=e.core._env
    sid=p.saveState(physicsClientId=bullet.CLIENT)
    outer=copy.deepcopy({k:v for k,v in e.__dict__.items() if k not in ['core','c','physics_model']})
    core=copy.deepcopy({k:v for k,v in e.core.__dict__.items() if k not in ['_env','pid_controller']})
    sim=copy.deepcopy({k:v for k,v in bullet.__dict__.items() if not callable(v)})
    return sid,outer,core,sim,copy.deepcopy(e.c)

def restore(e,snap):
    sid,outer,core,sim,c=snap;ec=e.core;bullet=ec._env
    p.restoreState(sid,physicsClientId=bullet.CLIENT)
    e.__dict__.clear();e.__dict__.update(copy.deepcopy(outer));e.core=ec;e.c=copy.deepcopy(c)
    e.physics_model=e.c
    ec.__dict__.clear();ec.__dict__.update(copy.deepcopy(core));ec._env=bullet;ec.pid_controller=e.c
    bullet.__dict__.clear();bullet.__dict__.update(copy.deepcopy(sim))

def choose(e,horizon=24):
    initial=e.teacher_action()
    if initial[4]<-.99:return ZERO.copy(),dict(gains=[0.],selected=0.,branched=False)
    snap=snapshot(e);costs=[];amplitudes=[0.,.25,.5,1.]
    try:
        for alpha in amplitudes:
            restore(e,snap);total=0.
            for t in range(horizon):
                a=e.teacher_action()
                a[4]=2*((a[4]+1)/2)*alpha-1
                _,r,d,tr,i=e.step(a)
                total+=-i['manual_style']+.03*((a[4]+1)/2)**2
                if d or tr:
                    total+=0 if i['mission_success'] else 30.
                    break
            s=e.core.obs_raw[0]
            total+=2*np.linalg.norm(e.reference-s[:3])**2+.2*np.linalg.norm(s[10:13]-e.target_velocity)**2
            costs.append(total)
        best=int(np.argmin(costs))
        if costs[0]-costs[best]<max(.02,.03*abs(costs[0])):best=0
        alpha=amplitudes[best]
        result=initial.copy();result[4]=2*((initial[4]+1)/2)*alpha-1
        if alpha==0:result=ZERO.copy()
        return result,dict(costs=costs,selected=alpha,branched=True,horizon=horizon)
    finally:
        restore(e,snap);p.removeState(snap[0],physicsClientId=e.core._env.CLIENT)
