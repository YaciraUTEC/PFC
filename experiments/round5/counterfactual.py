"""Paired simulator branches with complete state/RNG restoration.
Candidates are held for12ticks, followed by the same zero-residual base policy.
Production labels continue to the remaining mission deadline; the complete terminal outcome is
used whenever reached. This is a diagnostic/teacher oracle, not deployment input.
"""
import numpy as np
import pybullet as p
from ..round4.counterfactual import snapshot,restore as _restore
from .env import ZERO,DT

def restore(e,snap):
    _restore(e,snap)
    p.setTimeStep(e.core._env.PYB_TIMESTEP,physicsClientId=e.core._env.CLIENT)
    p.setPhysicsEngineParameter(numSolverIterations=int(e.config['solver_iterations']),
        solverResidualThreshold=float(e.config['solver_residual']),physicsClientId=e.core._env.CLIENT)

def oracle_action(e,mode):
    old=e.config['intervention'];e.config['intervention']=mode
    try:return e.teacher_action()
    finally:e.config['intervention']=old

def rollout_cost(e,action,mode,horizon=384,burst=12):
    e.config['intervention']=mode;total=0.;terminal=None
    for t in range(min(horizon,1440-e.t)):
        _,reward,d,tr,i=e.step(action if t<burst else ZERO)
        s=e.core.obs_raw[0];dist=np.linalg.norm(e.reference-s[:3])
        total-=(.999**t)*float(reward)
        if d or tr:
            terminal=i['outcome'];break
    if terminal is None:
        s=e.core.obs_raw[0]
        total+=2*np.linalg.norm(e.core.punto_B[:2]-s[:2])**2+.1*np.dot(s[13:16],s[13:16])+.25*np.dot(s[10:13],s[10:13])
    return float(total),terminal

def choose(e,student=None,horizon=384,include_observer=True,explore=False,student_mode='residual'):
    modes=['residual','effectiveness'] if include_observer else ['residual']
    candidates=[]
    for mode in modes:
        teacher=oracle_action(e,mode)
        if teacher[5]>0:
            for alpha in [.5,1.]:
                a=teacher.copy();a[4]=2*((a[4]+1)/2)*alpha-1;candidates.append((mode,a,'teacher'))
        if student is not None and mode==student_mode and np.max(np.abs(student[:4]))>.002:
            a=np.asarray(student,np.float32).copy();a[5]=1.;candidates.append((mode,a,'student'))
        if explore:
            # Deterministic proposal, no random consumption before saving state.
            direction=np.sin(np.arange(4)*1.7+e.t*.13+e.t*.001)
            candidates.append((mode,np.r_[direction,-.8,1.].astype(np.float32),'probe'))
    if not candidates:
        return {m:ZERO.copy() for m in modes},[],dict(branched=False)
    snap=snapshot(e);records=[];best={m:(float('inf'),ZERO.copy()) for m in modes}
    try:
        restore(e,snap);base,base_outcome=rollout_cost(e,ZERO,'residual',horizon)
        for mode,a,source in candidates:
            restore(e,snap);e.config['intervention']=mode;features=e.candidate_features(a)
            cost,outcome=rollout_cost(e,a,mode,horizon)
            gain=base-cost;minimum=max(.015,.03*abs(base))
            records.append(dict(features=features.tolist(),gain=gain,beneficial=bool(gain>minimum),
                base_cost=base,candidate_cost=cost,mode=mode,action=a.tolist(),source=source,
                base_outcome=base_outcome,outcome=outcome,horizon=horizon,burst=12))
            if cost<best[mode][0] and gain>minimum:best[mode]=(cost,a.copy())
        return {m:v[1] for m,v in best.items()},records,dict(branched=True,base_cost=base,horizon=horizon,burst=12)
    finally:
        restore(e,snap);p.removeState(snap[0],physicsClientId=e.core._env.CLIENT)
