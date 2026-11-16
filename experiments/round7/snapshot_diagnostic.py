"""Diagnose no-op restoration against natural execution, including future contact."""
import json
import numpy as np
import pybullet as p
import torch
from .common import OUT,config,read,save,silent
from .policy import TransformerPolicy
from .episode import infer
from .snapshot import Snapshot
from .branches import continuation
from ..round6.env import LandingEnv


def main():
    torch.set_num_threads(1);spec=read(OUT/'h3_filter_ablation_conclusion.json')['selected']['spec'];m=TransformerPolicy.from_spec(spec);state=None
    with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=2.))
    rows=[]
    try:
        with silent():obs,_=e.reset(seed=729811,options=dict(level=.295))
        for _ in range(80):a,state=infer(m,obs,state);obs,_,d,t,_=e.step(a);assert not(d or t)
        a,state=infer(m,obs,state);initial_params=p.getPhysicsEngineParameters(physicsClientId=e.core._env.CLIENT)
        actual_commands=[];impact=e._impact_step
        def observe_impact(rpm):
            actual_commands.append(np.asarray(rpm).copy());return impact(rpm)
        e._impact_step=observe_impact
        with Snapshot(e) as snap:
            def take(label):
                client=e.core._env.CLIENT;body=int(e.core._env.DRONE_IDS[0]);before=np.r_[*p.getBasePositionAndOrientation(body,physicsClientId=client),*p.getBaseVelocity(body,physicsClientId=client)]
                obs,_,d,t,info=e.step(a)
                row=dict(label=label,physics_kind=e.core._env.PHYSICS.value,physical_before=before,physical_after=e.core.obs_raw[0].copy(),actual_scaled_rpm=actual_commands[-1].copy(),obs=obs[-1].copy(),rpm=info['motor_command_rpm'].copy(),projected=info['projected_motor_delta'].copy(),true_eta=e.true_eta.copy(),estimated_eta=e.estimated_eta.copy(),engine=p.getPhysicsEngineParameters(physicsClientId=e.core._env.CLIENT),controller_id=id(e.c),estimator_controller_is_correct=getattr(e.estimator,'model',e.c) is e.c)
                rows.append(row);return row
            natural=take('natural')
            snap.restore();take('one_restore')
            snap.restore();future=continuation(e);snap.restore();take('after_terminal_future')
            snap.restore();continuation(e);snap.restore();sim=e.core._env;p.performCollisionDetection(physicsClientId=sim.CLIENT);p.getLinkStates(int(sim.DRONE_IDS[0]),list(range(5)),computeForwardKinematics=1,physicsClientId=sim.CLIENT);take('after_future_kinematic_refresh')
            snap.restore();start=natural['physical_before'];sim=e.core._env;body=int(sim.DRONE_IDS[0]);p.resetBasePositionAndOrientation(body,start[:3],start[3:7],physicsClientId=sim.CLIENT);p.resetBaseVelocity(body,linearVelocity=start[7:10],angularVelocity=start[10:13],physicsClientId=sim.CLIENT);p.performCollisionDetection(physicsClientId=sim.CLIENT);take('explicit_base_reset')
            for row in rows:
                row['actual_scaled_rpm_error']=float(np.abs(row['actual_scaled_rpm']-natural['actual_scaled_rpm']).max());row['physical_before_error']=float(np.abs(row['physical_before']-natural['physical_before']).max());row['physical_after_error']=float(np.abs(row['physical_after']-natural['physical_after']).max());row['observation_max_error']=float(np.abs(row['obs']-natural['obs']).max());row['rpm_max_error']=float(np.abs(row['rpm']-natural['rpm']).max());print('NOOP_SNAPSHOT',row['label'],row['observation_max_error'],row['rpm_max_error'],row['actual_scaled_rpm_error'],row['physical_before_error'],row['physical_after_error'],row['physics_kind'],flush=True)
            save(OUT/'snapshot_natural_diagnostic.json',dict(initial_parameters=initial_params,rows=rows))
    finally:e.close()
if __name__=='__main__':main()
