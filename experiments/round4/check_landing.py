import json
from pathlib import Path
import numpy as np
import pybullet as p
from .env import LandingEnv
from .evaluate import OUT
from .landing import LandingMonitor
records=[]
for label,height,vz in [('gentle',.016,-.10),('hard',.05,-1.0),('very_hard',.08,-2.0)]:
    e=LandingEnv(condition='exact',fixed_level=0.)
    try:
        e.reset(seed=450100);world=e.core._env;body=int(world.DRONE_IDS[0]);goal=e.core.punto_B.copy()
        p.resetBasePositionAndOrientation(body,[goal[0],goal[1],height],[0,0,0,1],physicsClientId=world.CLIENT)
        p.resetBaseVelocity(body,[0,0,vz],[0,0,0],physicsClientId=world.CLIENT)
        p.performCollisionDetection(physicsClientId=world.CLIENT)
        world._updateAndStoreKinematicInformation();e.core.obs_raw=world._getDroneStateVector(0)[None]
        e.contact=e._contacts();assert not e.contact
        monitor=LandingMonitor();first_result=None
        for tick in range(100):
            obs,contact,incoming=e._impact_step(np.zeros(4));e.core.obs_raw=obs;e.contact=contact
            success,bad=monitor.update(obs[0],goal,contact,True,impact_state=incoming)
            if monitor.first and first_result is None:first_result=dict(post_contact_vertical_speed=abs(float(obs[0,12])),first_contact=monitor.first.copy(),hard_rejected=bool(bad))
            if success or bad:break
        assert first_result is not None
        assert bool(bad)==(label!='gentle'),(label,first_result)
        assert bool(success)==(label=='gentle'),(label,first_result)
        records.append(dict(label=label,start_height=height,start_vertical_speed=vz,stable_landing=success,ticks=tick+1,**first_result))
    finally:e.close()
(OUT/'physical_landing_contract.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
