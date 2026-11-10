from pathlib import Path
import types,time,json
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.RobustDSLPIDControl import RobustDSLPIDControl
from .common import OUT,save

def main():
    revision=OUT.parents[2]/'revision'
    module=types.ModuleType('gym_pybullet_drones.control._BeforeAPI');module.__package__='gym_pybullet_drones.control'
    exec(compile((revision/'DSLPIDControl_before_api.py').read_text(),str(revision/'DSLPIDControl_before_api.py'),'exec'),module.__dict__)
    oldmodern=module.DSLPIDControl
    robust=types.ModuleType('robust_before');robust.__dict__['DSLPIDControl']=oldmodern
    src=(revision/'RobustDSLPIDControl_before_api.py').read_text().replace('from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl','')
    exec(compile(src,str(revision/'RobustDSLPIDControl_before_api.py'),'exec'),robust.__dict__)
    rng=np.random.default_rng(74851);results={}
    for label,oldcls,newcls in [('updated',oldmodern,DSLPIDControl),('robust',robust.RobustDSLPIDControl,RobustDSLPIDControl)]:
        a=oldcls(DroneModel.CF2X);b=newcls(DroneModel.CF2X);maximum=0.;start=time.time()
        if label=='robust':a.fault_aware=b.fault_aware=True
        for i in range(1000):
            if i%100==0:
                a.reset();b.reset();eta=rng.uniform(.4,1.,4);a.set_motor_effectiveness(eta);b.set_motor_effectiveness(eta)
            kw=dict(control_timestep=1/48,cur_pos=np.r_[rng.normal(0,.3,2),1.],cur_quat=p.getQuaternionFromEuler(rng.normal(0,.2,3)),cur_vel=rng.normal(0,.3,3),cur_ang_vel=rng.normal(0,.8,3),target_pos=np.array([0.,0.,1.]))
            ra=a.computeControl(**kw)[0];rb=b.computeControl(**kw)[0]
            maximum=max(maximum,float(np.max(np.abs(ra-rb))))
            np.testing.assert_allclose(ra,rb,rtol=0,atol=1e-9)
        results[label]=dict(requests=1000,max_rpm_difference=maximum,seconds=time.time()-start)
    save(OUT/'api_default_parity.json',results);print(results,flush=True)
if __name__=='__main__':main()