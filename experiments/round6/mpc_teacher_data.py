"""Observable slow NMPC teacher, distilled as bounded residuals over fast DSLPID.
The teacher sees exactly the student's measured state and estimated eta. It is
queried after an observable fault indication; no true fault identity/effectiveness.
"""
import os,json,time,contextlib,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.NMPCINDIControl import NMPCINDIControl
from gym_pybullet_drones.utils.enums import DroneModel
from .env import LandingEnv,ZERO
from .evaluate import OUT,save
from .train import selected_config
from .data_audit import audit

LEVELS=[.28,.29,.295,.296]
def collect_route(index,config,options,mode="immediate"):
    tag=f'route_mpc_rescue_{index:03d}' if mode=='rescue' else f'route_mpc_{index:03d}'
    path=OUT/'data'/(tag+'.npz');meta=path.with_suffix('.json')
    if path.exists() and meta.exists():return str(meta)
    start=time.time();seed=(621100 if mode=='rescue' else 621000)+index;arrays=[];records=[];offset=0
    with open(os.devnull,'w') as f,contextlib.redirect_stdout(f):e=LandingEnv(condition='estimated',config=dict(config,gate_mode='learned'))
    teacher=NMPCINDIControl(DroneModel.CF2X,**options)
    try:
        for level in LEVELS:
            for noise in [1.,2.]:
                flights=[]
                for behavior in ['baseline','nmpc_residual']:
                    e.config['noise_multiplier']=noise;e._sensor_sigma=.002*noise
                    obs,_=e.reset(seed=seed,options=dict(level=level));teacher.reset();latched=False
                    previous_command=np.full(4,e.c.HOVER_RPM);calls=0;solve=[];clipped=[];bad_count=0;clear_count=0;active=False;activation_steps=[]
                    columns={k:[] for k in ['s','ns','executed','reward','done','motor_command_rpm','projected_motor_delta']}
                    while True:
                        action=ZERO.copy();latched=latched or bool(np.min(e.estimated_eta)<.98)
                        if mode=='rescue':
                            yaw=abs(float((Rotation.from_quat(e.measured[3:7]).as_matrix().T@e.measured[13:16])[2]))
                            bad_count=bad_count+1 if latched and yaw>.8 else 0
                            clear_count=clear_count+1 if yaw<.35 and np.linalg.norm(e.measured[10:12])<.20 else 0
                            if not active and bad_count>=4:
                                active=True;teacher.reset();activation_steps.append(e.t)
                            if active and clear_count>=24:active=False;bad_count=0
                            teacher_on=active
                        else:teacher_on=latched or e.route_reached
                        if behavior=='nmpc_residual' and teacher_on and not e.touchdown_seen:
                            teacher.set_motor_effectiveness(e.estimated_eta)
                            teacher.set_motor_speeds(np.clip(previous_command*e.estimated_eta,0.,teacher._command_max))
                            rpm,_,_=teacher.computeControl(**e._arguments)
                            raw=((rpm/teacher.HOVER_RPM)**2-e.baseq)/e.qscale;delta=np.clip(raw,-1,1)
                            amount=float(np.max(np.abs(delta)))
                            if amount>.002:action=np.r_[delta/amount,2*amount-1,1.].astype(np.float32)
                            calls+=1;solve.append(teacher.last_diagnostics.get('solve_seconds',0.));clipped.append(float(np.any(np.abs(raw)>1)))
                        columns['s'].append(obs[-1].copy());obs,reward,d,t,info=e.step(action);previous_command=info['motor_command_rpm'].copy()
                        for k,value in [('ns',obs[-1].copy()),('executed',info['executed_action'].copy()),('reward',reward),('done',d or t),('motor_command_rpm',previous_command.copy()),('projected_motor_delta',info['projected_motor_delta'].copy())]:columns[k].append(value)
                        if d or t:break
                    flights.append((behavior,columns,info,dict(teacher_calls=calls,mean_solve_seconds=float(np.mean(solve)) if solve else 0.,clipped_fraction=float(np.mean(clipped)) if clipped else 0.,activation_steps=activation_steps)))
                baseline=flights[0][2]['stable_landing']
                for behavior,c,info,extra in flights:
                    n=len(c['s']);good=bool(info['stable_landing'] and (behavior=='baseline' or not baseline))
                    arrays.append(dict(s=np.asarray(c['s'],np.float32),ns=np.asarray(c['ns'],np.float32),executed=np.asarray(c['executed'],np.float32),
                        reward=np.asarray(c['reward'],np.float32),done=np.asarray(c['done'],bool),a=np.asarray(c['executed'],np.float32) if good else np.tile(ZERO,(n,1)),
                        valid=np.full(n,good,bool),train=np.ones(n,bool),motor_command_rpm=np.asarray(c['motor_command_rpm'],np.float32),projected_motor_delta=np.asarray(c['projected_motor_delta'],np.float32)))
                    records.append(dict(start=offset,stop=offset+n,seed=seed,level=level,condition='estimated',noise=noise,behavior=behavior,
                        stable_landing=info['stable_landing'],baseline_success=baseline,expert=good,split='train',outcome=info['outcome'],**extra));offset+=n
    finally:e.close()
    np.savez_compressed(path,**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    save(meta,dict(records=records,seconds=time.time()-start,config=config,teacher_options=options,teacher_inputs='measured_state_and_estimated_eta_only',source='observable_nmpc_residual_demonstrations',teacher_mode=mode))
    return str(meta)
def main():
    start=time.time();control=json.loads((OUT/'phase1_conclusion.json').read_text());name=control['best_slow_nmpc'];options=control['all_configs'][name]['nmpc_options']
    config=selected_config()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(collect_route,i,config,options) for i in range(8)]):print('MPC_EXPERT_DATA',f.result(),flush=True)
    result=audit()
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'data').glob('route_*.npz'))}
    fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    save(OUT/'dataset_fingerprint.json',dict(sha256=fingerprint,files=files))
    save(OUT/'expert_data_complete.json',dict(seconds=time.time()-start,teacher=name,teacher_true_eta=False,passed=True,audit=result))
if __name__=='__main__':main()
