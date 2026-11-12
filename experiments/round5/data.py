"""Route-separated common data, negatives, paired labels and student states."""
import argparse,json,time,os,contextlib,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .env import LandingEnv,ZERO,LEVELS,DIM
from .counterfactual import choose,oracle_action
from .evaluate import OUT
from ..run import save

def collect_route(index,config,dagger_models=None):
    phase='initial' if not dagger_models else 'dagger'
    directory=OUT/'data'/phase;target=directory/f'route_{index:03d}.npz';meta=target.with_suffix('.json')
    if target.exists() and meta.exists():return str(meta)
    seed=526000+index if index<48 else 527000+index-48
    start=time.time();arrays=[];episodes=[];branches=[];offset=0
    with open(os.devnull,'w') as silent,contextlib.redirect_stdout(silent):e=LandingEnv(config=config)
    models=[None]
    if dagger_models:
        from .models import load_agent
        models=[load_agent(item['path']) for item in dagger_models]
    try:
        for mi,model in enumerate(models):
            conditions=['exact','estimated'] if phase=='initial' else ['estimated']
            levels=LEVELS if phase=='initial' else [.02,.25,.28,.29,.295]
            for condition in conditions:
                e.condition=condition
                for level in levels:
                    e.config['intervention']=dagger_models[mi]['mode'] if model is not None else 'effectiveness' if index%3==2 else 'residual'
                    e.config['fault_clock']=[None,4.,6.,8.][index%4]
                    obs,_=e.reset(seed=seed,options=dict(level=level));state=None;columns={k:[] for k in ['s','a','observer','valid','observer_valid','eta','executed','reward','task','style','done','ns']}
                    held=ZERO.copy();remaining=0
                    while True:
                        s=obs[-1].copy();teacher=oracle_action(e,'residual');observer=oracle_action(e,'effectiveness');valid=False;obsvalid=False
                        student=None
                        if model is not None:
                            with torch.no_grad():aa,proposal,state=model.propose_actions(torch.as_tensor(s[None],dtype=torch.float64),state)
                            executed=aa[0].numpy().astype(np.float32);student=proposal[0].numpy().astype(np.float32)
                        # A fixed schedule; final approaches are queried twice as often.
                        period=48 if e.route_reached else 96
                        query=(e.t>=48 and e.t%period==0) and (teacher[5]>0 or student is not None or (e.t%288==0))
                        if query:
                            labels,records,detail=choose(e,student=student,horizon=1440-e.t,include_observer=True,explore=(e.t%288==0),student_mode=e.config['intervention'])
                            selected_residual=labels['residual'];selected_observer=labels['effectiveness']
                            if selected_residual[5]>0:teacher=selected_residual.copy()
                            else:teacher[5]=-1.
                            if selected_observer[5]>0:observer=selected_observer.copy()
                            else:observer[5]=-1.
                            valid=obsvalid=True
                            for b in records:branches.append(dict(seed=seed,condition=condition,level=level,t=e.t,split='train' if index<48 else 'validation',student_requested_on=None if student is None else bool(executed[5]>0),student_model=None if model is None else dagger_models[mi]['path'],**b))
                            if model is None:held=(selected_observer if e.config['intervention']=='effectiveness' else selected_residual).copy();remaining=12
                        if condition=='exact':
                            # Knowing exact eta makes the correction teacher identically zero.
                            teacher=ZERO.copy();observer=ZERO.copy();valid=obsvalid=True
                        action=executed if student is not None else held if remaining>0 else ZERO
                        if student is not None and index%2==1 and e.t>=96 and e.t%192<12:
                            action=student.copy();action[4]=2*((action[4]+1)/2)*.5-1
                        if model is None and index%3==0:action=ZERO
                        for k,value in [('s',s),('a',teacher),('observer',observer),('valid',valid),('observer_valid',obsvalid),('eta',e.true_eta.copy())]:columns[k].append(value)
                        obs,reward,d,tr,info=e.step(action);remaining=max(0,remaining-1)
                        columns['executed'].append(action.copy());columns['task'].append(info['task_reward']);columns['style'].append(info['manual_style'])
                        columns['reward'].append(reward);columns['done'].append(d or tr);columns['ns'].append(obs[-1].copy())
                        if d or tr:break
                    n=len(columns['s']);arrays.append({k:np.asarray(v,dtype=bool if k in ['valid','observer_valid','done'] else np.float32) for k,v in columns.items()})
                    row={k:v for k,v in info.items() if k not in ['state','action','next_state']}
                    row.update(seed=seed,condition=condition,level=level,start=offset,stop=offset+n,split='train' if index<48 else 'validation',student_model=None if model is None else dagger_models[mi]['path'],
                        behavior=('student_shadow_bursts' if index%2 else 'student') if model is not None else ['baseline','oracle_residual_bursts','oracle_observer_bursts'][index%3],source_phase=phase)
                    episodes.append(row);offset+=n
    finally:e.close()
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]});os.replace(temporary,target)
    sources=[Path(__file__),Path(__file__).with_name('counterfactual.py'),Path(__file__).with_name('env.py'),Path(__file__).with_name('estimator.py'),Path(__file__).with_name('supervisor.py'),Path(__file__).with_name('models.py'),Path('experiments/round4/models.py'),Path('experiments/models.py'),Path('experiments/scan.py'),Path('experiments/vendor_gtrxl.py'),Path('gym_pybullet_drones/control/DSLPIDControl.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    save(meta,dict(seconds=time.time()-start,episodes=episodes,branches=branches,config=config,phase=phase,source_sha256=hashes))
    return str(meta)

def collect(config,count=60,workers=6,dagger_models=None):
    start=time.time();phase='initial' if not dagger_models else 'dagger';directory=OUT/'data'/phase;directory.mkdir(parents=True,exist_ok=True)
    indices=list(range(count)) if phase=='initial' else list(range(8))+list(range(48,52))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(collect_route,i,config,dagger_models) for i in indices]
        for f in as_completed(futures):
            result=f.result();print('DATA',phase,result,'elapsed',round(time.time()-start),flush=True)
            save(OUT/f'data_{phase}_progress.json',dict(completed=sum((directory/f'route_{i:03d}.json').exists() for i in indices),total=len(indices),seconds=time.time()-start))
    save(OUT/f'data_{phase}_complete.json',dict(seconds=time.time()-start,routes=len(indices)))

def load(include_dagger=True):
    arrays=[];episodes=[];branches=[];offset=0
    phases=['initial','dagger'] if include_dagger else ['initial']
    for phase in phases:
        for path in sorted((OUT/'data'/phase).glob('route_*.npz')):
            z=dict(np.load(path));meta=json.loads(path.with_suffix('.json').read_text())
            episodes.extend([dict(r,start=r['start']+offset,stop=r['stop']+offset) for r in meta['episodes']]);branches.extend(meta['branches'])
            offset+=len(z['s']);arrays.append(z)
    if not arrays:raise RuntimeError('No round5data')
    result={k:np.concatenate([z[k] for z in arrays]) for k in arrays[0]};result.update(records=episodes,branches=branches)
    train=np.zeros(len(result['s']),bool);validation=train.copy();expert=train.copy();episode_start=np.zeros(len(result['s']),np.int64)
    for r in episodes:
        ids=slice(r['start'],r['stop']);episode_start[ids]=r['start'];train[ids]=r['split']=='train';validation[ids]=r['split']=='validation'
        expert[ids]=r['split']=='train' and r['student_model'] is None and r['stable_landing'] and r['configuration']['intervention']=='residual'
    result.update(train=train,validation=validation,expert=expert,episode_start=episode_start);return result

class SequenceData:
    def __init__(self,data,mode='residual'):
        self.data=data;self.mode=mode;self.key='a' if mode=='residual' else 'observer';self.valid_key='valid' if mode=='residual' else 'observer_valid'
        self.records=[r for r in data['records'] if r['split']=='train']
        self.positive=[];self.verified=[]
        for r in self.records:
            ids=np.arange(r['start'],r['stop']);vv=ids[data[self.valid_key][ids]]
            self.verified.extend([(r,int(i)) for i in vv]);self.positive.extend([(r,int(i)) for i in vv if data[self.key][i,5]>0])
    def sample(self,rng,batch=8,length=96,burnin=32):
        samples=[];targets=[];masks=[];valids=[]
        for _ in range(batch):
            candidates=self.positive if rng.random()<.35 and self.positive else self.verified
            if rng.random()<.65 and candidates:r,first=candidates[int(rng.integers(len(candidates)))];first=max(r['start'],first-int(rng.integers(0,max(1,length//2))))
            else:r=self.records[int(rng.integers(len(self.records)))];first=int(rng.integers(r['start'],r['stop']))
            start=max(r['start'],first-burnin);stop=min(start+length+burnin,r['stop']);n=stop-start
            x=np.zeros((length+burnin,DIM),np.float32);a=np.tile(ZERO,(length+burnin,1));mask=np.zeros(length+burnin,np.float32);vv=mask.copy()
            x[:n]=self.data['s'][start:stop];a[:n]=self.data[self.key][start:stop];mask[first-start:n]=1;vv[:n]=self.data[self.valid_key][start:stop]
            samples.append(x);targets.append(a);masks.append(mask);valids.append(vv)
        return np.stack(samples),np.stack(targets),np.stack(masks),np.stack(valids)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--count',type=int,default=60);args=p.parse_args()
    collect(json.loads((OUT/'core_selection.json').read_text())['config'],count=args.count,workers=args.workers)
