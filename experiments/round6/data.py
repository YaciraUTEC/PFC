"""Paired full-episode demonstration selection and genuine executed-action replay.
BC success labels are trajectory supervision, not claims of local causal benefit.
"""
import json,time,contextlib,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .env import LandingEnv,ZERO,DIM
from .evaluate import OUT,save
from ..round5.data import SequenceData as PreviousSequence

TRAIN_LEVELS=[0.,.02,.10,.25,.28,.29,.295,.296,.30,.35,.50]
def collect_route(index,config):
    dest=OUT/'data'/f'route_{index:03d}.npz';meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():return str(meta)
    seed=620000+index;arrays=[];records=[];offset=0;start=time.time()
    with open(os.devnull,'w') as f,contextlib.redirect_stdout(f):e=LandingEnv(config=dict(config,gate_mode='learned'))
    try:
        for level in TRAIN_LEVELS:
            for condition in ['exact','estimated']:
                for noise in [1.,2.]:
                    flights=[]
                    for behavior in ['baseline','oracle','bursts']:
                        if condition=='exact' and behavior!='baseline':continue
                        e.condition=condition;e.config['noise_multiplier']=noise;e._sensor_sigma=.002*noise
                        obs,_=e.reset(seed=seed,options=dict(level=level))
                        columns={k:[] for k in ['s','ns','executed','reward','done','motor_command_rpm','projected_motor_delta']}
                        while True:
                            action=ZERO.copy()
                            if behavior=='oracle' or (behavior=='bursts' and e.t%48<12):action=e.teacher_action()
                            columns['s'].append(obs[-1].copy())
                            obs,reward,d,t,info=e.step(action)
                            for k,v in [('ns',obs[-1].copy()),('executed',info['executed_action'].copy()),('reward',reward),('done',d or t),('motor_command_rpm',info['motor_command_rpm'].copy()),('projected_motor_delta',info['projected_motor_delta'].copy())]:columns[k].append(v)
                            if d or t:break
                        flights.append((behavior,columns,info))
                    baseline=flights[0][2]['stable_landing']
                    for behavior,c,info in flights:
                        n=len(c['s']);good=bool(info['stable_landing'] and (behavior=='baseline' or not baseline))
                        # Preserve negative trajectories for Q-learning, but not as expert BC.
                        a=np.asarray(c['executed'],np.float32) if good else np.tile(ZERO,(n,1))
                        arrays.append(dict(s=np.asarray(c['s'],np.float32),ns=np.asarray(c['ns'],np.float32),executed=np.asarray(c['executed'],np.float32),
                            reward=np.asarray(c['reward'],np.float32),done=np.asarray(c['done'],bool),a=a,
                            valid=np.full(n,good,bool),train=np.ones(n,bool),motor_command_rpm=np.asarray(c['motor_command_rpm'],np.float32),projected_motor_delta=np.asarray(c['projected_motor_delta'],np.float32)))
                        records.append(dict(start=offset,stop=offset+n,seed=seed,level=level,condition=condition,noise=noise,behavior=behavior,
                            stable_landing=info['stable_landing'],baseline_success=baseline,expert=good,split='train',outcome=info['outcome']))
                        offset+=n
    finally:e.close()
    np.savez_compressed(dest,**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    save(meta,dict(records=records,seconds=time.time()-start,source='new_round6_executed_actions',config=config))
    return str(meta)
def collect(config,count=16,workers=8):
    (OUT/'data').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(collect_route,i,config) for i in range(count)]):print('DATA_DONE',f.result(),flush=True)
def load():
    arrays=[];records=[];offset=0
    for p in sorted((OUT/'data').glob('route_*.npz')):
        z=dict(np.load(p));meta=json.loads(p.with_suffix('.json').read_text())
        records.extend(dict(r,start=r['start']+offset,stop=r['stop']+offset) for r in meta['records']);arrays.append(z);offset+=len(z['s'])
    if not arrays:raise RuntimeError('No round6 data')
    result={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};result['records']=records
    result['episode_start']=np.empty(len(result['s']),np.int64)
    for r in records:result['episode_start'][r['start']:r['stop']]=r['start']
    return result
class SequenceData:
    def __init__(self,data,max_noise=None):
        self.data=data;self.records=[r for r in data['records'] if r['expert'] and (max_noise is None or r['noise']<=max_noise)];self.mode='residual';self.positive=[];self.verified=[];self.onsets=[]
        for r in self.records:
            ids=np.arange(r['start'],r['stop']);self.verified.extend((r,int(i)) for i in ids[::16])
            active=data['a'][ids,5]>0
            onset=active&~np.r_[False,active[:-1]]
            self.onsets.extend((r,int(i)) for i in ids[onset])
            self.positive.extend((r,int(i)) for i in ids[active&((ids%8==0)|onset)])
        if not self.records:raise RuntimeError('No successful BC trajectories')
    def sample(self,rng,batch=8,length=96,burnin=32):
        xs=[];actions=[];masks=[];valids=[]
        for _ in range(batch):
            u=rng.random();use_onset=bool(self.onsets and u<.4)
            pool=self.onsets if use_onset else self.positive if self.positive and u<.7 else self.verified
            r,first=pool[int(rng.integers(len(pool)))]
            if use_onset:first=max(r['start'],first-12)
            start=max(r['start'],first-burnin);stop=min(r['stop'],start+length+burnin);n=stop-start
            x=np.zeros((length+burnin,DIM),np.float32);a=np.tile(ZERO,(length+burnin,1));m=np.zeros(length+burnin,np.float32)
            x[:n]=self.data['s'][start:stop];a[:n]=self.data['a'][start:stop];m[first-start:n]=1.
            xs.append(x);actions.append(a);masks.append(m);valids.append(m.copy())
        return np.stack(xs),np.stack(actions),np.stack(masks),np.stack(valids)
