"""Separate feasible proposal demonstrations from baseline-relative gate labels."""
import argparse,time,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from gym_pybullet_drones.control.NMPCINDIControl import NMPCINDIControl
from gym_pybullet_drones.utils.enums import DroneModel
from ..round6.env import LandingEnv,ZERO,DIM
from .common import OUT,OLD,config,save,read,silent,old_model,sha,close_milestone
from .episode import infer,action_from_delta,load_policy,terminal_info


def collect_group(index,split='train'):
    torch.set_num_threads(1);folder='demonstrations' if split=='train' else 'demonstrations_validation'
    target=OUT/folder/f'group_{index:03d}.npz';meta=target.with_suffix('.json')
    if target.exists() and meta.exists():return read(meta)
    start=time.time();group_seed=(700000 if split=='train' else 710000)+index;clock=[None,4.,6.,8.][index%4]
    p1=read(OLD/'phase1_conclusion.json');opts=p1['all_configs'][p1['best_slow_nmpc']]['nmpc_options']
    arrays=[];records=[];offset=0
    with silent():e=LandingEnv(config=config(fault_clock=clock,noise_multiplier=1.))
    with silent():teacher=NMPCINDIControl(DroneModel.CF2X,**opts)
    student=load_policy(old_model(201+index%5)) if index%4==3 else None
    cases=[(l,c) for l in [.28,.29,.295,.296] for c in ['estimated','exact']]
    if index%8==0:cases += [(0.,'estimated'),(.25,'estimated')]
    try:
        for level,condition in cases:
            paired=[]
            for behavior in ['baseline','teacher']:
                e.condition=condition
                with silent():obs,_=e.reset(seed=group_seed,options=dict(level=level))
                teacher.reset();memory=None;latched=False;student_ticks=0;previous=np.full(4,e.c.HOVER_RPM)
                cols={k:[] for k in ['s','ns','a','executed','valid','reward','done','motor_command_rpm','projected_motor_delta']}
                while True:
                    latched=latched or bool(e.estimated_eta.min()<.98)
                    a=ZERO.copy();is_teacher=True
                    if behavior=='teacher' and latched and not e.touchdown_seen:
                        # A fixed short student prefix exposes the teacher to student states.
                        if student is not None and student_ticks<24:
                            a,memory=infer(student,obs,memory);student_ticks+=1;is_teacher=False
                        else:
                            teacher.set_motor_effectiveness(e.estimated_eta)
                            teacher.set_motor_speeds(np.clip(previous*e.estimated_eta,0,teacher._command_max))
                            rpm,_,_=teacher.computeControl(**e._arguments)
                            a=action_from_delta(np.clip(((rpm/teacher.HOVER_RPM)**2-e.baseq)/e.qscale,-1,1))
                    before=obs[-1].copy();obs,reward,d,tr,info=e.step(a);previous=info['motor_command_rpm'].copy()
                    vals=dict(s=before,ns=obs[-1].copy(),a=info['executed_action'].copy(),executed=info['executed_action'].copy(),valid=is_teacher,
                        reward=reward,done=d or tr,motor_command_rpm=previous,projected_motor_delta=info['projected_motor_delta'])
                    for k,v in vals.items():cols[k].append(v)
                    if d or tr:break
                paired.append((behavior,cols,info))
            baseline=bool(paired[0][2]['stable_landing'])
            for behavior,cols,info in paired:
                n=len(cols['s']);good=bool(info['stable_landing'])
                arr={k:np.asarray(v,dtype=bool if k in ['done','valid'] else np.float32) for k,v in cols.items()}
                arr['valid'] &= good
                arrays.append(arr)
                records.append(dict(start=offset,stop=offset+n,seed=group_seed,group=index,level=level,condition=condition,noise=1.,clock=clock,
                    behavior=behavior,expert=good,baseline_success=baseline,rescue=good and not baseline and behavior=='teacher',
                    stable_landing=good,outcome=info['outcome'],student_prefix=student is not None and behavior=='teacher',
                    active_steps=int((arr['valid']&(arr['a'][:,5]>0)).sum())))
                offset+=n
    finally:e.close()
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    result=dict(group=index,seed=group_seed,seconds=time.time()-start,records=records,teacher_options=opts,
        proposal_rule='Any successful executed teacher trajectory; baseline success does not disqualify a feasible proposal.',teacher_true_eta=False)
    save(meta,result);return result


def collect(count=16,workers=8):
    start=time.time();(OUT/'demonstrations').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(collect_group,i) for i in range(count)]):
            r=f.result();print('DEMONSTRATION_GROUP',r['group'],'seconds',round(r['seconds'],1),'successful_teacher',sum(x['expert'] and x['behavior']=='teacher' for x in r['records']),flush=True)
    rows=[r for i in range(count) for r in read(OUT/'demonstrations'/f'group_{i:03d}.json')['records']]
    result=dict(groups=count,seconds=time.time()-start,episodes=len(rows),successful_proposals=sum(r['expert'] and r['behavior']=='teacher' for r in rows),
        active_demonstration_episodes=sum(r['expert'] and r['active_steps']>0 for r in rows),active_steps=sum(r['active_steps'] for r in rows),
        distinct_active_seeds=len({r['seed'] for r in rows if r['active_steps']>0}),rescues=sum(r['rescue'] for r in rows),
        by_level={str(l):dict(episodes=sum(r['level']==l for r in rows),successful_teacher=sum(r['level']==l and r['expert'] and r['behavior']=='teacher' for r in rows)) for l in sorted({r['level'] for r in rows})})
    save(OUT/f'demonstration_audit_{count}.json',result);return result


def load(groups=16,old=True,successful_only=True):
    paths=(sorted((OLD/'data').glob('route_*.npz')) if old else [])+[OUT/'demonstrations'/f'group_{i:03d}.npz' for i in range(groups)]
    chunks=[];records=[];offset=0
    for p in paths:
        meta=read(p.with_suffix('.json'));rows=meta['records']
        with np.load(p) as archive:
            z={k:archive[k] for k in ['s','ns','a','executed','valid','reward','done']}
            for r in rows:
                if r['noise']>1.:continue
                if successful_only and not r['expert']:continue
                a,b=r['start'],r['stop'];n=b-a
                valid=z['valid'][a:b].copy()
                if not r['expert']:valid[:]=False
                chunk=dict(s=z['s'][a:b].copy(),ns=z['ns'][a:b].copy(),a=z['a'][a:b].copy(),executed=z['executed'][a:b].copy(),
                    valid=valid,reward=z['reward'][a:b].copy(),done=z['done'][a:b].copy())
                if p.is_relative_to(OLD):chunk['valid'][:]=bool(r['expert'])
                chunks.append(chunk);records.append(dict(r,start=offset,stop=offset+n));offset+=n
    if not chunks:raise RuntimeError('No proposal data')
    data={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]};data['records']=records;return data

class ProposalSequences:
    """Route-uniform sampling then active/onset sampling within that route."""
    mode='feasible_proposal'
    def __init__(self,data):
        self.data=data;self.records=[];self.active_records=[];self.onsets=[];self.positive=[];self.verified=[]
        for r in data['records']:
            ids=np.arange(r['start'],r['stop']);valid=data['valid'][ids];ids=ids[valid]
            if not len(ids):continue
            rec=dict(r,valid_ids=ids)
            active=ids[data['a'][ids,5]>0];rec['active_ids']=active
            self.records.append(rec);self.verified.append(rec)
            if len(active):self.active_records.append(rec);self.positive.append(rec)
        if not self.records:raise RuntimeError('No valid demonstrations')
    def sample(self,rng,batch=8,length=96,burnin=32):
        xs=[];aa=[];masks=[];gvalid=[]
        for _ in range(batch):
            active=bool(self.active_records and rng.random()<.65);pool=self.active_records if active else self.records
            r=pool[int(rng.integers(len(pool)))];ids=r['active_ids'] if active else r['valid_ids'];first=int(ids[int(rng.integers(len(ids)))])
            start=max(r['start'],first-burnin);stop=min(r['stop'],start+length+burnin);n=stop-start
            x=np.zeros((length+burnin,DIM),np.float32);a=np.tile(ZERO,(length+burnin,1));mask=np.zeros(length+burnin,np.float32)
            x[:n]=self.data['s'][start:stop];a[:n]=self.data['a'][start:stop];mask[first-start:n]=self.data['valid'][first:stop]
            xs.append(x);aa.append(a);masks.append(mask);gvalid.append(mask.copy())
        return np.stack(xs),np.stack(aa),np.stack(masks),np.stack(gvalid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',type=int,default=16);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    print('DATA_AUDIT',collect(args.groups,args.workers),flush=True)
