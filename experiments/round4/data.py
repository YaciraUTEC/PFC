"""Successful teachers, negative trajectories and counterfactual gate labels.
Splits are by geometry; all unsuccessful trajectories remain in the record.
"""
import argparse,json,time,contextlib,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from .env import LandingEnv,LEVELS,ZERO
from .counterfactual import choose
from .evaluate import OUT
from ..run import save

def collect_route(index):
    directory=OUT/'data_governor_v2';target=directory/f'route_{index:03d}.npz';meta=directory/f'route_{index:03d}.json'
    if target.exists() and meta.exists():return str(meta)
    seed=420000+index;start=time.time();arrays=[];records=[];offset=0;branchlog=[]
    with open(os.devnull,'w') as silent,contextlib.redirect_stdout(silent):
        e=LandingEnv(guard=True)
    try:
        for condition in ['exact','estimated','biased']:
            e.condition=condition
            for level in LEVELS:
                o,_=e.reset(seed=seed,options=dict(level=level));ss=[];aa=[];rr=[];dd=[];nn=[]
                while True:
                    a=e.teacher_action()
                    if e.need()==0.:a=ZERO.copy()
                    # Validation uses an identical fixed label procedure, never model/test feedback.
                    if e.t%48==0 and a[4]>-.99:
                        a,detail=choose(e,horizon=24);branchlog.append(dict(condition=condition,level=level,t=e.t,**detail))
                    ss.append(o[-1].copy());aa.append(a.copy())
                    o,r,d,t,i=e.step(a);rr.append(r);dd.append(d or t);nn.append(i['next_state'])
                    if d or t:break
                n=len(ss);arrays.append(dict(s=np.asarray(ss,np.float32),a=np.asarray(aa,np.float32),
                    reward=np.asarray(rr,np.float32),done=np.asarray(dd,bool),ns=np.asarray(nn,np.float32)))
                row={k:v for k,v in i.items() if k not in ['state','action','next_state']}
                row.update(seed=seed,condition=condition,start=offset,stop=offset+n,split='train' if index<32 else 'validation')
                records.append(row);offset+=n
    finally:e.close()
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**{k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})
    os.replace(temp,target);save(meta,dict(episodes=records,seconds=time.time()-start,branches=branchlog))
    return str(meta)

def collect(count=40,workers=4):
    (OUT/'data_governor_v2').mkdir(parents=True,exist_ok=True);start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(collect_route,i) for i in range(count)]
        for f in as_completed(fs):
            name=f.result();print('completed',name,flush=True)
            save(OUT/'data_progress.json',dict(routes_done=sum((OUT/'data_governor_v2'/f'route_{i:03d}.json').exists() for i in range(count)),routes=count,seconds=time.time()-start))
    save(OUT/'data_complete.json',dict(routes=count,seconds=time.time()-start))
def load():
    arrays=[];records=[];offset=0
    for p in sorted((OUT/'data_governor_v2').glob('route_*.npz')):
        z=dict(np.load(p));meta=json.loads(p.with_suffix('.json').read_text())
        for r in meta['episodes']:records.append(dict(r,start=r['start']+offset,stop=r['stop']+offset))
        offset+=len(z['s']);arrays.append(z)
    data={k:np.concatenate([x[k] for x in arrays]) for k in arrays[0]}
    data['records']=records;return data

class SequenceData:
    def __init__(self,data):
        self.data=data
        self.records=[r for r in data['records'] if r['split']=='train' and r['mission_success']]
        if not self.records:raise RuntimeError('No successful teachers')
    def sample(self,rng,batch=8,length=64,burnin=16):
        samples=[];targets=[];masks=[]
        for _ in range(batch):
            r=self.records[int(rng.integers(len(self.records)))]
            # Half all states / half post-fault or descent; preserve true episode boundary.
            lo=r['start']
            if rng.random()<.5:lo=min(r['stop']-1,lo+int((r['stop']-lo)*.35))
            first=int(rng.integers(lo,r['stop']))
            start=max(r['start'],first-burnin);stop=min(start+length+burnin,r['stop'])
            n=stop-start;s=np.zeros((length+burnin,64),np.float32)
            a=np.tile(ZERO,(length+burnin,1));mask=np.zeros(length+burnin,np.float32)
            s[:n]=self.data['s'][start:stop];a[:n]=self.data['a'][start:stop]
            mask[max(0,first-start):n]=1
            samples.append(s);targets.append(a);masks.append(mask)
        return np.stack(samples),np.stack(targets),np.stack(masks)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--routes',type=int,default=40);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();collect(a.routes,a.workers)
