"""Independent feasible-action fidelity isolates proposal quality from gating."""
import argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import numpy as np
import torch
from ..round4.models import detach
from ..round6.models import Agent
from .data import collect_group
from .common import OUT,save,read,old_model,close_milestone
from .episode import load_policy


def collect():
    start=time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(collect_group,i,'validation') for i in range(24,32)]):
            r=f.result();print('PROPOSAL_VALIDATION_GROUP',r['seed'],round(r['seconds']),flush=True)
    save(OUT/'proposal_validation_data_complete.json',dict(seeds=list(range(710024,710032)),seconds=time.time()-start,distinct_from_demonstrations=True,used_for_development_only=True))


def samples():
    out=[]
    for p in sorted((OUT/'demonstrations_validation').glob('*.npz')):
        metadata=read(p.with_suffix('.json'))
        with np.load(p) as z:
            arrays={k:z[k] for k in ['s','a','valid']}
            for r in metadata['records']:
                if r['behavior']!='teacher' or not r['expert']:continue
                a,b=r['start'],r['stop'];out.append((r,arrays['s'][a:b].copy(),arrays['a'][a:b].copy(),arrays['valid'][a:b].copy()))
    return out


def fidelity(path,data):
    m=load_policy(path).cuda().float().eval();groups=defaultdict(list);active_frames=0
    with torch.no_grad():
        for r,s,target,valid in data:
            state=None;proposal=[]
            for k in range(0,len(s),128):
                x=torch.as_tensor(s[k:k+128][None],device='cuda');z,state=m.encoder(x,state);mu=m.actor(z)[0];a=mu[:,:5].clamp(-5,5).tanh()
                delta=m.physical_delta(mu) if hasattr(m,'physical_delta') else a[:,:4]*(a[:,4:5]+1)/2
                proposal.append(delta.cpu().numpy());state=detach(state)
            pred=np.concatenate(proposal);actual=target[:,:4]*(target[:,4:5]+1)/2
            use=valid&(target[:,5]>0)
            if use.any():groups[r['seed']].append(float(np.square(pred[use]-actual[use]).mean()));active_frames+=int(use.sum())
    del m;torch.cuda.empty_cache()
    if not groups:raise RuntimeError('No held-out active proposals')
    return dict(active_mse=float(np.mean([np.mean(v) for v in groups.values()])),routes=len(groups),episodes=sum(len(v) for v in groups.values()),active_frames=active_frames)


def select():
    torch.set_num_threads(1);start=time.time();data=samples();rows=[]
    for g in [0,16,64,256]:
        for seed in [301,302,303]:
            path=OUT/'bc_curve'/f'groups_{g}'/f'seed_{seed}'/'final.pt'
            if not path.exists():raise RuntimeError('Finish every matched BC fit first')
            metric=fidelity(path,data);rows.append(dict(groups=g,seed=seed,path=str(path),**metric));print('PROPOSAL_FIDELITY',g,seed,metric,flush=True)
    reference=fidelity(old_model(201,'bc'),data)
    means={g:float(np.mean([r['active_mse'] for r in rows if r['groups']==g])) for g in [0,16,64,256]};winner=min(means,key=lambda g:(means[g],g))
    close_milestone('h1_proposal_selection','Select proposal scale by independent executed-action fidelity at matched 3600 updates; gate and native-flight curves remain separate evidence.',
        selected_groups=winner,route_mean_mse_by_scale=means,reference_round6_bc=reference,results=rows,seconds=time.time()-start,
        caveat='Action fidelity does not establish flight success; complete paired terminal continuations and independent full flights follow.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.collect:collect()
    else:select()
