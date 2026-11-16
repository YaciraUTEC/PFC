"""One protected supervised head refinement from terminal paired options.
No new online SAC benchmark is claimed. Recurrent features are reconstructed
from complete student histories, not shuffled isolated states or stale memories.
"""
import argparse,copy,time,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from ..round4.models import detach
from ..round5.bc import setup
from ..round6.models import physics_observation
from ..round6.offpolicy import OffPolicy,normalize
from .common import OUT,read,save,sha,close_milestone,gate_path
from .episode import load_policy
from .study import proposers,run,rank,baseline_for
from .evaluation import cases


def terminal_targets(z):
    groups=defaultdict(lambda:defaultdict(list))
    for i,(q,option) in enumerate(zip(z['trace_index'],z['x'][:,-1])):groups[int(q)][(float(option),tuple(z['x'][i,184:188]))].append(i)
    targets={}
    for q,options in groups.items():
        choices=[];negative=False
        for (duration,action_key),ids in options.items():
            labels=z['y'][ids];negative |= bool(np.any((labels==2)|(labels==0)))
            if len(np.unique(z['future'][ids]))>=2 and np.any(labels==1) and not np.any(labels==2):
                action=z['x'][ids[0],184:188];choices.append((float(np.linalg.norm(action)),duration,action.copy()))
        if choices:targets[q]=(min(choices,key=lambda x:(x[0],x[1]))[2],True)
        elif negative:targets[q]=(np.zeros(4,np.float32),False)
    return targets


def encoded_targets(model,cache):
    if cache.exists():
        with np.load(cache) as z:return z['features'],z['targets'],z['positive']
    prior=model.prior if isinstance(model,OffPolicy) else model
    features=[];targets=[];positive=[]
    with torch.no_grad():
        for p in sorted((OUT/'gate_data/train').glob('*.npz')):
            with np.load(p) as archive:z={k:archive[k] for k in ['x','y','trace','trace_index','episode_boundaries','future']}
            desired=terminal_targets(z)
            for begin,end in zip(z['episode_boundaries'][:-1],z['episode_boundaries'][1:]):
                indices=sorted(q for q in desired if begin<=q<end)
                if not indices:continue
                state=None;stop=indices[-1]+1
                for first in range(int(begin),int(stop),128):
                    last=min(first+128,int(stop));raw=physics_observation(torch.as_tensor(z['trace'][first:last][None],device='cuda'))
                    hidden,state=prior.encoder(raw,state);full=torch.cat([raw,hidden],-1)[0];state=detach(state)
                    for q in indices:
                        if first<=q<last:
                            features.append(full[q-first].cpu().numpy());target,is_positive=desired[q];targets.append(target);positive.append(is_positive)
    if not features:raise RuntimeError('No terminal paired targets')
    x=np.asarray(features,np.float32);y=np.asarray(targets,np.float32);pos=np.asarray(positive,bool);cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,features=x,targets=y,positive=pos);return x,y,pos


def actor_delta(model,features):
    if isinstance(model,OffPolicy):
        mean,_=model.actor.get_mean_and_std(features,training=False);return mean.tanh()
    mu=model.actor(features[:,-model.cfg['width']:])
    if hasattr(model,'physical_delta'):return model.physical_delta(mu)
    a=mu[:,:5].clamp(-5,5).tanh();return a[:,:4]*(a[:,4:5]+1)/2


def fit(source,seed):
    dest=OUT/'refinement'/source/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    setup(seed);start=time.time();path=proposers(seed)[source];m=load_policy(path).cuda().float().eval()
    x,y,pos=encoded_targets(m,dest/'encoded_targets.npz');encode_seconds=time.time()-start
    for p in m.parameters():p.requires_grad_(False)
    for p in m.actor.parameters():p.requires_grad_(True)
    reference=copy.deepcopy(m).eval()
    for p in reference.parameters():p.requires_grad_(False)
    rng=np.random.default_rng(seed+103);positive=np.flatnonzero(pos);negative=np.flatnonzero(~pos);opt=torch.optim.AdamW(m.actor.parameters(),lr=3e-5,weight_decay=1e-5);logs=[]
    prior=[]
    with torch.no_grad():
        for first in range(0,len(x),1024):prior.append(actor_delta(reference,torch.as_tensor(x[first:first+1024],device='cuda')).cpu().numpy())
    prior=np.concatenate(prior)
    for it in range(1,1201):
        if len(positive) and len(negative):ids=np.r_[rng.choice(positive,128),rng.choice(negative,128)]
        else:ids=rng.integers(len(x),size=256)
        xx=torch.as_tensor(x[ids],device='cuda');target=torch.as_tensor(y[ids],device='cuda');old=torch.as_tensor(prior[ids],device='cuda')
        pred=actor_delta(m,xx);supervised=F.mse_loss(pred,target);anchor=F.mse_loss(pred,old);loss=supervised+.3*anchor
        lr=3e-5*min(1.,it/100)*(.1+.9*.5*(1+math.cos(math.pi*it/1200)))
        for g in opt.param_groups:g['lr']=lr
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite protected refinement')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.actor.parameters(),.5);opt.step()
        if isinstance(m,OffPolicy):normalize(m.actor)
        if it%200==0:
            row=dict(update=it,loss=float(loss),target_loss=float(supervised),anchor_loss=float(anchor),seconds=time.time()-start);logs.append(row);save(dest/'progress.json',row);print('PROTECTED_HEAD_REFINEMENT',source,seed,row,flush=True)
    m.cpu().save(dest/'model.pt',refinement='supervised_terminal_pairs',updates=1200,source=str(path),source_sha=sha(path),encoder_frozen=True)
    result=dict(source=source,seed=seed,path=str(dest/'model.pt'),seconds=time.time()-start,encode_seconds=encode_seconds,updates=1200,
        paired_query_targets=len(y),rescue_targets=len(positive),abstention_targets=len(negative),history='complete recurrent reconstruction',new_online_steps=0,started_epoch=start,finished_epoch=time.time(),head_update_seconds=time.time()-start-encode_seconds)
    save(dest/'complete.json',result);return result


def train():
    if not (OUT/'h3_filter_ablation_conclusion.json').exists():raise RuntimeError('Close the filter ablation first')
    source=read(OUT/'h3_filter_ablation_conclusion.json')['selected']['name'].split('_')[0]
    # The five consistent proposal replicas were completed before H2.
    if source=='bc':
        selected=read(OUT/'selected_proposal.json')['paths_by_seed']
        assert all(Path(selected[str(s)]).exists() for s in [301,302,303,304,305])
    rows=[fit(source,s) for s in [301,302,303,304,305]]
    save(OUT/'refinement_proposers.json',[r['path'] for r in rows])
    template=read(OUT/'h3_filter_ablation_conclusion.json')['selected']['spec'];behavior=[]
    for seed in [301,302,303,304,305]:
        spec=dict(template,proposer=proposers(seed)[source])
        if spec.get('mode')=='counterfactual':spec['gate']=str(gate_path(seed))
        behavior.append(spec)
    save(OUT/'refinement_behavior_specs.json',behavior)
    close_milestone('h4_supervised_refinement','Five proposal heads refined with terminal rescue/harm supervision and prior anchoring; further flight validation decides retention.',source=source,results=rows)


def select():
    start=time.time();h3=read(OUT/'h3_filter_ablation_conclusion.json');template=dict(h3['selected']['spec']);source=read(OUT/'h4_supervised_refinement_conclusion.json')['source']
    cc=cases(range(710020,710024),levels=[0.,.25,.295,.296],clocks=[None,4.,8.]);base=baseline_for(cc,'refinement_flights');rows=[];families={}
    variants=[('unrefined',False,False),('refined',True,False)] if template.get('mode')!='counterfactual' else [('unrefined_original_gate',False,False),('unrefined_updated_gate',False,True),('refined_original_gate',True,False),('refined_updated_gate',True,True)]
    for family,refined,updated in variants:
        local=[]
        for seed in [301,302,303,304,305]:
            spec=dict(template);spec['proposer']=proposers(seed)[source] if not refined else str(OUT/'refinement'/source/f'seed_{seed}'/'model.pt')
            if spec.get('mode')=='counterfactual':
                spec['gate']=str(gate_path(seed,feedback=updated))
                if refined and not updated:spec['controller_version_override']=0.
            row=run(family+'_'+str(seed),spec,cc,base,'refinement_flights');local.append(row);rows.append(row)
        metric=dict(cases=sum(r['metrics']['cases'] for r in local),success=float(np.mean([r['metrics']['success'] for r in local])),
            critical_difference=float(np.mean([r['metrics']['critical_difference'] for r in local])),gains=sum(r['metrics']['gains'] for r in local),losses=sum(r['metrics']['losses'] for r in local),
            exact_losses=sum(r['metrics']['exact_losses'] for r in local),mild_losses=sum(r['metrics']['mild_losses'] for r in local))
        families[family]=dict(name=family,refined_head=refined,updated_authorizer=updated,metrics=metric,entries=[r['spec'] for r in local])
    chosen=max(families.values(),key=rank)
    close_milestone('h4_protected_finetuning','Retain the refinement only if independent full-flight development supports it. Five-model family selection is complete before final test.',
        selected_family=chosen['name'],selected=chosen,comparisons=families,results=rows,seconds=time.time()-start,
        no_online_sac_training_claim=True)
    save(OUT/'selected_candidate.json',dict(entries=chosen['entries'],family=chosen['name'],source=source,development=chosen['metrics'],parameters_frozen_pending_tests=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','select'],required=True);a=p.parse_args();train() if a.phase=='train' else select()
