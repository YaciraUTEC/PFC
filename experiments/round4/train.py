"""Contiguous-sequence PPO with demonstration retention and frozen-BC KL.
The critic has separate parameters; first8192 steps only warm the critic.
"""
import argparse,copy,json,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.vec_env import SubprocVecEnv
from .models import Agent,ARCHS,configurations,tree_map,detach,select,concatenate
from .env import LandingEnv,LEVELS
from .data import load,SequenceData
from .evaluate import OUT,evaluate,close_pool
from ..run import save

SEEDS=[71,72,73,74,75]
BUDGET=524288
VAL_SEEDS=list(range(430000,430008))
VAL_LEVELS=[0.,.10,.20,.25,.28,.29]
torch.set_num_threads(1)

def setup(seed):
    np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)

def clone_loss(mu,target,mask):
    a=mu.tanh();gate=(a[...,4]+1)/2;tg=(target[...,4]+1)/2
    residual=(gate[...,None]*a[...,:4]-tg[...,None]*target[...,:4]).square().mean(-1)
    gate_loss=F.binary_cross_entropy_with_logits(2*mu[...,4],tg,reduction='none')
    direction=(a[...,:4]-target[...,:4]).square().mean(-1)*tg
    return ((4*residual+.15*gate_loss+.05*direction)*mask).sum()/mask.sum().clamp_min(1)

def pretrain(m,dataset,seed,updates=1200):
    rng=np.random.default_rng(seed);opt=m.optimizer(3e-4);log=[];start=time.time()
    m.train()
    for it in range(updates):
        obs,target,mask=dataset.sample(rng,batch=8,length=64,burnin=32)
        mu,_,_=m(tensor(obs));loss=clone_loss(mu,tensor(target),tensor(mask))
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
        if it%200==0:log.append(dict(update=it,loss=float(loss)));print('BC',m.cfg['arch'],seed,it,float(loss),flush=True)
    m.eval();return dict(seconds=time.time()-start,updates=updates,losses=log)

def validation(m):
    rows=[]
    for condition in ['exact','estimated']:
        rr=evaluate(m,condition=condition,seeds=VAL_SEEDS,levels=VAL_LEVELS)
        rows.extend([dict(r,condition=condition) for r in rr])
    mild=[r['stable_landing'] for r in rows if r['level']<=.10]
    severe=[r['stable_landing'] for r in rows if r['level']>.10]
    effort=float(np.mean([r['effort'] for r in rows]))
    # Mild-safety penalty is explicit; individual conditions also reported.
    score=float(np.mean(severe))-.5*(1-float(np.mean(mild)))-.001*effort
    return dict(score=score,success=float(np.mean([r['stable_landing'] for r in rows])),
        mild=float(np.mean(mild)),severe=float(np.mean(severe)),effort=effort,episodes=rows)

def hyperparameters(dataset):
    directory=OUT/'hpo';directory.mkdir(parents=True,exist_ok=True)
    for arch in ARCHS:
        if (directory/f'{arch}_selection.json').exists():continue
        candidates=[]
        for index,cfg in enumerate(configurations(arch)):
            dest=directory/f'{arch}_{index}';dest.mkdir(exist_ok=True)
            setup(67)
            if (dest/'result.json').exists():result=json.loads((dest/'result.json').read_text());candidates.append(result);continue
            m=Agent(cfg).cuda();timing=pretrain(m,dataset,67,updates=600)
            m.save(dest/'bc.pt',phase='hpo')
            finetune(m,dataset,67,dest,budget=65536)
            vv=json.loads((dest/'validation.json').read_text())
            best_validation=max(vv,key=lambda x:x['score'])
            result=dict(config=cfg,index=index,timing=timing,validation=best_validation,
                parameters=sum(p.numel() for p in m.parameters()))
            save(dest/'result.json',result);candidates.append(result);del m;torch.cuda.empty_cache()
        choice=max(candidates,key=lambda x:x['validation']['score'])
        save(directory/f'{arch}_selection.json',dict(selected_index=choice['index'],config=choice['config'],
            rule='highest reserved-validation severe landing score,penalize mild regressions;tie effort',candidates=candidates))
        print('HPO_SELECTED',arch,choice['index'],choice['validation']['score'],flush=True)
    close_pool()

def make_vec(seed,n=8):
    def factory(index):
        def init():
            torch.set_num_threads(1)
            condition='exact' if index%4==0 else 'biased' if index==7 else 'estimated'
            return LandingEnv(condition=condition,guard=True)
        return init
    v=SubprocVecEnv([factory(i) for i in range(n)],start_method='spawn');v.seed(seed);return v

def sequences(dones,length=64):
    T,N=dones.shape;out=[]
    for j in range(N):
        episode_start=0;start=0
        while start<T:
            end=min(T,start+length)
            hits=np.flatnonzero(dones[start:end,j])
            if len(hits):end=start+int(hits[0])+1
            pre=min(16,start-episode_start)
            out.append((start,end,j,start-pre,pre))
            if dones[end-1,j]:episode_start=end
            start=end
    return out

def seq_batch(items,obs,states):
    B=len(items);length=max(b-a for a,b,*_ in items);padded=np.zeros((B,length,64),np.float32);mask=np.zeros((B,length),np.float32)
    initial=[];groups={}
    for k,(a,b,j,p,pre) in enumerate(items):
        padded[k,:b-a]=obs[a:b,j];mask[k,:b-a]=1
        initial.append(tree_map(lambda x:x[j:j+1].cuda(),states[p]))
        groups.setdefault(pre,[]).append(k)
    initial=concatenate(initial)
    return tensor(padded),tensor(mask),initial,groups

def advance_burnin(m,items,obs,initial,groups):
    result=tree_map(lambda x:x.clone(),initial)
    with torch.no_grad():
        for pre,indices in groups.items():
            if pre==0:continue
            x=np.stack([obs[items[k][3]:items[k][0],items[k][2]] for k in indices])
            _,st=m.encoder(tensor(x),select(initial,indices))
            def put(dst,src):
                if isinstance(dst,(tuple,list)):
                    for d,s in zip(dst,src):put(d,s)
                else:dst[indices]=src
            put(result,st)
    return detach(result)

def padded_values(items,arr,length):
    out=np.zeros((len(items),length,*arr.shape[2:]),np.float32)
    for k,(a,b,j,*_) in enumerate(items):out[k,:b-a]=arr[a:b,j]
    return tensor(out)

def finetune(m,dataset,seed,dest,budget=BUDGET):
    reference=copy.deepcopy(m).eval()
    if m.cfg['arch']=='lstm':reference.encoder.core.flatten_parameters()
    for p in reference.parameters():p.requires_grad_(False)
    v=make_vec(510000+seed);N=v.num_envs;T=256;obs=v.reset()[:,0]
    state=None;refstate=None;opt=m.optimizer(m.cfg['lr']);rng=np.random.default_rng(seed+713)
    start=time.time();validation_seconds=0.;log=[];steps=0
    initial_validation_start=time.time()
    base=validation(m);validation_seconds=time.time()-initial_validation_start;best=base['score'];m.save(dest/'best.pt',steps=0,validation_score=best)
    save(dest/'validation.json',[dict(steps=0,**base)]);evaluations=[dict(steps=0,**base)]
    try:
        while steps<budget:
            m.eval();observations=[];actions=[];logps=[];values=[];rewards=[];dones=[];memories=[];refmus=[]
            for t in range(T):
                if state is None:state=m.encoder.initial(N)
                memories.append(tree_map(lambda x:x.detach().cpu().clone(),state))
                observations.append(obs.copy())
                with torch.no_grad():
                    a,lp,value,state=m.act(tensor(obs),state)
                    rm,_,refstate=reference(tensor(obs)[:,None],refstate)
                nxt,r,done,infos=v.step(a.cpu().numpy())
                # Timeout has a bootstrap value; true landing/crash does not.
                for j,(d,i) in enumerate(zip(done,infos)):
                    if d and i.get('TimeLimit.truncated',False):
                        terminal=i['terminal_observation'][0]
                        with torch.no_grad():r[j]+=.997*float(m.critic(tensor(terminal)))
                actions.append(a.cpu().numpy());logps.append(lp.cpu().numpy());values.append(value.cpu().numpy())
                rewards.append(r);dones.append(done);refmus.append(rm[:,0].cpu().numpy())
                keep=tensor(~done)
                state=tree_map(lambda x:x*keep.reshape(N,*([1]*(x.ndim-1))),detach(state))
                refstate=tree_map(lambda x:x*keep.reshape(N,*([1]*(x.ndim-1))),detach(refstate))
                obs=nxt[:,0]
            with torch.no_grad():last=m.critic(tensor(obs)).squeeze(-1).cpu().numpy()
            observations=np.asarray(observations);actions=np.asarray(actions);logps=np.asarray(logps)
            values=np.asarray(values);rewards=np.asarray(rewards);dones=np.asarray(dones);refmus=np.asarray(refmus)
            advantages=np.zeros_like(rewards);acc=np.zeros(N)
            for t in range(T-1,-1,-1):
                boot=last if t==T-1 else values[t+1]
                delta=rewards[t]+.997*boot*(~dones[t])-values[t]
                acc=delta+.997*.95*(~dones[t])*acc;advantages[t]=acc
            returns=advantages+values;advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            items=sequences(dones);losses=[];kls=[];m.train()
            progress=steps/max(budget,1);warm=min(1.,(steps+2048)/16384) if m.cfg['arch']=='gtrxl' else 1.
            lr=m.cfg['lr']*warm*(.1+.9*.5*(1+np.cos(np.pi*progress)))
            for group in opt.param_groups:group['lr']=lr
            stop=False
            for epoch in range(3):
                rng.shuffle(items)
                for first in range(0,len(items),8):
                    batch=items[first:first+8];x,mask,initial,groups=seq_batch(batch,observations,memories)
                    initial=advance_burnin(m,batch,observations,initial,groups)
                    mu,val,_=m(x,initial);L=x.shape[1];a=padded_values(batch,actions,L)
                    oldlp=padded_values(batch,logps[...,None],L).squeeze(-1)
                    adv=padded_values(batch,advantages[...,None],L).squeeze(-1)
                    ret=padded_values(batch,returns[...,None],L).squeeze(-1)
                    oldmu=padded_values(batch,refmus,L)
                    dist=torch.distributions.Normal(mu,m.log_std.exp());raw=torch.atanh(a.clamp(-.999999,.999999))
                    lp=(dist.log_prob(raw)-torch.log(1-a.square()+1e-6)).sum(-1)
                    ratio=(lp-oldlp).exp();den=mask.sum().clamp_min(1)
                    actor=(-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv)*mask).sum()/den
                    vf=((val-ret).square()*mask).sum()/den
                    refdist=torch.distributions.Normal(oldmu,reference.log_std.exp())
                    kl=(torch.distributions.kl_divergence(dist,refdist).sum(-1)*mask).sum()/den
                    approx=(((ratio-1)-(lp-oldlp))*mask).sum()/den
                    if steps<8192:loss=.5*vf
                    else:
                        sx,ta,ma=dataset.sample(rng,batch=4,length=32,burnin=16)
                        smu,_,_=m(tensor(sx));bc=clone_loss(smu,tensor(ta),tensor(ma))
                        loss=actor+.5*vf+.01*kl+(.3+.7*(1-progress))*bc
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
                    with torch.no_grad():m.log_std.clamp_(-4.,-2.3)
                    losses.append(float(loss));kls.append(float(kl))
                    if float(approx)>.02:stop=True;break
                if stop:break
            steps+=T*N;m.eval()
            if steps%16384==0:
                save(dest/'progress.json',dict(steps=steps,total=budget,seconds=time.time()-start,loss=float(np.mean(losses)),bc_kl=float(np.mean(kls)),lr=lr))
                print('PPO_PROTECTED',m.cfg['arch'],seed,steps,'/',budget,'seconds',round(time.time()-start),flush=True)
            if steps%131072==0 or steps>=budget:
                ts=time.time();result=validation(m);validation_seconds+=time.time()-ts
                evaluations.append(dict(steps=steps,**result));save(dest/'validation.json',evaluations)
                if result['score']>best:
                    best=result['score'];m.save(dest/'best.pt',steps=steps,validation_score=best)
                m.save(dest/'latest.pt',steps=steps)
            del observations,actions,memories,initial
        m.save(dest/'final.pt',steps=steps)
        save(dest/'complete.json',dict(steps=steps,seconds=time.time()-start,validation_seconds=validation_seconds,best_score=best,
             selected_steps=max(evaluations,key=lambda x:x['score'])['steps'],base_score=base['score'],protected=True))
    finally:v.close()

def train(dataset,archs=ARCHS,seeds=SEEDS,budget=BUDGET,keep_pool=False):
    for arch in archs:
        cfg=json.loads((OUT/'hpo'/f'{arch}_selection.json').read_text())['config']
        for seed in seeds:
            dest=OUT/'models'/arch/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():continue
            setup(seed);m=Agent(cfg).cuda()
            if not (dest/'bc.pt').exists():
                timing=pretrain(m,dataset,seed);m.save(dest/'bc.pt',phase='bc')
                save(dest/'bc_timing.json',timing)
            else:m=Agent.load(dest/'bc.pt')
            finetune(m,dataset,seed,dest,budget)
            del m;torch.cuda.empty_cache()
    if not keep_pool:close_pool()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['hpo','train','all'],default='all')
    p.add_argument('--archs',nargs='+',default=ARCHS);p.add_argument('--seeds',type=int,nargs='+',default=SEEDS);p.add_argument('--budget',type=int,default=BUDGET)
    args=p.parse_args();data=load();dataset=SequenceData(data)
    if args.phase in ['hpo','all']:hyperparameters(dataset)
    if args.phase in ['train','all']:train(dataset,args.archs,args.seeds,args.budget)
