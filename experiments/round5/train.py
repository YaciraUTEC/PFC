"""Matched recurrent PPO, explicit Bernoulli abstention and updated reward screen.
Task deadlines are terminal. Actor and BC reference keep episode memory. The
critic uses detached history features; no value gradient enters policy memory.
"""
import argparse,copy,json,time
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.vec_env import SubprocVecEnv
from .models import Agent,ARCHS,tree_map,detach,select,concatenate,load_agent
from .env import LandingEnv,DIM
from .data import load,SequenceData
from .bc import pretrain,setup,tensor,starting_config,SEEDS
from .evaluate import OUT,evaluate,close_pool
from ..run import save
from .losses import clone_loss
BUDGET=524288;GAMMA=.999;LAMBDA=.97

def core_config():return json.loads((OUT/'core_selection.json').read_text())['config']
def validation(m,config=None,seeds=None):
    reference=copy.deepcopy(m).cpu().double().eval();rows=[]
    cfg=core_config();cfg.update(config or {})
    for condition in ['exact','estimated']:
        rows.extend([dict(r,condition=condition) for r in evaluate(reference,condition=condition,seeds=range(528000,528008) if seeds is None else seeds,levels=[0.,.02,.25,.28,.29,.295],config=cfg)])
    del reference
    mild=[r['stable_landing'] for r in rows if r['level']<=.25];severe=[r['stable_landing'] for r in rows if r['condition']=='estimated' and r['level']>=.28]
    exact=[r['stable_landing'] for r in rows if r['condition']=='exact']
    score=float(np.mean(severe))-.75*(1-float(np.mean(mild)))-.75*(1-float(np.mean(exact)))
    return dict(score=score,severe=float(np.mean(severe)),mild=float(np.mean(mild)),exact=float(np.mean(exact)),episodes=rows)

def make_vec(seed,n=8):
    cfg=core_config()
    def factory(index):
        def init():
            torch.set_num_threads(1)
            return LandingEnv(condition='exact' if index%4==0 else 'estimated',config=cfg)
        return init
    v=SubprocVecEnv([factory(j) for j in range(n)],start_method='spawn');v.seed(seed);return v

def sequences(dones,length=96):
    T,N=dones.shape;items=[]
    for j in range(N):
        episode_start=0;start=0
        while start<T:
            end=min(T,start+length);hits=np.flatnonzero(dones[start:end,j])
            if len(hits):end=start+int(hits[0])+1
            pre=min(32,start-episode_start);items.append((start,end,j,start-pre,pre))
            if dones[end-1,j]:episode_start=end
            start=end
    return items

def seq_batch(items,obs,states):
    length=max(b-a for a,b,*_ in items);x=np.zeros((len(items),length,DIM),np.float32);mask=np.zeros((len(items),length),np.float32);initial=[];groups={}
    for k,(a,b,j,p,pre) in enumerate(items):
        x[k,:b-a]=obs[a:b,j];mask[k,:b-a]=1.;initial.append(tree_map(lambda t:t[j:j+1].cuda(),states[p]));groups.setdefault(pre,[]).append(k)
    initial=concatenate(initial);state=tree_map(lambda t:t.clone(),initial)
    return tensor(x),tensor(mask),state,groups

def advance_burnin(m,items,obs,state,groups):
    with torch.no_grad():
        for pre,indices in groups.items():
            if pre==0:continue
            prefix=np.stack([obs[items[k][3]:items[k][0],items[k][2]] for k in indices]);_,new=m.encoder(tensor(prefix),select(state,indices))
            def put(dst,src):
                if isinstance(dst,(list,tuple)):
                    for a,b in zip(dst,src):put(a,b)
                else:dst[indices]=src
            put(state,new)
    return detach(state)

def padded(items,array,length,canonical_actions=False):
    out=np.zeros((len(items),length,*array.shape[2:]),np.float32)
    if canonical_actions:out[...,4:]=-1.
    for k,(a,b,j,*_) in enumerate(items):out[k,:b-a]=array[a:b,j]
    return tensor(out)

def mixed_kl(m,mu,reference_mu,reference_std):
    p,q=m.distribution(mu);rp=torch.distributions.Normal(reference_mu[...,:5],reference_std.exp())
    rq=torch.distributions.Bernoulli(logits=reference_mu[...,5])
    return torch.distributions.kl_divergence(q,rq)+q.probs*torch.distributions.kl_divergence(p,rp).sum(-1)

def validation_routes(dest):
    return range(527100,527116) if dest.parent.name=='hpo' else range(528000,528008)

def finetune(m,dataset,data,seed,dest,budget=BUDGET,method='manual'):
    total_start=time.time();reward_seconds=0.;reference=copy.deepcopy(m).eval()
    if m.cfg['arch']=='lstm':reference.encoder.core.flatten_parameters()
    for p in reference.parameters():p.requires_grad_(False)
    reward=None;reward_metadata={};rdata=dict(data,a=data['executed'],train=data['train']&(data['s'][:,-1]<.5),validation=data['validation']&(data['s'][:,-1]<.5),records=[r for r in data['records'] if r.get('configuration',{}).get('intervention','residual')=='residual'])
    if method in ['meirl','rime']:
        reward_started=time.time()
        from .rewards import MEIRL,RIME
        reward={'meirl':MEIRL,'rime':RIME}[method](rdata)
        if (dest/'reward_initial.pt').exists():
            ck=torch.load(dest/'reward_initial.pt',map_location='cuda',weights_only=False);reward.load_state_dict(ck['state']);reward_metadata=ck['metadata']
            if method=='meirl':reward.elites=ck['elites']
        else:
            reward_metadata=reward.fit_dynamics(rdata,seed) if method=='meirl' else reward.fit(rdata,seed)
            reward_metadata['calibration']=reward.calibrate(rdata)
            torch.save(dict(state=reward.state_dict(),metadata=reward_metadata,elites=getattr(reward,'elites',None)),dest/'reward_initial.pt')
        reward.eval()
        if method=='rime':torch.save(dict(state=reward.state_dict(),metadata=reward_metadata),dest/'reward.pt')
    if reward is not None:reward_seconds=time.time()-reward_started
    v=make_vec(536000+seed);N=v.num_envs;T=256;obs=v.reset()[:,0];state=None;refstate=None
    action_generator=torch.Generator(device='cuda').manual_seed(seed+7189)
    opt=m.optimizer(m.cfg['lr']);rng=np.random.default_rng(seed+715);start=time.time();steps=0;validation_seconds=0.;evaluations=[]
    ts=time.time();base=validation(m,seeds=validation_routes(dest));validation_seconds+=time.time()-ts;best=base['score'];m.save(dest/'best.pt',method=method,steps=0,validation_score=best)
    evaluations.append(dict(steps=0,**base));save(dest/'validation.json',evaluations)
    try:
        while steps<budget:
            m.eval();observations=[];actions=[];logps=[];values=[];rewards=[];dones=[];memories=[];refmus=[];nextobs=[]
            for t in range(T):
                if state is None:state=m.encoder.initial(N)
                memories.append(tree_map(lambda x:x.detach().cpu().clone(),state));observations.append(obs.copy())
                with torch.no_grad():
                    a,lp,val,state=m.act(tensor(obs),state,generator=action_generator);rmu,_,refstate=reference(tensor(obs)[:,None],refstate);rmu=rmu[:,0]
                nxt,raw_reward,done,infos=v.step(a.cpu().numpy());rr=raw_reward.copy()
                if method=='csil':
                    with torch.no_grad():
                        prior=m.prior_log_prob(a);cloned=reference.log_prob(rmu,a)
                        alpha=float(m.cfg.get('csil_alpha',.01));beta=.5*alpha
                        learned=alpha*(cloned-prior)-beta*(lp-prior)
                    rr=np.asarray([i['task_reward'] for i in infos],np.float32)+learned.clamp(-.005,.005).cpu().numpy()
                elif reward is not None:
                    with torch.no_grad():
                        ns=np.stack([i['next_state'] for i in infos]);style=reward(tensor(obs),a,tensor(ns),tensor(done))
                    rr=np.asarray([i['task_reward'] for i in infos],np.float32)+.001*style.cpu().numpy()
                # All30sdeadline events are actual task terminals. No value is added.
                if any(i.get('TimeLimit.truncated',False) for i in infos):raise RuntimeError('Unexpected external truncation in fixed-deadline task')
                actions.append(a.cpu().numpy());logps.append(lp.cpu().numpy());values.append(val.cpu().numpy());rewards.append(rr/100.);dones.append(done);refmus.append(rmu.cpu().numpy())
                nextobs.append(np.stack([i['next_state'] for i in infos]))
                keep=tensor(~done);state=tree_map(lambda x:x*keep.reshape(N,*([1]*(x.ndim-1))),detach(state));refstate=tree_map(lambda x:x*keep.reshape(N,*([1]*(x.ndim-1))),detach(refstate))
                obs=nxt[:,0]
            with torch.no_grad():_,last,_=m(tensor(obs)[:,None],state);last=last[:,0].cpu().numpy()
            observations=np.asarray(observations);actions=np.asarray(actions);logps=np.asarray(logps);values=np.asarray(values);rewards=np.asarray(rewards);dones=np.asarray(dones);refmus=np.asarray(refmus)
            if method=='meirl' and steps<65536:
                learner=dict(s=observations.reshape(-1,DIM),a=actions.reshape(-1,6),done=dones.reshape(-1),logp=logps.reshape(-1))
                disc=reward.adversarial_update_mixed(rdata,learner,m,seed+steps,updates=8);reward.calibrate(rdata)
                if steps+T*N>=65536:
                    reward_metadata.update(discovery_steps=65536,discriminator_loss=disc)
                    torch.save(dict(state=reward.state_dict(),metadata=reward_metadata,elites=reward.elites),dest/'reward.pt')
            adv=np.zeros_like(rewards);acc=np.zeros(N)
            for t in range(T-1,-1,-1):
                boot=last if t==T-1 else values[t+1];delta=rewards[t]+GAMMA*boot*(~dones[t])-values[t]
                acc=delta+GAMMA*float(m.cfg.get('gae_lambda',LAMBDA))*(~dones[t])*acc;adv[t]=acc
            ret=adv+values;adv=(adv-adv.mean())/(adv.std()+1e-8);items=sequences(dones);losses=[];kls=[]
            progress=steps/max(budget,1);lr=m.cfg['lr']*min(1.,(steps+2048)/16384)*(.1+.9*.5*(1+np.cos(np.pi*progress)))
            for group in opt.param_groups:group['lr']=lr
            m.train();stop=False
            for epoch in range(3):
                rng.shuffle(items)
                for first in range(0,len(items),8):
                    batch=items[first:first+8];x,mask,initial,groups=seq_batch(batch,observations,memories);initial=advance_burnin(m,batch,observations,initial,groups)
                    mu,val,_=m(x,initial);L=x.shape[1];a=padded(batch,actions,L,canonical_actions=True);oldlp=padded(batch,logps[...,None],L).squeeze(-1);aa=padded(batch,adv[...,None],L).squeeze(-1);target=padded(batch,ret[...,None],L).squeeze(-1)
                    oldmu=padded(batch,refmus,L);lp=m.log_prob(mu,a);ratio=(lp-oldlp).exp();den=mask.sum().clamp_min(1)
                    actor=(-torch.minimum(ratio*aa,ratio.clamp(.9,1.1)*aa)*mask).sum()/den;vf=((val-target).square()*mask).sum()/den
                    kl=(mixed_kl(m,mu,oldmu,reference.log_std)*mask).sum()/den;approx=(((ratio-1)-(lp-oldlp))*mask).sum()/den
                    if steps<16384:loss=.5*vf
                    else:
                        sx,ta,ma,va=dataset.sample(rng,batch=4,length=48,burnin=32);smu,_,_=m(tensor(sx));bc=clone_loss(smu,tensor(ta),tensor(ma),tensor(va))
                        loss=actor+.5*vf+.01*kl+(.1+.4*(1-progress))*bc
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite recurrent PPO loss')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
                    with torch.no_grad():m.log_std.clamp_(-4.,-1.8)
                    losses.append(float(loss));kls.append(float(kl))
                    if float(approx)>.015:stop=True;break
                if stop:break
            steps+=T*N;m.eval()
            if steps%16384==0:
                record=dict(steps=steps,total=budget,method=method,seconds=time.time()-start,loss=float(np.mean(losses)),bc_kl=float(np.mean(kls)),lr=lr)
                save(dest/'progress.json',record);print('PPO',m.cfg['arch'],method,seed,steps,'/',budget,'seconds',round(time.time()-start),flush=True)
            if steps%131072==0 or steps>=budget:
                ts=time.time();result=validation(m,seeds=validation_routes(dest));validation_seconds+=time.time()-ts;evaluations.append(dict(steps=steps,**result));save(dest/'validation.json',evaluations)
                if result['score']>best:best=result['score'];m.save(dest/'best.pt',method=method,steps=steps,validation_score=best)
                m.save(dest/'latest.pt',method=method,steps=steps)
        m.save(dest/'final.pt',method=method,steps=steps)
        save(dest/'complete.json',dict(started_epoch=total_start,ppo_started_epoch=start,finished_epoch=time.time(),steps=steps,method=method,gae_lambda=float(m.cfg.get('gae_lambda',LAMBDA)),seconds=time.time()-total_start,ppo_seconds=time.time()-start,reward_seconds=reward_seconds,setup_seconds=start-total_start-reward_seconds,validation_seconds=validation_seconds,base_score=base['score'],best_score=best,selected_steps=max(evaluations,key=lambda x:x['score'])['steps'],reward_metadata=reward_metadata))
    finally:v.close()

def hyperparameters(data,dataset):
    for arch in ARCHS:
        target=OUT/'hpo'/f'{arch}_selection.json'
        if target.exists():continue
        base=starting_config(arch);configs=[dict(base,lr=lr,gae_lambda=lam) for lam in [.97,.995] for lr in [base['lr'],base['lr']*2]]
        results=[]
        for index,cfg in enumerate(configs):
            dest=OUT/'hpo'/f'{arch}_{index}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'result.json').exists():results.append(json.loads((dest/'result.json').read_text()));continue
            setup(97);m=Agent(cfg).cuda();pretrain(m,dataset,97,dest,updates=400);finetune(m,dataset,data,97,dest,budget=65536)
            vv=json.loads((dest/'validation.json').read_text());result=dict(index=index,config=cfg,validation=max(vv,key=lambda x:x['score']));save(dest/'result.json',result);results.append(result);del m;torch.cuda.empty_cache()
        choice=max(results,key=lambda x:x['validation']['score']);save(target,dict(config=choice['config'],selected_index=choice['index'],candidates=results,rule='Reserved HPO routes,2LRx2GAE_lambda,shared400BC+65536PPO per candidate'))
    close_pool()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['hpo','train','rewards'],default='train');p.add_argument('--worker',type=int,default=0);p.add_argument('--budget',type=int,default=BUDGET);args=p.parse_args()
    data=load();dataset=SequenceData(data)
    if args.phase=='hpo':hyperparameters(data,dataset);return
    jobs=[(a,s,'manual') for a in ARCHS for s in SEEDS] if args.phase=='train' else [('lstm',s,method) for method in ['csil','meirl','rime'] for s in SEEDS]
    for index,(arch,seed,method) in enumerate(jobs):
        if index%2!=args.worker:continue
        group='models' if method=='manual' else 'reward_models';dest=OUT/group/(arch if method=='manual' else method)/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        setup(seed);cfg=json.loads((OUT/'hpo'/f'{arch}_selection.json').read_text())['config']
        if method!='manual':
            source=OUT/'models'/'lstm'/f'seed_{seed}'/'bc.pt';m=Agent.load(source);m.save(dest/'bc.pt',method=method,shared_bc=str(source))
        elif (dest/'bc.pt').exists():m=Agent.load(dest/'bc.pt')
        else:m=Agent(cfg).cuda();pretrain(m,dataset,seed,dest)
        finetune(m,dataset,data,seed,dest,budget=args.budget,method=method);del m;torch.cuda.empty_cache()
    close_pool()
if __name__=='__main__':main()
