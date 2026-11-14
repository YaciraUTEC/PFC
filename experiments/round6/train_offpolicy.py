"""Frozen-memory replay and matched online budgets for FlashSAC/FQL."""
import argparse,copy,json,time,math
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .offpolicy import OffPolicy,compress_actions,normalize,project_distribution
from .vendor_flash.reward_normalization import RewardNormalizer
from .models import Agent,ARCHS,physics_observation
from .data import load
from .train import validation,make_vec,selected_config
from .curriculum import NoiseSchedule
from .evaluate import OUT,save,SEEDS
from ..round5.bc import setup
from ..round4.models import tree_map,detach

def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)
def behavior_actions(model,features,proposal,exploration_mask):
    selected=model.select(features,proposal)
    explore=torch.as_tensor(exploration_mask,device=features.device,dtype=torch.bool).reshape(-1,1)
    selected=torch.where(explore,proposal,selected)
    selected[0]=0.
    return selected

class Replay:
    def __init__(self,dim,capacity=1600000):
        self.capacity=capacity;self.n=0;self.position=0
        self.x=np.empty((capacity,dim),np.float32);self.nx=np.empty_like(self.x);self.a=np.empty((capacity,4),np.float32)
        self.r=np.empty(capacity,np.float32);self.done=np.empty(capacity,np.float32)
    def append(self,x,a,r,nx,done):
        n=len(x)
        if n>self.capacity:x=x[-self.capacity:];a=a[-self.capacity:];r=r[-self.capacity:];nx=nx[-self.capacity:];done=done[-self.capacity:];n=len(x)
        ids=(np.arange(n)+self.position)%self.capacity
        self.x[ids]=x;self.a[ids]=a;self.r[ids]=r;self.nx[ids]=nx;self.done[ids]=done
        self.position=(self.position+n)%self.capacity;self.n=min(self.capacity,self.n+n)
    def sample(self,rng,batch=512):
        ids=rng.integers(self.n,size=batch)
        return tuple(tensor(a[ids]) for a in [self.x,self.a,self.r,self.nx,self.done])
def encode_data(m,data):
    n=len(data['s']);x=np.empty((n,m.dim),np.float32);nx=np.empty_like(x)
    start=time.time()
    for j,r in enumerate(data['records']):
        state=None;begin,end=r['start'],r['stop'];raw=np.concatenate([data['s'][begin:end],data['ns'][end-1:end]],0)
        parts=[]
        with torch.no_grad():
            for k in range(0,len(raw),128):
                ob=tensor(raw[k:k+128])[None];z,state=m.prior.encoder(ob,state);parts.append(torch.cat([physics_observation(ob),z],-1)[0].cpu().numpy());state=detach(state)
        features=np.concatenate(parts);x[begin:end]=features[:-1];nx[begin:end]=features[1:]
    return x,nx,time.time()-start

def actor_prior(m,x):
    with torch.no_grad():
        mu=m.prior.actor(x[:,-m.prior.cfg['width']:]);out=mu[:,:5].clamp(-5,5).tanh()
        return torch.where(mu[:,5:]>0,out[:,:4]*(out[:,4:5]+1)/2,torch.zeros_like(out[:,:4]))

def initialize_heads(m,replay,rng,updates=1500):
    opt=torch.optim.Adam([p for n,p in m.named_parameters() if n.startswith('actor.') or n.startswith('flow.')],lr=3e-4)
    start=time.time()
    for it in range(updates):
        x,a,rr,nx,done=replay.sample(rng,512)
        if m.method=='flashsac':
            mean,_=m.actor.get_mean_and_std(x,training=True);loss=F.mse_loss(mean.tanh(),actor_prior(m,x))
        else:
            noise=torch.randn_like(a);t=torch.rand((len(a),1),device=a.device);xt=(1-t)*noise+t*a
            flow=F.mse_loss(m.flow(torch.cat([x,xt,t],-1)),a-noise)
            target=m.flow_actions(x,noise);proposal=m.actor(torch.cat([x,noise],-1));loss=flow+F.mse_loss(proposal,target)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        if m.method=='flashsac':normalize(m.actor)
    return time.time()-start

class Trainer:
    def __init__(self,m,target_sigma=.15,fql_alpha=10.):
        self.m=m;self.update_count=0;self.target_sigma=float(target_sigma);self.fql_alpha=float(fql_alpha)
        self.actor_opt=torch.optim.Adam(m.actor.parameters(),lr=3e-4)
        self.critic_opt=torch.optim.Adam(m.critic.parameters(),lr=3e-4)
        self.benefit_opt=torch.optim.Adam(m.benefit.parameters(),lr=3e-4)
        if m.method=='flashsac':self.alpha_opt=torch.optim.Adam([m.log_alpha],lr=3e-4)
        else:self.flow_opt=torch.optim.Adam(m.flow.parameters(),lr=3e-4)
        self.normalizer=RewardNormalizer(.999,5.,True,torch.device('cuda'))
    def update(self,batch):
        m=self.m;x,a,r,nx,done=batch
        with torch.no_grad():
            next_candidate,_=m.proposal(nx);next_deployed=m.select(nx,next_candidate,target=True)
            next_benefit=m.benefit_q(nx,next_deployed,target=True).min(0).values
            benefit_target=r/100.+.999*(1-done)*next_benefit
        benefit_loss=(m.benefit_q(x,a)-benefit_target[None]).square().mean()
        self.benefit_opt.zero_grad();benefit_loss.backward();torch.nn.utils.clip_grad_norm_(m.benefit.parameters(),1.);self.benefit_opt.step()
        if m.method=='flashsac':
            with torch.no_grad():
                next_action,lp=m.proposal(nx)
                all_x=torch.cat([x,nx]);all_a=torch.cat([a,next_action]);qs,info=m.target(all_x,all_a,training=True)
                nq=qs[:,len(x):];logs=info['log_prob'][:,len(x):];idx=nq.argmin(0);logs=logs[idx,torch.arange(len(x),device=x.device)]
                entropy=m.log_alpha.exp()*lp
                # Same official running-return normalization; initial offline
                # rewards use the initialization prior until online return statistics accumulate.
                rr=self.normalizer.normalize_rewards(r)
                target=project_distribution(logs,rr,done,entropy)
            qs,info=m.critic(all_x,all_a,training=True);q_loss=-(target[None]*info['log_prob'][:,:len(x)]).sum(-1).mean()
        else:
            with torch.no_grad():
                na,_=m.proposal(nx);nq=m.q(nx,na,target=True).mean(0)
                target=r/100.+.999*(1-done)*nq
            pred=m.q(x,a);q_loss=(pred-target[None]).square().mean()
        if not torch.isfinite(q_loss):raise RuntimeError('Nonfinite critic loss')
        self.critic_opt.zero_grad();q_loss.backward();torch.nn.utils.clip_grad_norm_(m.critic.parameters(),1.);self.critic_opt.step()
        if m.method=='flashsac':normalize(m.critic)
        actor_loss=None;bc_loss=None
        if self.update_count%2==0 or m.method=='fql':
            for p in m.critic.parameters():p.requires_grad_(False)
            if m.method=='flashsac':
                # Obtain log probabilities in the same cross-batch actor pass.
                # The sampled action's exact log density is returned by actor.
                combined,infos=m.actor(torch.cat([x,nx]),training=True);action=combined[:len(x)];lp=infos['log_prob'][:len(x)]
                q=m.q(x,action).min(0).values
                prior=actor_prior(m,x);bc_loss=F.mse_loss(action,prior)
                actor_loss=(m.log_alpha.exp().detach()*lp-q).mean()+.1*q.abs().mean().detach()*bc_loss
            else:
                noise=torch.randn_like(a);t=torch.rand((len(a),1),device=x.device);xt=(1-t)*noise+t*a
                bc_loss=F.mse_loss(m.flow(torch.cat([x,xt,t],-1)),a-noise)
                self.flow_opt.zero_grad();bc_loss.backward();torch.nn.utils.clip_grad_norm_(m.flow.parameters(),1.);self.flow_opt.step()
                noise=torch.randn_like(a);target_flow=m.flow_actions(x,noise);raw=m.actor(torch.cat([x,noise],-1));action=raw.clamp(-1,1)
                q=m.q(x,action).mean(0);actor_loss=self.fql_alpha*F.mse_loss(raw,target_flow)-q.mean()
            self.actor_opt.zero_grad();actor_loss.backward();torch.nn.utils.clip_grad_norm_(m.actor.parameters(),1.);self.actor_opt.step()
            if m.method=='flashsac':
                normalize(m.actor);entropy=-lp.detach().mean();target_entropy=.5*4*math.log(2*math.pi*math.e*self.target_sigma**2)
                alpha_loss=m.log_alpha.exp()*(entropy-target_entropy);self.alpha_opt.zero_grad();alpha_loss.backward();self.alpha_opt.step()
                with torch.no_grad():m.log_alpha.clamp_(-10.,1.)
            for p in m.critic.parameters():p.requires_grad_(True)
        with torch.no_grad():
            for p,tp in zip(m.critic.parameters(),m.target.parameters()):tp.lerp_(p,.005 if m.method=='fql' else .01)
            for p,tp in zip(m.benefit.parameters(),m.benefit_target.parameters()):tp.lerp_(p,.005)
        self.update_count+=1
        return dict(critic_loss=float(q_loss),benefit_loss=float(benefit_loss),actor_loss=None if actor_loss is None else float(actor_loss),bc_loss=None if bc_loss is None else float(bc_loss))

def fit(arch,seed,method,dest,budget=524288,offline_updates=100000,target_sigma=.15,fql_alpha=10.,prior_seed=None,validation_start=611000,validation_count=4):
    total_started=time.time();setup(seed)
    phase2=json.loads((OUT/'phase2_conclusion.json').read_text());sch=phase2['schedule_by_arch'][arch]
    prior_seed=seed if prior_seed is None else prior_seed
    prior=Agent.load(OUT/'phase2'/arch/f'seed_{prior_seed}'/sch/'best.pt','cuda');m=OffPolicy(prior,method).cuda().eval();m.evaluation_seed=seed
    data_started=time.time();data=load();data_seconds=time.time()-data_started;start=time.time();features,next_features,encode_seconds=encode_data(m,data)
    replay=Replay(m.dim,max(1600000,len(features)+budget));replay.append(features,compress_actions(data['executed']),data['reward'],next_features,data['done']);del features,next_features,data
    rng=np.random.default_rng(seed+915);initial_seconds=initialize_heads(m,replay,rng);trainer=Trainer(m,target_sigma=target_sigma,fql_alpha=fql_alpha)
    # Initial return-scale prior, not a measured offline variance; online stats then adapt.
    trainer.normalizer.G_r_max.fill_(100.);trainer.normalizer.G_rms.var.fill_(400.)
    off_start=time.time()
    if method=='fql':
        for it in range(offline_updates):
            metric=trainer.update(replay.sample(rng))
            if it%1000==0:save(dest/'offline_progress.json',dict(update=it,total=offline_updates,seconds=time.time()-off_start,**metric));print('FQL_OFFLINE',arch,seed,it,flush=True)
    offline_seconds=time.time()-off_start
    schedule=NoiseSchedule(sch,budget);config=selected_config();v=make_vec(config,seed+10000);N=v.num_envs;v.env_method('set_noise_upper',schedule.upper);obs=v.reset()[:,0];state=None
    opt_start=time.time();steps=0;validation_seconds=0.;validations=[];best=-np.inf;episodes=[];episode_counts=Counter();completed_steps=Counter()
    fixed_noise=torch.zeros((N,4),device='cuda');remaining=np.zeros(N,int)
    try:
        ts=time.time();result=validation(m,seed_start=validation_start,count=validation_count);validation_seconds+=time.time()-ts;best=result['score'];validations.append(dict(steps=0,**result));m.save(dest/'best.pt',steps=0,validation_score=best)
        while steps<budget:
            with torch.no_grad():
                x,state=m.features(tensor(obs),state)
                if method=='flashsac':
                    refresh=remaining<=0
                    if refresh.any():fixed_noise[refresh]=torch.randn((int(refresh.sum()),4),device='cuda');remaining[refresh]=np.minimum(rng.zipf(2.,int(refresh.sum())),16)
                    proposal,_=m.proposal(x,noise=fixed_noise);remaining-=1
                else:proposal,_=m.proposal(x)
                # Explicit baseline behavior and exploratory candidate exposure.
                a=behavior_actions(m,x,proposal,rng.random(N)<.25)
                amount=a.abs().amax(-1);direction=a/amount.clamp_min(1e-9)[:,None]
                action=torch.cat([direction,(2*amount-1)[:,None],torch.where(amount>.002,1.,-1.)[:,None]],-1).cpu().numpy()
            nxt,reward,done,infos=v.step(action)
            actual=np.stack([i['next_state'] for i in infos]);executed=np.stack([i['executed_action'] for i in infos])
            with torch.no_grad():nx,_=m.features(tensor(actual),state)
            replay.append(x.cpu().numpy(),compress_actions(executed),reward,nx.cpu().numpy(),done)
            trainer.normalizer.update_reward_stats(tensor(reward),tensor(done).bool(),torch.zeros(N,dtype=torch.bool,device='cuda'))
            metric=trainer.update(replay.sample(rng));steps+=N;episodes.extend(i for d,i in zip(done,infos) if d)
            for d,i in zip(done,infos):
                if d:episode_counts[str(i['level'])]+=1;completed_steps[str(i['level'])]+=i['episode_steps']
            keep=tensor(~done);state=tree_map(lambda z:z*keep.reshape(N,*([1]*(z.ndim-1))),detach(state));remaining[done]=0;obs=nxt[:,0]
            if steps%16384==0:
                upper=schedule.update(steps,episodes);episodes=[];v.env_method('set_noise_upper',upper)
                save(dest/'progress.json',dict(steps=steps,total=budget,seconds=time.time()-opt_start,method=method,noise_upper=upper,updates=trainer.update_count,**metric));save(dest/'schedule.json',schedule.decisions)
                print('OFFPOLICY',method,arch,seed,steps,round(time.time()-opt_start),flush=True)
            if steps%131072==0 or steps>=budget:
                ts=time.time();result=validation(m,seed_start=validation_start,count=validation_count);validation_seconds+=time.time()-ts;validations.append(dict(steps=steps,**result));save(dest/'validation.json',validations)
                if result['score']>best:best=result['score'];m.save(dest/'best.pt',steps=steps,validation_score=best)
                m.save(dest/'latest.pt',steps=steps)
        m.save(dest/'final.pt',steps=steps);save(dest/'validation.json',validations)
        save(dest/'complete.json',dict(method=method,arch=arch,seed=seed,steps=steps,started_epoch=total_started,finished_epoch=time.time(),seconds=time.time()-total_started,data_seconds=data_seconds,encode_seconds=encode_seconds,initial_seconds=initial_seconds,offline_seconds=offline_seconds,online_with_validation_seconds=time.time()-opt_start,validation_seconds=validation_seconds,best_score=best,schedule=sch,encoder='frozen_selected_PPO_memory',prior_seed=prior_seed,target_sigma=target_sigma,fql_alpha=fql_alpha,offline_updates=offline_updates if method=='fql' else 0,critic_update_to_transition_ratio=1/N,strict_deadline_terminal=True,dataset_sha256=json.loads((OUT/'dataset_fingerprint.json').read_text())['sha256'],episodes_by_level=dict(episode_counts),completed_steps_by_level=dict(completed_steps)))
    finally:v.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--budget',type=int,default=524288);p.add_argument('--offline-updates',type=int,default=100000);a=p.parse_args()
    if not (OUT/'phase2_conclusion.json').exists():raise RuntimeError('Phase2 conclusion required')
    while not (OUT/'hpo_selection.json').exists():time.sleep(5)
    hpo=json.loads((OUT/'hpo_selection.json').read_text())
    jobs=[(arch,seed,method) for arch in ARCHS for seed in SEEDS for method in ['flashsac','fql']]
    for j,(arch,seed,method) in enumerate(jobs):
        worker=(ARCHS.index(arch)+SEEDS.index(seed)+(method=='fql'))%a.workers
        if worker!=a.worker:continue
        dest=OUT/'phase3'/method/arch/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        fit(arch,seed,method,dest,a.budget,a.offline_updates,**hpo['selected'][arch][method])
if __name__=='__main__':main()
