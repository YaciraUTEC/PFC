"""Matched recurrent PPO with protected BC and training-only noise curricula."""
import argparse,copy,json,time
from collections import Counter
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from . import train_utils as old
from ..round5.bc import setup,tensor,pretrain
from ..round5.losses import clone_loss
from ..round4.models import tree_map,detach
from .models import Agent,ARCHS,configuration
from .env import DIM
from .curriculum import CurriculumEnv,NoiseSchedule
from .data import load,SequenceData
from .evaluate import OUT,save,evaluate,SEEDS
from .phase1 import objective
from .resource_control import validation_slot
BUDGET=524288

def selected_config():
    path=OUT/'phase1_conclusion.json'
    if not path.exists() or not (OUT/'phase1_closed.json').exists() or not (OUT/'phase1_physical_closed.json').exists() or not (OUT/'phase1_corrections_closed.json').exists():raise RuntimeError('Closed phase1 conclusion required before training')
    return dict(json.loads(path.read_text())['config'],gate_mode='learned')
def validation(m,config=None,seed_start=611000,count=4):
    with validation_slot(OUT):return _validation(m,config,seed_start,count)

def _validation(m,config=None,seed_start=611000,count=4):
    cfg=selected_config() if config is None else dict(config)
    reference=copy.deepcopy(m).cpu().double().eval();rows=[]
    for condition in ['exact','estimated']:
        for noise in [1.,2.]:
            result=evaluate(dict(cfg,noise_multiplier=noise),condition,range(seed_start,seed_start+count),[0.,.25,.28,.29,.295,.296],reference,workers=8)
            rows.extend(dict(r,condition=condition,noise=noise) for r in result['episodes'])
    return dict(score=objective(rows),episodes=rows)
def make_vec(config,seed,n=8):
    def factory(j):
        def init():
            torch.set_num_threads(1)
            return CurriculumEnv(training_seed=seed+j,condition='exact' if j%4==0 else 'estimated',config=config)
        return init
    v=SubprocVecEnv([factory(j) for j in range(n)],start_method='spawn');v.seed(630000+seed);return v

def finetune(m,dataset,seed,dest,schedule_kind,budget=BUDGET):
    config=selected_config();schedule=NoiseSchedule(schedule_kind,budget);start=time.time();validation_seconds=0.
    reference=copy.deepcopy(m).eval()
    for p in reference.parameters():p.requires_grad_(False)
    v=make_vec(config,seed);N=v.num_envs;T=256;v.env_method('set_noise_upper',schedule.upper)
    obs=v.reset()[:,0];state=None;refstate=None
    opt=m.optimizer(m.cfg['lr']);rng=np.random.default_rng(seed+815);steps=0;validations=[]
    ts=time.time();base=validation(m);validation_seconds+=time.time()-ts;best=base['score']
    m.save(dest/'best.pt',steps=0,schedule=schedule_kind,validation_score=best);validations.append(dict(steps=0,**base));save(dest/'validation.json',validations)
    episode_counts=Counter();completed_steps=Counter()
    episodes=[];generator=torch.Generator(device='cuda').manual_seed(seed+77189)
    try:
        while steps<budget:
            m.eval();observations=[];actions=[];logps=[];values=[];rewards=[];dones=[];memories=[];refmus=[]
            for t in range(T):
                if state is None:state=m.encoder.initial(N)
                memories.append(tree_map(lambda x:x.detach().cpu().clone(),state));observations.append(obs.copy())
                with torch.no_grad():
                    a,lp,val,state=m.act(tensor(obs),state,generator=generator);rmu,_,refstate=reference(tensor(obs)[:,None],refstate);rmu=rmu[:,0]
                nxt,rr,done,infos=v.step(a.cpu().numpy())
                actions.append(a.cpu().numpy());logps.append(lp.cpu().numpy());values.append(val.cpu().numpy());rewards.append(rr/100.);dones.append(done);refmus.append(rmu.cpu().numpy())
                episodes.extend(i for d,i in zip(done,infos) if d)
                for d,i in zip(done,infos):
                    if d:episode_counts[str(i['level'])]+=1;completed_steps[str(i['level'])]+=i['episode_steps']
                keep=tensor(~done);state=tree_map(lambda x:x*keep.reshape(N,*([1]*(x.ndim-1))),detach(state));refstate=tree_map(lambda x:x*keep.reshape(N,*([1]*(x.ndim-1))),detach(refstate));obs=nxt[:,0]
            with torch.no_grad():_,last,_=m(tensor(obs)[:,None],state);last=last[:,0].cpu().numpy()
            observations=np.asarray(observations);actions=np.asarray(actions);logps=np.asarray(logps);values=np.asarray(values);rewards=np.asarray(rewards);dones=np.asarray(dones);refmus=np.asarray(refmus)
            adv=np.zeros_like(rewards);acc=np.zeros(N)
            for t in range(T-1,-1,-1):
                boot=last if t==T-1 else values[t+1];delta=rewards[t]+.999*boot*(~dones[t])-values[t]
                acc=delta+.999*.995*(~dones[t])*acc;adv[t]=acc
            returns=adv+values;adv=(adv-adv.mean())/(adv.std()+1e-8);items=old.sequences(dones,length=96);losses=[];kls=[]
            progress=steps/max(budget,1);lr=m.cfg['lr']*min(1.,(steps+2048)/16384)*(.1+.9*.5*(1+np.cos(np.pi*progress)))
            for group in opt.param_groups:group['lr']=lr
            m.train();stop=False
            for epoch in range(3):
                rng.shuffle(items)
                for first in range(0,len(items),8):
                    batch=items[first:first+8];x,mask,initial,groups=old.seq_batch(batch,observations,memories);initial=old.advance_burnin(m,batch,observations,initial,groups)
                    mu,val,_=m(x,initial);L=x.shape[1];a=old.padded(batch,actions,L,canonical_actions=True);oldlp=old.padded(batch,logps[...,None],L).squeeze(-1);aa=old.padded(batch,adv[...,None],L).squeeze(-1);target=old.padded(batch,returns[...,None],L).squeeze(-1)
                    oldmu=old.padded(batch,refmus,L);lp=m.log_prob(mu,a);ratio=(lp-oldlp).exp();den=mask.sum().clamp_min(1)
                    actor=(-torch.minimum(ratio*aa,ratio.clamp(.9,1.1)*aa)*mask).sum()/den;vf=((val-target).square()*mask).sum()/den
                    kl=(old.mixed_kl(m,mu,oldmu,reference.log_std)*mask).sum()/den;approx=(((ratio-1)-(lp-oldlp))*mask).sum()/den
                    if steps<16384:loss=.5*vf
                    else:
                        sx,ta,ma,va=dataset.sample(rng,batch=4,length=48,burnin=32);smu,_,_=m(tensor(sx));bc=clone_loss(smu,tensor(ta),tensor(ma),tensor(va))
                        loss=actor+.5*vf+.02*kl+(.2+.4*(1-progress))*bc
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
                    with torch.no_grad():m.log_std.clamp_(-4.,-1.8)
                    losses.append(float(loss));kls.append(float(kl))
                    if float(approx)>.015:stop=True;break
                if stop:break
            steps+=T*N;m.eval()
            if steps%16384==0:
                upper=schedule.update(steps,episodes);episodes=[];v.env_method('set_noise_upper',upper)
                record=dict(steps=steps,total=budget,schedule=schedule_kind,seconds=time.time()-start,loss=float(np.mean(losses)),bc_kl=float(np.mean(kls)),lr=lr,noise_upper=upper)
                save(dest/'progress.json',record);save(dest/'schedule.json',schedule.decisions);print('PPO',m.cfg['arch'],schedule_kind,seed,steps,round(time.time()-start),upper,flush=True)
            if steps%131072==0 or steps>=budget:
                ts=time.time();result=validation(m);validation_seconds+=time.time()-ts;validations.append(dict(steps=steps,**result));save(dest/'validation.json',validations)
                if result['score']>best:best=result['score'];m.save(dest/'best.pt',steps=steps,schedule=schedule_kind,validation_score=best)
                m.save(dest/'latest.pt',steps=steps,schedule=schedule_kind)
        m.save(dest/'final.pt',steps=steps,schedule=schedule_kind)
        save(dest/'complete.json',dict(steps=steps,started_epoch=start,finished_epoch=time.time(),seconds=time.time()-start,validation_seconds=validation_seconds,best_score=best,seed=seed,schedule=schedule_kind,config=config,cloning_noise_max=1.,dataset_sha256=json.loads((OUT/'dataset_fingerprint.json').read_text())['sha256'] if (OUT/'dataset_fingerprint.json').exists() else 'smoke_initial_data',gae_lambda=.995,gamma=.999,episodes_by_level=dict(episode_counts),completed_steps_by_level=dict(completed_steps)))
    finally:v.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--budget',type=int,default=BUDGET);p.add_argument('--schedules',nargs='+',default=['nominal','linear','adaptive','full']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.smoke:a.budget=max(a.budget,18432)
    else:
        while not (OUT/'expert_data_complete.json').exists() or not (OUT/'dataset_fingerprint.json').exists() or not (OUT/'rescue_expert_data_complete.json').exists():time.sleep(5)
    data=load();dataset=SequenceData(data,max_noise=1.)
    jobs=[(arch,seed,sch) for arch in ARCHS for seed in SEEDS for sch in a.schedules]
    if a.smoke:jobs=[('mamba3',197,'linear'),('gtrxl',197,'adaptive')]
    for index,(arch,seed,sch) in enumerate(jobs):
        # All schedules of one model seed share exactly one BC writer.
        group=index if a.smoke else ARCHS.index(arch)*len(SEEDS)+SEEDS.index(seed)
        if group%a.workers!=a.worker:continue
        dest=OUT/'phase2'/arch/f'seed_{seed}'/sch;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        bcdir=OUT/'bc'/arch/f'seed_{seed}';bcdir.mkdir(parents=True,exist_ok=True);setup(seed)
        data_hash=json.loads((OUT/'dataset_fingerprint.json').read_text())['sha256']+':nominal_BC:physics_only_onsets' if not a.smoke else 'smoke_initial_data'
        prior_hash=json.loads((bcdir/'data_version.json').read_text())['sha256'] if (bcdir/'data_version.json').exists() else None
        if prior_hash!=data_hash or not (bcdir/'bc.pt').exists() or not (bcdir/'bc_timing.json').exists():
            m=Agent(configuration(arch)).cuda();pretrain(m,dataset,seed,bcdir,updates=1200 if not a.smoke else 8)
            save(bcdir/'data_version.json',dict(sha256=data_hash))
        m=Agent.load(bcdir/'bc.pt','cuda');finetune(m,dataset,seed,dest,sch,budget=a.budget)
        del m;torch.cuda.empty_cache()
if __name__=='__main__':main()
