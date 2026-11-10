"""Data, bounded PPO and fixed evaluation utilities for the first IRL screen."""
from pathlib import Path
import json,time,copy
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from ..models import TemporalFeatures
from ..run import save
from .env import TrialEnv,LEVELS
OUT=Path(__file__).resolve().parents[2]/'runs/irl_round3'
SEEDS=[11,22,33,44,55]
ZERO=np.array([0.,0.,0.,0.,-1.],np.float32)
torch.set_num_threads(1)

class BoundedPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self,latent_pi):
        return SquashedDiagGaussianDistribution(self.action_space.shape[0]).proba_distribution(self.action_net(latent_pi),self.log_std)

class FlatState(__import__('gymnasium').ObservationWrapper):
    def __init__(self,e):
        super().__init__(e)
        self.observation_space=__import__('gymnasium').spaces.Box(-10,10,(48,),dtype=np.float32)
    def observation(self,o):return o[-1]

def make_vec(n=8,seed=0,flat=False):
    def init():
        torch.set_num_threads(1)
        e=TrialEnv(history=1 if flat else 32)
        return FlatState(e) if flat else e
    v=SubprocVecEnv([init for _ in range(n)],start_method='spawn') if n>1 else DummyVecEnv([init])
    v.seed(seed);return v

def new_policy(v,seed=11,flat=False):
    kw=dict(net_arch=dict(pi=[64,64],vf=[64,64]),ortho_init=False,log_std_init=-2.3)
    if not flat:
        kw.update(features_extractor_class=TemporalFeatures,features_extractor_kwargs=dict(architecture='lstm',width=64,layers=1,features_dim=64))
    m=PPO(BoundedPolicy,v,learning_rate=lambda p:2e-4*(.1+.9*p),n_steps=256,batch_size=256,n_epochs=5,
          gamma=.995,gae_lambda=.95,clip_range=.15,ent_coef=.001,target_kl=.015,max_grad_norm=.5,
          seed=seed,device='cuda',policy_kwargs=kw,verbose=0)
    nn.init.zeros_(m.policy.action_net.weight);nn.init.zeros_(m.policy.action_net.bias)
    with torch.no_grad():m.policy.action_net.bias[-1]=-2.
    return m

def episode(e,policy,seed,level=None,collect=False,mode='teacher'):
    o,_=e.reset(seed=int(seed),options={} if level is None else dict(level=float(level)))
    states=[];actions=[];nexts=[];tasks=[];styles=[];terminals=[];histories=[]
    rng=np.random.default_rng(seed+12345)
    while True:
        if policy is not None:a=policy.predict(o,deterministic=True)[0]
        elif mode=='teacher':a=e.teacher_action()
        elif mode=='zero':a=ZERO.copy()
        else:
            a=e.teacher_action();a[:4]+=rng.normal(0,.28,4);a[4]+=rng.normal(0,.4)
            if np.all(e.estimated_eta==1):a=ZERO.copy()
            a=np.clip(a,-.999,.999)
        before=o[-1].copy()
        if collect:histories.append(o.copy())
        o,r,d,t,info=e.step(a)
        if collect:
            states.append(before);actions.append(a.copy());nexts.append(info['next_state']);tasks.append(info['task_reward']);styles.append(info['manual_style']);terminals.append(d or t)
        if d or t:break
    row={k:v for k,v in info.items() if k not in ['state','action','next_state']}
    row.update(seed=int(seed),level=float(e.level),mode=mode)
    if not collect:return row
    return dict(s=np.asarray(states,np.float32),a=np.asarray(actions,np.float32),ns=np.asarray(nexts,np.float32),
        task=np.asarray(tasks,np.float32),style=np.asarray(styles,np.float32),done=np.asarray(terminals,bool),
        h=np.asarray(histories,np.float32)),row

def collect_data():
    if (OUT/'data.npz').exists():return
    OUT.mkdir(parents=True,exist_ok=True);start=time.time();records=[];arrays=[];offset=0
    e=TrialEnv()
    try:
        for i in range(210):
            level=LEVELS[i%7];seed=300000+i//7
            for mode in ['teacher','zero','perturbed']:
                arr,row=episode(e,None,seed,level,True,mode)
                row.update(start=offset,stop=offset+len(arr['s']),split='reward_train' if i<168 else 'reward_validation')
                offset=row['stop'];arrays.append(arr);records.append(row)
            if i%14==13:print('data',i+1,'/210 scenario routes,',offset,'transitions',flush=True)
    finally:e.close()
    # Histories are reconstructed from each episode on demand, reducing disk/RAM.
    keys=['s','a','ns','task','style','done']
    np.savez_compressed(OUT/'data.npz',**{k:np.concatenate([x[k] for x in arrays]) for k in keys})
    save(OUT/'data_episodes.json',dict(seconds=time.time()-start,episodes=records))

def load_data():
    z=dict(np.load(OUT/'data.npz'));rec=json.loads((OUT/'data_episodes.json').read_text())['episodes']
    z['records']=rec
    train=np.zeros(len(z['s']),bool);expert=np.zeros_like(train);validation=np.zeros_like(train)
    for r in rec:
        sel=slice(r['start'],r['stop'])
        if r['split']=='reward_train':
            train[sel]=True
            if r['mode']=='teacher' and r['mission_success']:expert[sel]=True
        else:validation[sel]=True
    z.update(train=train,expert=expert,validation=validation)
    return z

def histories(data,ids,length=32):
    # done at j means that j+1 starts a new episode.
    ends=np.r_[-1,np.flatnonzero(data['done'])]
    starts=ends[np.searchsorted(ends,ids,side='left')-1]+1
    offsets=ids[:,None]-np.arange(length-1,-1,-1)[None,:]
    valid=offsets>=starts[:,None]
    return data['s'][np.maximum(offsets,0)]*valid[:,:,None]

def pretrain_policy(m,data,seed=11,updates=1800,flat=False):
    rng=np.random.default_rng(seed);ids=np.flatnonzero(data['expert'])
    # Half post-detection / half all states avoids a trivial always-off clone.
    active=ids[np.any(data['s'][ids,31:35]>.001,axis=1)]
    opt=torch.optim.Adam([p for n,p in m.policy.named_parameters() if n!='log_std'],lr=3e-4)
    start=time.time();losses=[]
    m.policy.set_training_mode(True)
    for it in range(updates):
        ix=np.r_[rng.choice(ids,128),rng.choice(active,128)]
        obs=data['s'][ix] if flat else histories(data,ix)
        obs=torch.as_tensor(obs,device=m.device,dtype=torch.float32)
        target=torch.as_tensor(np.arctanh(np.clip(data['a'][ix],-.999,.999)),device=m.device)
        dist=m.policy.get_distribution(obs);mu=dist.distribution.mean
        loss=(mu-target).square().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.policy.parameters(),.5);opt.step()
        if it%300==0:losses.append(float(loss));print('BC',it,float(loss),flush=True)
    m.policy.set_training_mode(False)
    return dict(seconds=time.time()-start,loss=losses,updates=updates)

def evaluate(m=None,controller='updated',condition='uncertain',seeds=range(200000,200100),levels=LEVELS,mode='zero'):
    e=TrialEnv(controller=controller,condition=condition);rows=[]
    try:
        for level in levels:
            for seed in seeds:rows.append(episode(e,m,seed,level,False,mode))
    finally:e.close()
    return rows

if __name__=='__main__':collect_data()