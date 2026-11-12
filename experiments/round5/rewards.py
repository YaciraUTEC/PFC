"""Updated reward-learning adaptations on the identical80-input landing task.
CSIL uses the actual recurrent BC policy in train.py, including its binary gate.
ME-IRL retains transition-aware shaping, with common recurrent PPO instead ofSAC.
RIME uses scripted landing preferences, not human feedback; no paper-level claim.
"""
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from ..round3.rewards import Reward,MEIRL as PreviousMEIRL,mlp,tensor
from .evaluate import OUT
from ..run import save

class MEIRL(PreviousMEIRL):
    def __init__(self,data):
        Reward.__init__(self,data)
        self.r=mlp(80,1);self.phi=mlp(80,1)
        self.dynamics=nn.ModuleList([mlp(86,160) for _ in range(7)]);self.elites=list(range(5));self.to('cuda')
        self.opt=torch.optim.Adam(list(self.r.parameters())+list(self.phi.parameters()),lr=3e-4)
        self.dyn_opt=torch.optim.Adam(self.dynamics.parameters(),lr=5e-4)
    def shaped(self,s,a,done):
        return self.raw(s,a)+.999*(1-done)*self.expected_potential(s,a)-self.phi(self.norm(s)).squeeze(-1)
    def adversarial_update_mixed(self,data,learner,policy,seed,updates=8):
        rng=np.random.default_rng(seed);experts=np.flatnonzero(data['expert']);N=len(learner['s']);losses=[]
        if not len(experts):raise RuntimeError('No successful real expert transitions forME-IRL')
        for _ in range(updates):
            li=rng.integers(N,size=64)
            exact_experts=experts[np.max(np.abs(data['s'][experts,48:52]),axis=1)<1e-10]
            estimated_experts=experts[np.max(np.abs(data['s'][experts,48:52]),axis=1)>=1e-10]
            kinds=np.max(np.abs(learner['s'][li,48:52]),axis=1)<1e-10
            ei=np.asarray([rng.choice(exact_experts if k else estimated_experts) for k in kinds])
            s=tensor(np.r_[data['s'][ei],learner['s'][li]]);a=tensor(np.r_[data['executed'][ei],learner['a'][li]])
            done=tensor(np.r_[data['done'][ei],learner['done'][li]])
            # Expert density replays the available episode prefix, not a flat actor.
            lengths=ei-data['episode_start'][ei]+1
            histories=np.zeros((64,int(lengths.max()),80),np.float32)
            for j,idx in enumerate(ei):
                prefix=data['s'][int(data['episode_start'][idx]):idx+1];histories[j,:len(prefix)]=prefix
            with torch.no_grad():
                mu,_,_=policy(tensor(histories));expert_lp=policy.log_prob(mu[torch.arange(64,device='cuda'),torch.as_tensor(lengths-1,device='cuda')],a[:64])
                lp=torch.cat([expert_lp,tensor(learner['logp'][li])])
            logit=self.shaped(s,a,done)-lp
            labels=torch.cat([torch.ones(64,device='cuda'),torch.zeros(64,device='cuda')])
            loss=F.binary_cross_entropy_with_logits(logit,labels)+1e-3*(self.raw(s,a).square().mean()+self.phi(self.norm(s)).square().mean())
            self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(self.r.parameters())+list(self.phi.parameters()),10.);self.opt.step();losses.append(float(loss))
        return float(np.mean(losses))

class RIME(Reward):
    def __init__(self,data):
        super().__init__(data);self.models=nn.ModuleList([mlp(86,1,last=nn.Tanh()) for _ in range(3)]);self.to('cuda')
    def members(self,s,a):
        x=torch.cat([self.norm(s),a],-1);return torch.stack([m(x).squeeze(-1) for m in self.models])
    def raw(self,s,a,ns=None,done=None):return self.members(s,a).mean(0)
    def make_pairs(self,data,rng,split):
        groups={}
        for row in data['records']:
            if row['split']==split:groups.setdefault((row['seed'],row['level'],row['condition'],row['configuration']['intervention']),[]).append(row)
        pairs=[];labels=[]
        for records in groups.values():
            if len(records)<2:continue
            for _ in range(32):
                a,b=rng.choice(records,2,replace=False)
                if min(a['stop']-a['start'],b['stop']-b['start'])<96:continue
                p=np.stack([np.arange(a['stop']-96,a['stop']),np.arange(b['stop']-96,b['stop'])])
                quality=[100*float(r['stable_landing'])+float(data['reward'][r['start']:r['stop']].sum()) for r in [a,b]]
                label=.5 if abs(quality[0]-quality[1])<.02 else float(quality[0]>quality[1])
                pairs.append(p);labels.append(label)
        return np.asarray(pairs),np.asarray(labels,np.float32)
    def fit(self,data,seed,warm_updates=400,preference_updates=20,epochs=3):
        start=time.time();rng=np.random.default_rng(seed);ids=np.flatnonzero(data['train'])
        opt=torch.optim.Adam(self.parameters(),lr=3e-4);sample=rng.choice(ids,min(8000,len(ids)),replace=False)
        states=self.norm(tensor(data['s'][sample])).detach().cpu().numpy();dist=cKDTree(states).query(states,k=6,workers=1)[0][:,-1]
        intrinsic=np.log(np.maximum(dist,1e-5));intrinsic=np.clip((intrinsic-intrinsic.mean())/(3*max(intrinsic.std(),1e-4)),-.999,.999)
        for _ in range(warm_updates):
            j=rng.integers(len(sample),size=256);ix=sample[j];loss=(self.members(tensor(data['s'][ix]),tensor(data['executed'][ix]))-tensor(intrinsic[j])[None]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        pairs,clean=self.make_pairs(data,rng,'train')
        if not len(pairs):raise RuntimeError('No route/condition matched landing preferences')
        labels=clean.copy();noise=rng.random(len(labels))<.1;labels[noise]=1-labels[noise]
        rho=0.;var=1.;count=0;running_mean=0.;logs=[]
        for update in range(preference_updates):
            predictions=[]
            with torch.no_grad():
                for ix in np.array_split(np.arange(len(pairs)),max(1,len(pairs)//32)):
                    pp=pairs[ix];v=self.members(tensor(data['s'][pp]),tensor(data['executed'][pp])).sum(-1)
                    predictions.append(torch.sigmoid(v[:,:,0]-v[:,:,1]).mean(0).cpu().numpy())
            probability=np.clip(np.concatenate(predictions),1e-8,1-1e-8)
            kl=-labels*np.log(probability)-(1-labels)*np.log(1-probability)
            beta=max(1.,3.-2.*update/20);threshold=-np.log(rho+1e-8)+.5*rho+min(beta*var,3.)
            trusted=kl<threshold;flipped=(kl>-np.log(.001))&(labels!=.5);selected=np.flatnonzero(trusted|flipped)
            targets=labels.copy();targets[flipped]=1-targets[flipped]
            if trusted.any():
                values=kl[trusted];n=len(values);delta=values.mean()-running_mean;total=count+n
                var=(var*count+values.var()*n+delta**2*count*n/max(total,1))/max(total,1)
                running_mean+=.1*delta;rho+=.1*(values.max()-rho);count=total
            for epoch in range(epochs):
                rng.shuffle(selected)
                for ix in np.array_split(selected,max(1,len(selected)//32)):
                    if not len(ix):continue
                    pp=pairs[ix];v=self.members(tensor(data['s'][pp]),tensor(data['executed'][pp])).sum(-1)
                    logits=v[:,:,0]-v[:,:,1];target=tensor(targets[ix])[None].expand_as(logits)
                    loss=F.binary_cross_entropy_with_logits(logits,target)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.parameters(),10.);opt.step()
            logs.append(dict(update=update,trusted=int(trusted.sum()),flipped=int(flipped.sum())))
        vp,vl=self.make_pairs(data,np.random.default_rng(812), 'validation');accuracy=None
        if len(vp):
            predictions=[]
            with torch.no_grad():
                for pp in np.array_split(vp,max(1,len(vp)//32)):
                    v=self.members(tensor(data['s'][pp]),tensor(data['executed'][pp])).sum(-1)
                    predictions.extend(torch.sigmoid(v[:,:,0]-v[:,:,1]).mean(0).cpu().numpy())
            mask=vl!=.5;accuracy=float(np.mean((np.asarray(predictions)[mask]>.5)==(vl[mask]>.5))) if mask.any() else None
        return dict(seconds=time.time()-start,pairs=len(pairs),validation_pairs=len(vp),heldout_landing_preference_accuracy=accuracy,noise_fraction=float(noise.mean()),updates=logs,calibration=self.calibrate(dict(data,a=data['executed'])),
            provenance='Scripted full-flight landing outcome and return preferences on matched route,level,condition; last96steps;10percent synthetic flips;not human preferences.')
