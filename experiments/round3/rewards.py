"""Equation-level Torch ports for a controlled reward-learning screen.
CSIL stationary normal head: google-deepmind/csil (Apache-2.0).
RIME selection: CJReinforce/RIME_ICML2024; ME-IRL transition-aware shaping
Zhan et al., L4DC 2026. PPO optimizer / finite task are explicit adaptations.
"""
import math,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from .common import OUT,save

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda')
def mlp(i,o,hidden=128,last=None):
    layers=[nn.Linear(i,hidden),nn.ELU(),nn.Linear(hidden,hidden),nn.ELU(),nn.Linear(hidden,o)]
    if last:layers.append(last)
    return nn.Sequential(*layers)

class Reward(nn.Module):
    def __init__(self,data):
        super().__init__()
        s=data['s'][data['train']]
        self.register_buffer('mean',tensor(s.mean(0)))
        self.register_buffer('std',tensor(np.maximum(s.std(0),.05)))
        self.register_buffer('center',tensor(0.));self.register_buffer('scale',tensor(1.))
    def norm(self,s):return ((s-self.mean)/self.std).clamp(-10,10)
    def forward(self,s,a,ns=None,done=None):return ((self.raw(s,a,ns,done)-self.center)/self.scale).clamp(-5,5)
    @torch.no_grad()
    def calibrate(self,data):
        ix=np.flatnonzero(data['train'])[::3];v=[]
        for ids in np.array_split(ix,max(1,len(ix)//1024)):
            v.append(self.raw(tensor(data['s'][ids]),tensor(data['a'][ids]),tensor(data['ns'][ids]),tensor(data['done'][ids])).cpu().numpy())
        vals=np.concatenate(v);self.center.copy_(tensor(vals.mean()));self.scale.copy_(tensor(max(vals.std(),.1)))
        return dict(mean=float(self.center),std=float(self.scale),q=np.quantile(vals,[0,.1,.5,.9,1]).tolist())

class StationaryNormal(nn.Module):
    """Periodic features and PSD heteroscedastic head; smaller official CSIL head."""
    def __init__(self,dim=128):
        super().__init__();self.dim=dim
        self.body=nn.Sequential(nn.Linear(48,128),nn.ELU(),nn.Linear(128,32),nn.ELU())
        self.random_weights=nn.Parameter(torch.randn(32,dim//2))
        self.log_lengthscales=nn.Parameter(torch.full((32,),-5.))
        self.loc_weights=nn.Parameter(torch.zeros(dim,5))
        self.tril=nn.Parameter(torch.zeros(5,dim,dim))
    def forward(self,s):
        h=self.body(s)*self.log_lengthscales.exp()
        wx=h@self.random_weights
        f=torch.cat([torch.sin(wx+math.pi/4)+torch.cos(wx+math.pi/4),torch.sin(wx-math.pi/4)+torch.cos(wx-math.pi/4)],-1)/math.sqrt(self.dim)
        L=torch.tril(self.tril,-1)+torch.diag_embed(self.tril.diagonal(dim1=-2,dim2=-1).exp())
        mu=f@self.loc_weights
        v=torch.einsum('bi,dij->bdj',f,L).square().sum(-1)
        std=torch.tanh(torch.sqrt(1e-4+v)).clamp_min(.02)
        return mu,std

class CSIL(Reward):
    def __init__(self,data):super().__init__(data);self.policy=StationaryNormal();self.to('cuda')
    def raw(self,s,a,ns=None,done=None):
        mu,std=self.policy(self.norm(s));z=torch.atanh(a.clamp(-.999,.999))
        logq=torch.distributions.Normal(mu,std).log_prob(z).sum(-1)
        logp=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(std)).log_prob(z).sum(-1)
        # Same tanh transform cancels its Jacobian in log(q / p).
        return .1*(logq-logp)
    def fit(self,data,seed,updates=3500):
        rng=np.random.default_rng(seed);ids=np.flatnonzero(data['expert']);active=ids[np.any(data['s'][ids,31:35]>.001,axis=1)]
        opt=torch.optim.Adam(self.parameters(),lr=5e-4);log=[];start=time.time()
        for it in range(updates):
            ix=np.r_[rng.choice(ids,128),rng.choice(active,128)];s=tensor(data['s'][ix]);a=tensor(data['a'][ix]).clamp(-.999,.999)
            mu,std=self.policy(self.norm(s));z=torch.atanh(a)
            loss=-torch.distributions.Normal(mu,std).log_prob(z).sum(-1).mean()
            loss+=1e-5*(self.policy.loc_weights.square().sum()+self.policy.tril.square().sum())
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.parameters(),10);opt.step()
            if it%500==0:log.append(float(loss));print('CSIL',it,float(loss),flush=True)
        return dict(seconds=time.time()-start,loss=log,calibration=self.calibrate(data))

class MEIRL(Reward):
    def __init__(self,data):
        super().__init__(data);self.r=mlp(48,1);self.phi=mlp(48,1)
        self.dynamics=nn.ModuleList([mlp(53,96) for _ in range(7)])
        self.elites=list(range(5));self.to('cuda')
        self.opt=torch.optim.Adam(list(self.r.parameters())+list(self.phi.parameters()),lr=3e-4)
        self.dyn_opt=torch.optim.Adam(self.dynamics.parameters(),lr=5e-4)
    def raw(self,s,a,ns=None,done=None):return self.r(self.norm(s)).squeeze(-1)
    def expected_potential(self,s,a):
        x=torch.cat([self.norm(s),a],-1);pred=[]
        for j in self.elites:
            with torch.no_grad():
                mean,lv=self.dynamics[j](x).chunk(2,-1);lv=lv.clamp(-6,1)
                nxt=self.norm(s)+mean+torch.randn_like(mean)*(.5*lv).exp()
            pred.append(self.phi(nxt.clamp(-10,10)).squeeze(-1))
        return torch.stack(pred).mean(0)
    def shaped(self,s,a,done):return self.raw(s,a)+.995*(1-done)*self.expected_potential(s,a)-self.phi(self.norm(s)).squeeze(-1)
    def fit_dynamics(self,data,seed,updates=600):
        rng=np.random.default_rng(seed);train=np.flatnonzero(data['train'] & ~data['done']);val=np.flatnonzero(data['validation'] & ~data['done']);start=time.time()
        for it in range(updates):
            loss=0.
            for model in self.dynamics:
                ix=rng.choice(train,128);s=tensor(data['s'][ix]);a=tensor(data['a'][ix]);ns=tensor(data['ns'][ix])
                mean,lv=model(torch.cat([self.norm(s),a],-1)).chunk(2,-1);lv=lv.clamp(-6,1)
                target=self.norm(ns)-self.norm(s)
                loss+=((mean-target).square()*(-lv).exp()+lv).mean()
            self.dyn_opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.dynamics.parameters(),10);self.dyn_opt.step()
        with torch.no_grad():
            ix=rng.choice(val,min(2048,len(val)),False);s=tensor(data['s'][ix]);a=tensor(data['a'][ix]);ns=tensor(data['ns'][ix]);target=self.norm(ns)-self.norm(s)
            losses=[float((m(torch.cat([self.norm(s),a],-1)).chunk(2,-1)[0]-target).square().mean()) for m in self.dynamics]
        self.elites=np.argsort(losses)[:5].tolist()
        return dict(seconds=time.time()-start,validation_mse=losses,elites=self.elites)
    def adversarial_update(self,data,learner,policy,seed,updates=80):
        rng=np.random.default_rng(seed);ex=np.flatnonzero(data['expert']);N=len(learner['s']);losses=[]
        for it in range(updates):
            ei=rng.choice(ex,128);li=rng.integers(N,size=128)
            s=tensor(np.r_[data['s'][ei],learner['s'][li]]);a=tensor(np.r_[data['a'][ei],learner['a'][li]]).clamp(-.999,.999)
            done=tensor(np.r_[data['done'][ei],learner['done'][li]])
            with torch.no_grad():lp=policy.get_distribution(s).log_prob(a)
            f=self.shaped(s,a,done);logit=f-lp
            target=torch.cat([torch.ones(128,device='cuda'),torch.zeros(128,device='cuda')])
            loss=F.binary_cross_entropy_with_logits(logit,target)+1e-3*(self.raw(s,a).square().mean()+self.phi(self.norm(s)).square().mean())
            self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(self.r.parameters())+list(self.phi.parameters()),10);self.opt.step()
            losses.append(float(loss))
        return float(np.mean(losses))

class RIME(Reward):
    def __init__(self,data):
        super().__init__(data);self.models=nn.ModuleList([mlp(53,1,last=nn.Tanh()) for _ in range(3)]);self.to('cuda')
    def members(self,s,a):
        x=torch.cat([self.norm(s),a],-1);return torch.stack([m(x).squeeze(-1) for m in self.models])
    def raw(self,s,a,ns=None,done=None):return self.members(s,a).mean(0)
    def fit(self,data,seed):
        rng=np.random.default_rng(seed);start=time.time();ids=np.flatnonzero(data['train']);opt=torch.optim.Adam(self.parameters(),lr=3e-4)
        # Published entropy warm start, fitted on training states only.
        sample=rng.choice(ids,min(20000,len(ids)),False);states=self.norm(tensor(data['s'][sample])).detach().cpu().numpy()
        dist=cKDTree(states).query(states,k=6,workers=1)[0][:,-1]
        intrinsic=np.log(np.maximum(dist,1e-5));intrinsic=np.clip((intrinsic-intrinsic.mean())/(3*max(intrinsic.std(),1e-4)),-.999,.999)
        for it in range(400):
            j=rng.integers(len(sample),size=256);ix=sample[j];v=self.members(tensor(data['s'][ix]),tensor(data['a'][ix]));loss=(v-tensor(intrinsic[j])[None,:]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        # Pairs share route, failure level and segment start; no difficulty shortcut.
        groups={}
        for r in data['records']:
            if r['split']=='reward_train':groups.setdefault((r['seed'],r['level']),[]).append(r)
        pairs=[];labels=[]
        for records in groups.values():
            for _ in range(28):
                r1,r2=rng.choice(records,2,replace=False);length=min(r1['stop']-r1['start'],r2['stop']-r2['start'])
                if length<32:continue
                start_index=int(rng.integers(length-31));ix1=np.arange(r1['start']+start_index,r1['start']+start_index+32);ix2=np.arange(r2['start']+start_index,r2['start']+start_index+32)
                quality=[float(np.sum(data['task'][ix]+.02*data['style'][ix])) for ix in [ix1,ix2]]
                label=.5 if abs(quality[0]-quality[1])<.005 else float(quality[0]>quality[1])
                pairs.append(np.stack([ix1,ix2]));labels.append(label)
        pairs=np.asarray(pairs);clean=np.asarray(labels,np.float32);labels=clean.copy()
        noise=rng.random(len(labels))<.10;labels[noise]=1-labels[noise]
        np.savez_compressed(OUT/f'rime_pairs_{seed}.npz',indices=pairs,labels=labels,clean_labels=clean,noise=noise)
        rho=0.;var=1.;count=0;running_mean=0.;logs=[]
        for update in range(20):
            probabilities=[]
            with torch.no_grad():
                for ix in np.array_split(np.arange(len(pairs)),max(1,len(pairs)//128)):
                    pp=pairs[ix];s=tensor(data['s'][pp]);a=tensor(data['a'][pp]);sums=self.members(s,a).sum(-1)
                    probabilities.append(torch.sigmoid(sums[:, :,0]-sums[:,:,1]).mean(0).cpu().numpy())
            p=np.clip(np.concatenate(probabilities),1e-8,1-1e-8)
            kl=-labels*np.log(p)-(1-labels)*np.log(1-p)
            beta=max(1.,3.-2.*update/20);threshold=-np.log(rho+1e-8)+.5*rho+min(beta*var,3.)
            trusted=kl<threshold;flipped=(kl>-np.log(.001)) & (labels!=.5)
            selected=np.flatnonzero(trusted|flipped);targets=labels.copy();targets[flipped]=1-targets[flipped]
            if trusted.any():
                vals=kl[trusted];batch_count=len(vals);delta=vals.mean()-running_mean;total=count+batch_count
                var=(var*count+vals.var()*batch_count+delta**2*count*batch_count/max(total,1))/max(total,1)
                running_mean+=.1*delta;rho+=.1*(vals.max()-rho);count=total
            for epoch in range(3):
                rng.shuffle(selected)
                for ix in np.array_split(selected,max(1,len(selected)//128)):
                    if not len(ix):continue
                    pp=pairs[ix];v=self.members(tensor(data['s'][pp]),tensor(data['a'][pp])).sum(-1)
                    logits=v[:,:,0]-v[:,:,1];target=tensor(targets[ix])[None,:].expand_as(logits)
                    loss=F.binary_cross_entropy_with_logits(logits,target)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.parameters(),10);opt.step()
            logs.append(dict(update=update,trusted=int(trusted.sum()),flipped=int(flipped.sum()),threshold=float(threshold),clean_accuracy=float(np.mean((p>.5)==(clean>.5)))))
            if update%5==0:print('RIME',logs[-1],flush=True)
        return dict(seconds=time.time()-start,pairs=len(pairs),noise_fraction=float(noise.mean()),updates=logs,calibration=self.calibrate(data))