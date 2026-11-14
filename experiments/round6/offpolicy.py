"""FlashSAC and FQL heads on a frozen, recurrent Mamba-3/GTrXL representation.
Freezing the memory avoids stale hidden states in off-policy replay. Raw state
is concatenated to memory. Both heads retain a learned Q-difference OFF option.
FlashSAC critic/actor blocks are official MIT source. FQL losses are a PyTorch
port of seohongpark/fql/agents/fql.py, not a rerun of the authors' benchmarks.
"""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .models import Agent,physics_observation
from .vendor_flash.network import FlashSACActor,FlashSACDoubleCritic

ACTION_DIM=4
class MLP(nn.Module):
    def __init__(self,dim,out,width=512,layers=4,layer_norm=True):
        super().__init__();items=[];current=dim
        for _ in range(layers):
            linear=nn.Linear(current,width);nn.init.xavier_uniform_(linear.weight);nn.init.zeros_(linear.bias)
            items.extend([linear,nn.GELU()])
            if layer_norm:items.append(nn.LayerNorm(width,eps=1e-6))
            current=width
        linear=nn.Linear(current,out);nn.init.xavier_uniform_(linear.weight);nn.init.zeros_(linear.bias)
        items.append(linear);self.layers=nn.Sequential(*items)
    def forward(self,x):return self.layers(x)
def normalize(m):
    with torch.no_grad():
        for module in m.modules():
            if hasattr(module,'normalize_parameters'):module.normalize_parameters()
def project_distribution(log_probs,reward,done,entropy,gamma=.999,minimum=-5.,maximum=5.):
    n=log_probs.shape[-1];support=torch.linspace(minimum,maximum,n,device=log_probs.device,dtype=log_probs.dtype)
    target=(reward[:,None]+gamma*(support[None]-entropy[:,None])*(1-done[:,None])).clamp(minimum,maximum)
    b=(target-minimum)/((maximum-minimum)/(n-1));lo=b.floor().long();hi=(lo+1).clamp_max(n-1);frac=b-lo
    out=torch.zeros_like(log_probs);out.scatter_add_(1,lo,log_probs.exp()*(1-frac));out.scatter_add_(1,hi,log_probs.exp()*frac)
    return out
class OffPolicy(nn.Module):
    def __init__(self,prior,method='flashsac',gate=True):
        super().__init__();self.prior=prior;self.method=method;self.gate=gate;self.threshold=.005;self.evaluation_seed=0
        for p in prior.parameters():p.requires_grad_(False)
        self.dim=prior.dim+prior.cfg['width']
        if method=='flashsac':
            self.actor=FlashSACActor(2,self.dim,128,ACTION_DIM)
            self.critic=FlashSACDoubleCritic(2,self.dim+ACTION_DIM,256,101,-5.,5.)
            normalize(self.actor);normalize(self.critic)
            self.log_alpha=nn.Parameter(torch.tensor(math.log(.01)))
        elif method=='fql':
            self.actor=MLP(self.dim+ACTION_DIM,ACTION_DIM,layer_norm=False)
            self.flow=MLP(self.dim+ACTION_DIM+1,ACTION_DIM,layer_norm=False)
            self.critic=nn.ModuleList([MLP(self.dim+ACTION_DIM,1),MLP(self.dim+ACTION_DIM,1)])
        else:raise ValueError(method)
        self.benefit=nn.ModuleList([MLP(self.dim+ACTION_DIM,1,width=128,layers=2),MLP(self.dim+ACTION_DIM,1,width=128,layers=2)])
        self.benefit_target=copy.deepcopy(self.benefit)
        for p in self.benefit_target.parameters():p.requires_grad_(False)
        self.target=copy.deepcopy(self.critic)
        for p in self.target.parameters():p.requires_grad_(False)
    def features(self,obs,state=None):
        obs=physics_observation(obs)
        with torch.no_grad():z,new=self.prior.encoder(obs[:,None],state)
        return torch.cat([obs,z[:,0]],-1),new
    def q(self,features,actions,target=False,training=False):
        c=self.target if target else self.critic
        if self.method=='flashsac':return c(features,actions,training=training)[0]
        return torch.stack([m(torch.cat([features,actions],-1)).squeeze(-1) for m in c],0)
    def proposal(self,features,deterministic=False,noise=None):
        if self.method=='flashsac':
            mean,std=self.actor.get_mean_and_std(features,training=False)
            raw=mean if deterministic else mean+std*(torch.randn_like(mean) if noise is None else noise)
            a=raw.tanh();lp=(torch.distributions.Normal(mean,std).log_prob(raw)-2*(math.log(2)-raw-F.softplus(-2*raw))).sum(-1)
            return a,lp
        noise=torch.randn((*features.shape[:-1],ACTION_DIM),device=features.device,dtype=features.dtype) if noise is None else noise
        return self.actor(torch.cat([features,noise],-1)).clamp(-1,1),torch.zeros(features.shape[0],device=features.device)
    def benefit_q(self,features,actions,target=False):
        critics=self.benefit_target if target else self.benefit
        return torch.stack([m(torch.cat([features,actions],-1)).squeeze(-1) for m in critics],0)
    def select(self,features,actions,target=False):
        if not self.gate:return actions
        with torch.no_grad():
            candidate=self.benefit_q(features,actions,target=target);off=self.benefit_q(features,torch.zeros_like(actions),target=target)
            on=(candidate-off).amin(0)>self.threshold
        return torch.where(on[:,None],actions,torch.zeros_like(actions))
    def evaluation_noise(self,clock):
        # A reproducible Gaussian stream, common across routes and reset at
        # every episode. Integer hashing is independent of evaluation batching.
        h=(clock.long()[:,None]*1664525+torch.arange(4,device=clock.device)[None]*1013904223+self.evaluation_seed*69069+9973)&0xffffffff
        def mix(v):
            v=((v^(v>>16))*0x45d9f3b)&0xffffffff
            v=((v^(v>>16))*0x45d9f3b)&0xffffffff
            return v^(v>>16)
        u=(mix(h).double()+.5)/4294967296.;v=(mix(h+817313).double()+.5)/4294967296.
        return (torch.sqrt(-2*torch.log(u))*torch.cos(2*math.pi*v)).to(next(self.parameters()).dtype)
    def predict_actions(self,obs,state=None):
        memory=None if state is None else state[0]
        clock=torch.zeros(len(obs),device=obs.device) if state is None else state[1]
        features,new=self.features(obs,memory)
        noise=self.evaluation_noise(clock) if self.method=='fql' else None
        proposal,_=self.proposal(features,deterministic=True,noise=noise)
        action=self.select(features,proposal)
        amount=action.abs().amax(-1).clamp(0,1);on=amount>.002
        direction=action/amount.clamp_min(1e-9)[:,None]
        a=torch.cat([direction,(2*amount-1)[:,None],torch.where(on,1.,-1.)[:,None]],-1)
        return a,[new,clock+1]
    def flow_actions(self,x,noise):
        a=noise
        with torch.no_grad():
            for i in range(10):
                t=torch.full((*x.shape[:-1],1),i/10,device=x.device,dtype=x.dtype)
                a=a+self.flow(torch.cat([x,a,t],-1))/10
        return a.clamp(-1,1)
    def save(self,path,**metadata):
        torch.save(dict(method=self.method,gate=self.gate,threshold=self.threshold,evaluation_seed=self.evaluation_seed,prior_config=self.prior.cfg,prior_dim=self.prior.dim,state=self.state_dict(),metadata=metadata),path)
    @classmethod
    def load(cls,path,device='cpu'):
        d=torch.load(path,map_location=device,weights_only=False);m=cls(Agent(d['prior_config'],d['prior_dim']),d['method'],d['gate']).to(device)
        m.load_state_dict(d['state']);m.threshold=d['threshold'];m.evaluation_seed=d.get('evaluation_seed',0);return m.eval()

def compress_actions(a):
    a=np.asarray(a);amount=(a[...,4:5]+1)/2;active=a[...,5:6]>0
    return np.where(active,a[...,:4]*amount,0.).astype(np.float32)
