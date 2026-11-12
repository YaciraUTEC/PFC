"""Mixed continuous correction and explicit binary abstention.
The value network sees detached policy memory features: history without value
loss gradients changing the actor's encoder. This is not a second shared loss.
"""
import torch
from torch import nn
from torch.nn import functional as F
from ..round4.models import Agent as PreviousAgent,ARCHS,configurations,tree_map,detach,select,concatenate

class Agent(PreviousAgent):
    def __init__(self,cfg,dim=80):
        super().__init__(cfg,dim);w=cfg['width']
        self.actor=nn.Sequential(nn.Linear(w,64),nn.Tanh(),nn.Linear(64,6))
        self.critic=nn.Sequential(nn.LayerNorm(dim+w),nn.Linear(dim+w,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.full((5,),-2.5))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():self.actor[-1].bias[4]=-2.;self.actor[-1].bias[5]=-3.
    def _apply(self,fn,recurse=True):
        # Positional/mask caches are not registered buffers in the vendorGTrXL.
        # A copied GPU policy must rebuild them after a device or dtype change.
        for module in self.modules():
            if hasattr(module,'pos_embedding_dict'):module.pos_embedding_dict.clear()
            if hasattr(module,'att_mask'):module.att_mask.clear()
        return super()._apply(fn,recurse=recurse)
    def forward(self,obs,state=None):
        z,st=self.encoder(obs,state)
        mu=self.actor(z);mu=torch.cat([mu[...,:5].clamp(-5,5),mu[...,5:]],-1)
        return mu,self.critic(torch.cat([obs,z.detach()],-1)).squeeze(-1),st
    def distribution(self,mu):
        return torch.distributions.Normal(mu[...,:5],self.log_std.exp()),torch.distributions.Bernoulli(logits=mu[...,5])
    def log_prob(self,mu,actions):
        continuous,binary=self.distribution(mu);a=actions[...,:5]
        raw=torch.atanh(a.clamp(-.999999,.999999))
        on=(actions[...,5]+1)/2
        return on*(continuous.log_prob(raw)-torch.log(1-a.square()+1e-6)).sum(-1)+binary.log_prob(on)
    def act(self,obs,state=None,deterministic=False,generator=None):
        mu,v,st=self(obs[:,None],state);mu=mu[:,0];v=v[:,0]
        continuous,binary=self.distribution(mu)
        raw=mu[:,:5] if deterministic else mu[:,:5]+self.log_std.exp()*torch.randn(mu[:,:5].shape,device=mu.device,dtype=mu.dtype,generator=generator)
        on=(mu[:,5]>0).to(mu.dtype) if deterministic else torch.bernoulli(binary.probs,generator=generator)
        continuous_action=raw.tanh();off=torch.zeros_like(continuous_action);off[:,4]=-1.
        continuous_action=torch.where(on[:,None]>0,continuous_action,off)
        a=torch.cat([continuous_action,(2*on-1)[:,None]],-1)
        return a,self.log_prob(mu,a),v,st
    def propose_actions(self,obs,state=None):
        z,st=self.encoder(obs[:,None],state);mu=self.actor(z[:,0])
        on=mu[:,5:]>getattr(self,'gate_logit_threshold',0.);off=torch.zeros_like(mu[:,:5]);off[:,4]=-1.
        continuous=mu[:,:5].clamp(-5,5).tanh()
        action=torch.cat([torch.where(on,continuous,off),torch.where(on,1.,-1.)],-1)
        proposal=torch.cat([continuous,torch.ones_like(mu[:,5:])],-1)
        return action,proposal,st
    def predict_actions(self,obs,state=None):
        action,_,st=self.propose_actions(obs,state);return action,st
    def prior_log_prob(self,actions):
        a=actions[...,:5];raw=torch.atanh(a.clamp(-.999999,.999999))
        normal=torch.distributions.Normal(torch.zeros_like(raw),torch.ones_like(raw))
        binary=torch.distributions.Bernoulli(probs=torch.full_like(actions[...,5],.1))
        on=(actions[...,5]+1)/2
        return on*(normal.log_prob(raw)-torch.log(1-a.square()+1e-6)).sum(-1)+binary.log_prob(on)

def clone_loss(mu,target,mask,gate_valid=None):
    a=mu[...,:5].tanh();amount=(a[...,4]+1)/2;ta=(target[...,4]+1)/2
    tg=(target[...,5]+1)/2
    # Magnitude and direction are physical regression; gate means paired benefit.
    effective=(amount[...,None]*a[...,:4]-ta[...,None]*target[...,:4]).square().mean(-1)
    direction=(a[...,:4]-target[...,:4]).square().mean(-1)*ta
    regression=(3*effective+.1*(amount-ta).square()+.05*direction)*mask
    valid=mask if gate_valid is None else mask*gate_valid
    gate=(F.binary_cross_entropy_with_logits(mu[...,5],tg,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
    return regression.sum()/mask.sum().clamp_min(1)+.3*gate

def load_agent(path,device='cpu'):
    return Agent.load(path,device=device).double().eval()
