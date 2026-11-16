"""GTrXL proposal-only repair: a unique physical delta and observable eligibility.
No PPO/SAC training interface is implied for the direct supervised head.
"""
import torch
from ..round6.models import Agent as Previous,physics_observation

class ProposalAgent(Previous):
    def physical_delta(self,mu):
        if self.cfg.get('direct_motor_head'):return mu[...,:4].tanh()
        a=mu[...,:5].clamp(-5,5).tanh();return a[...,:4]*(a[...,4:5]+1)/2
    def forward(self,obs,state=None):
        observed=self.cfg.get('proposal_gate')=='observed_fault'
        if not observed:return super().forward(obs,state)
        memory=None if state is None else state[0];previous=None if state is None else state[1]
        clean=physics_observation(obs);hidden,new=self.encoder(clean,memory);mu=self.actor(hidden)
        seen=(clean[...,31:35]>.08).any(-1).to(mu.dtype)
        cumulative=seen.cumsum(1)>0
        if previous is not None:cumulative |= previous[:,None].bool()
        gate=torch.where(cumulative,torch.full_like(seen,10.),torch.full_like(seen,-10.))
        continuous=mu[...,:5] if self.cfg.get('direct_motor_head') else mu[...,:5].clamp(-5,5)
        out=torch.cat([continuous,gate[...,None]],-1)
        value=self.critic(torch.cat([clean,hidden.detach()],-1)).squeeze(-1)
        return out,value,(new,cumulative[:,-1])
    def propose_actions(self,obs,state=None):
        from .policy import expand_delta
        mu,_,new=self(obs[:,None],state);mu=mu[:,0];delta=self.physical_delta(mu);on=mu[:,5]>0
        return expand_delta(torch.where(on[:,None],delta,torch.zeros_like(delta))),expand_delta(delta),new
    def predict_actions(self,obs,state=None):
        action,_,new=self.propose_actions(obs,state);return action,new
    def act(self,*args,**kwargs):raise RuntimeError('ProposalAgent is supervised; use the explicit composed authorizer for control.')
