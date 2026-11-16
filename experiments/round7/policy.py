"""GTrXL proposal, canonical GTrXL risk memory and explicit bounded options.
All decision inputs are deployable. The risk encoder is common across proposers
so that changing a proposal seed does not rotate the gate's representation.
"""
import torch,re
from torch import nn
import numpy as np
from ..round6.models import Agent,physics_observation
from ..round6.offpolicy import OffPolicy
from .common import old_model,read,save
from .episode import load_policy

OPTIONS=[(1,1.),(24,.5),(96,1.),(1440,1.)]

def option_durations(obs):
    available=(obs[:,72]*1440).round().clamp(1,1440)
    return torch.stack([torch.minimum(torch.full_like(available,float(t)),available) for t,a in OPTIONS],1)

def controller_descriptor(path):
    """Known recovery-controller identity, never the physical route/fault seed."""
    text=str(path).replace('\\','/').lower();match=re.search(r'seed_(\d+)',text)
    seed=int(match.group(1)) if match else 301;index=seed-(201 if seed<300 else 301)
    if not 0<=index<5:raise ValueError('Unexpected controller seed '+str(seed))
    out=np.zeros(7,np.float64);out[0]=float('/flashsac/' in text);out[1+index]=1.;out[6]=float('/refinement/' in text)
    return out

def propose(model,obs,state=None):
    if isinstance(model,OffPolicy):
        memory=None if state is None else state[0]
        clock=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device) if state is None else state[1]
        f,new=model.features(obs,memory);noise=model.evaluation_noise(clock) if model.method=='fql' else None
        delta,_=model.proposal(f,deterministic=True,noise=noise)
        native=(model.select(f,delta).abs().amax(-1)>.002)
        return delta,native,[new,clock+1]
    mu,_,new=model(obs[:,None],state);mu=mu[:,0];a=mu[:,:5].clamp(-5,5).tanh()
    delta=model.physical_delta(mu) if hasattr(model,'physical_delta') else a[:,:4]*(a[:,4:5]+1)/2
    return delta,mu[:,5]>0,new

def canonical_risk():
    return OffPolicy.load(old_model(201),'cpu').prior

def risk_features(risk,obs,state=None):
    clean=physics_observation(obs);z,new=risk.encoder(clean[:,None],state)
    return torch.cat([clean,z[:,0]],-1),new

def expand_delta(delta):
    amount=delta.abs().amax(-1).clamp(0,1);on=amount>.002
    direction=torch.where(on[:,None],delta/amount.clamp_min(1e-9)[:,None],torch.zeros_like(delta))
    magnitude=torch.where(on,2*amount-1,-torch.ones_like(amount))
    return torch.cat([direction,magnitude[:,None],torch.where(on,1.,-1.)[:,None]],-1)

class OutcomeGate(nn.Module):
    def __init__(self,dim=196,members=3):
        super().__init__();self.dim=dim
        self.register_buffer('center',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim));self.register_buffer('temperature',torch.ones(()))
        self.members=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.LayerNorm(128),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4)) for _ in range(members)])
    def forward(self,x):
        z=((x-self.center)/self.scale).clamp(-12,12)
        return torch.stack([m(z)/self.temperature for m in self.members],0)
    def probabilities(self,x):return self(x).softmax(-1)
    def save(self,path,**meta):torch.save(dict(dim=self.dim,members=len(self.members),state=self.state_dict(),metadata=meta),path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=False);m=cls(d['dim'],d['members']);m.load_state_dict(d['state']);return m.eval()

class TransformerPolicy(nn.Module):
    def __init__(self,proposer,gate=None,*,mode='native',minimum_gain=.05,maximum_harm=.02,uncertainty_penalty=1.,screen=None,continuation_gain=-.005):
        super().__init__();self.proposer=proposer;self.risk=canonical_risk();self.gate=gate
        self.register_buffer('controller_id',torch.as_tensor(controller_descriptor(getattr(proposer,'checkpoint_path','')),dtype=torch.float64))
        self.continuation_gain=float(continuation_gain);self.mode=mode;self.minimum_gain=minimum_gain;self.maximum_harm=maximum_harm;self.uncertainty_penalty=uncertainty_penalty;self.screen=screen
        for p in self.parameters():p.requires_grad_(False)
    def predict_actions(self,obs,state=None):
        if state is None:pm=rm=None;option_scale=torch.zeros(len(obs),dtype=obs.dtype,device=obs.device);remaining=torch.zeros_like(option_scale)
        else:pm,rm,option_scale,remaining=state
        raw,native,pm=propose(self.proposer,obs,pm)
        # Preserve the useful native veto; the new authorizer certifies whether
        # to run this feedback recovery policy rather than forcing raw outputs.
        delta=torch.where(native[:,None],raw,torch.zeros_like(raw))
        if self.mode=='counterfactual':feat,rm=risk_features(self.risk,obs,rm)
        else:feat=None
        diagnostics=dict(proposal_delta=raw.detach(),native_delta=delta.detach(),native_on=native.detach(),previous_remaining=remaining.detach())
        duration=torch.ones_like(remaining);scale=torch.ones_like(remaining)
        if self.mode=='native':selected=delta;plan_on=selected.abs().amax(-1)>.002
        elif self.mode=='forced':selected=raw;plan_on=selected.abs().amax(-1)>.002
        elif self.mode=='off':selected=torch.zeros_like(delta);plan_on=torch.zeros_like(native)
        elif self.mode=='counterfactual':
            aa=torch.stack([delta*alpha for ticks,alpha in OPTIONS],1)
            budgets=option_durations(obs);dd=budgets[:,:,None]/96
            identity=self.controller_id[None,None].expand(len(obs),len(OPTIONS),-1)
            x=torch.cat([feat[:,None].expand(-1,len(OPTIONS),-1),aa,identity,dd],-1)
            probs=self.gate.probabilities(x);adv=probs[...,1]-probs[...,2]
            lower=adv.mean(0)-self.uncertainty_penalty*adv.std(0,unbiased=False)
            harm=probs[...,2].mean(0)+self.uncertainty_penalty*probs[...,2].std(0,unbiased=False)
            permitted=(lower>self.minimum_gain)&(harm<self.maximum_harm)&(aa.abs().amax(-1)>.002)
            scores=torch.where(permitted,lower,torch.full_like(lower,-1e6));idx=scores.argmax(-1);on=permitted.any(-1)
            ticks=budgets
            scales=torch.tensor([a for t,a in OPTIONS],dtype=obs.dtype,device=obs.device)
            duration=ticks[torch.arange(len(obs),device=obs.device),idx];scale=scales[idx]
            continuing_delta=delta*option_scale[:,None]
            held_x=torch.cat([feat,continuing_delta,self.controller_id[None].expand(len(obs),-1),remaining.clamp(1,1440)[:,None]/96],-1)
            hp=self.gate.probabilities(held_x);ha=hp[...,1]-hp[...,2]
            held_gain=ha.mean(0)-self.uncertainty_penalty*ha.std(0,unbiased=False)
            held_harm=hp[...,2].mean(0)+self.uncertainty_penalty*hp[...,2].std(0,unbiased=False)
            continuing=(remaining>0)&(held_gain>=self.continuation_gain)&(held_harm<self.maximum_harm)
            interrupted=(remaining>0)&~continuing
            duration=torch.where(continuing,remaining,duration);scale=torch.where(continuing,option_scale,scale)
            plan_on=(on|continuing)&~interrupted
            selected=torch.where(plan_on[:,None],delta*scale[:,None],torch.zeros_like(delta))
            pmean=probs.mean(0)[torch.arange(len(obs),device=obs.device),idx]
            pmean=torch.where(continuing[:,None],hp.mean(0),pmean)
            diagnostics.update(predicted_rescue=pmean[:,1].detach(),predicted_harm=pmean[:,2].detach(),predicted_base_success=(pmean[:,0]+pmean[:,2]).detach(),interrupted=interrupted.detach())
        else:raise ValueError(self.mode)
        diagnostics['authorized_delta']=selected.detach()
        if self.screen is not None:selected=self.screen(obs,selected,duration)
        veto=(diagnostics['authorized_delta'].abs().amax(-1)>.002)&(selected.abs().amax(-1)<=.002)
        plan_on=plan_on&~veto&(obs[:,57]<.5)
        selected=torch.where((obs[:,57]<.5)[:,None],selected,torch.zeros_like(selected))
        remaining=torch.where(plan_on,(duration-1).clamp_min(0),torch.zeros_like(duration))
        option_scale=torch.where(plan_on,scale,torch.zeros_like(scale))
        selected=torch.where((selected.abs().amax(-1)>.002)[:,None],selected,torch.zeros_like(selected))
        diagnostics.update(final_delta=selected.detach(),remaining=remaining.detach(),screen_veto=veto.detach())
        self.last_diagnostics=diagnostics
        return expand_delta(selected),[pm,rm,option_scale,remaining]
    @classmethod
    def from_spec(cls,spec):
        from .screen import LandingScreen
        proposer=load_policy(spec['proposer'])
        if spec.get('native_threshold') is not None and hasattr(proposer,'threshold'):proposer.threshold=spec['native_threshold']
        result=cls(proposer,OutcomeGate.load(spec['gate']) if spec.get('gate') else None,
            mode=spec.get('mode','native'),minimum_gain=spec.get('minimum_gain',.05),maximum_harm=spec.get('maximum_harm',.02),uncertainty_penalty=spec.get('uncertainty_penalty',1.),
            screen=LandingScreen(**spec['screen']) if spec.get('screen') else None,continuation_gain=spec.get('continuation_gain',-.005)).double().eval()
        if spec.get('controller_version_override') is not None:result.controller_id[6]=float(spec['controller_version_override'])
        return result
