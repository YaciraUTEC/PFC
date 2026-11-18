"""Reproduce the known handoff and verify complete new-option semantics."""
import math,time
import numpy as np
import torch
from .common import OUT,R7,read,save,close
from .factor_gate import FactorGate
from .policy_v2 import PreserveOptions
from .options import rollout
from ..round7.common import config,silent
from ..round7.policy import propose
from ..round7.episode import load_policy
from ..round7.replay_fork import ReplayFork
from ..round7.branches import continuation
from ..round6.env import LandingEnv,ZERO

def constant_gate():
 gate=FactorGate(dim=201).double()
 with torch.no_grad():
  for p in gate.parameters():p.zero_()
  for b,o in zip(gate.base_members,gate.option_members):b[-1].bias.fill_(0.);o[-1].bias.copy_(torch.tensor([math.log(.99999/.00001),math.log(.9/.1)],dtype=torch.float64))
 return gate

def main():
 torch.set_num_threads(1);start=time.time();spec=read(R7/'selected_candidate.json')['entries'][3];model=load_policy(spec['proposer'])
 with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=1.,fault_clock=None));obs,_=e.reset(seed=740035,options=dict(level=.296))
 pm=None;history=[]
 try:
  while e.t<109:
   with torch.no_grad():_,_,pm=propose(model,torch.as_tensor(obs[-1:],dtype=torch.float64),pm)
   history.append(ZERO.copy());obs,_,_,_,_=e.step(ZERO)
  with torch.no_grad():raw,on,pm=propose(model,torch.as_tensor(obs[-1:],dtype=torch.float64),pm)
  results={};checks=0
  with ReplayFork(e,history) as fork:
   base=continuation(fork.restore());results['baseline']=base
   for native in [True,False]:
    r=rollout(fork.restore(),model,pm,raw[0].numpy(),bool(on[0]),.5,24,native);results['native' if native else 'raw']=r
   checks=fork.checks;assert fork.maximum_error==0.
  assert base['outcome']=='stable_landing' and base['episode_steps']==645
  assert results['native']['outcome']=='timeout' and results['native']['episode_steps']==1440
  assert results['raw']['outcome']=='stable_landing' and results['raw']['episode_steps']==1137
 finally:e.close()
 for native,option in [(True,1),(False,5)]:
  policy=PreserveOptions(load_policy(spec['proposer']),constant_gate(),allowed_options=[option],minimum_gain=2.).double().eval()
  with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=1.,fault_clock=None));obs,_=e.reset(seed=740035,options=dict(level=.296))
  state=None
  try:
   while True:
    policy.minimum_gain=2. if e.t<109 else .005
    with torch.no_grad():action,state=policy.predict_actions(torch.as_tensor(obs[-1:],dtype=torch.float64),state)
    obs,_,d,t,info=e.step(action[0].numpy())
    if d or t:break
   expected=results['native' if native else 'raw'];assert info['outcome']==expected['outcome'] and info['episode_steps']==expected['episode_steps'];error=float(np.abs(np.asarray(info['terminal_position'])-expected['terminal_position']).max());assert error==0.;results['policy_native' if native else 'policy_raw']=dict(outcome=info['outcome'],episode_steps=info['episode_steps'],terminal_error=error)
  finally:e.close()
 close('h2_natural_option_semantics','New native and raw feedback options and their latched complete policies reproduce known natural continuations exactly; first-action projected features match applied motor delta.',results=results,prefix_checks=checks,seconds=time.time()-start)
if __name__=='__main__':main()
