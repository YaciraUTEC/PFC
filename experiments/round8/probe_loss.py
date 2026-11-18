"""Matched-state control-proposal probe, separate from the learned gate."""
import time
import numpy as np
import torch
from .common import OUT,R7,read,save,close
from .options import rollout
from ..round7.common import config,silent
from ..round7.episode import load_policy
from ..round7.policy import propose
from ..round7.replay_fork import ReplayFork
from ..round7.branches import continuation
from ..round6.env import LandingEnv,ZERO

def main():
 torch.set_num_threads(1);start=time.time();case=dict(seed=810239,level=.296,condition='estimated',noise=1.,clock=None);tick=82;paths=[r['proposer'] for r in read(R7/'selected_candidate.json')['entries']];models=[load_policy(path) for path in paths];states=[None]*5
 with silent():e=LandingEnv(condition=case['condition'],config=config(noise_multiplier=case['noise'],fault_clock=case['clock']));obs,_=e.reset(seed=case['seed'],options=dict(level=case['level']))
 history=[]
 try:
  while e.t<tick:
   with torch.no_grad():
    for j,m in enumerate(models):_,_,states[j]=propose(m,torch.as_tensor(obs[-1:],dtype=torch.float64),states[j])
   history.append(ZERO.copy());obs,_,_,_,_=e.step(ZERO)
  proposals=[]
  with torch.no_grad():
   for j,m in enumerate(models):raw,on,states[j]=propose(m,torch.as_tensor(obs[-1:],dtype=torch.float64),states[j]);proposals.append((raw[0].numpy(),bool(on[0])))
  rows=[]
  with ReplayFork(e,history) as fork:
   base=continuation(fork.restore())
   for j,m in enumerate(models):
    raw,on=proposals[j];r=rollout(fork.restore(),m,states[j],raw,on,.5,24,True);rows.append(dict(proposer=201+j,outcome=r['outcome'],steps=r['episode_steps'],first_contact=r.get('first_contact'),terminal_position=r['terminal_position'],initial_delta=raw,initial_native_on=on))
   assert fork.maximum_error==0.
  factual=read(OUT/'generalization/native/seed_505/results.json');original=next(r for r in factual['episodes'] if all(r[k]==v for k,v in case.items()));assert rows[4]['outcome']==original['outcome'] and rows[4]['steps']==original['episode_steps'];err=float(np.abs(np.asarray(rows[4]['terminal_position'])-original['terminal_position']).max());assert err==0.
 finally:e.close()
 close('h4_matched_proposal_probe','Matched state and identical24-half native option separate proposal quality from gate prediction; factual failure reproduced exactly. Individual case, not a global ranking.',case=case,tick=tick,baseline=dict(outcome=base['outcome'],steps=base['episode_steps']),proposers=rows,factual_position_error=err,seconds=time.time()-start)
if __name__=='__main__':main()
