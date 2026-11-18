"""Whole-flight and OFF-proof equivalence of exact-task execution optimization."""
import time
import numpy as np
import torch
from .common import OUT,read,save,close
from .fast_policy import FastFactory
from .policy_v2 import PreserveOptions
from .policy_v3 import EntryPreserveOptions
from .cache import certify
from ..round7.episode import fly
from ..round7.fast_gtrxl import install
from ..round7.power import awake

def main():
 torch.set_num_threads(1);start=time.time();template=read(OUT/'h5_conservative_validation_conclusion.json')['entries'][0]['spec'];specs=[dict(template,gate=str(OUT/'factor_gates/base_options_calibrated/seed_502.pt')),dict(template,gate=str(OUT/'factor_gates/active_v1/seed_501.pt')),dict(template,gate=str(OUT/'factor_gates/active_v1/seed_501.pt'),version=3,option_mode='both',allowed_options=[5])];cc=[dict(seed=760023,level=.295,condition='estimated',noise=2.,clock=4.),dict(seed=810085,level=.296,condition='estimated',noise=1.,clock=None),dict(seed=760006,level=.296,condition='estimated',noise=2.,clock=4.),dict(seed=740000,level=0.,condition='estimated',noise=1.,clock=None),dict(seed=810150,level=.296,condition='estimated',noise=2.,clock=None),dict(seed=812000,level=.295,condition='estimated',noise=2.,clock=None)];checks=[]
 with awake():
  for index,spec in enumerate(specs):
   slow=install((EntryPreserveOptions if spec.get('version')==3 else PreserveOptions).from_spec(spec));fast=install(FastFactory.from_spec(spec));maximum=0.;rows=[];hist=[]
   for case in cc:
    a,ta=fly(**case,model=slow,collect=True);b,tb=fly(**case,model=fast,collect=True);assert a['outcome']==b['outcome'] and a['episode_steps']==b['episode_steps'] and a['stable_landing']==b['stable_landing'];error=float(np.abs(np.asarray(a['terminal_position'])-b['terminal_position']).max());maximum=max(maximum,error);assert error==0.
    for field in ['s','ns','a']:
     aa=np.asarray([x[field] for x in ta]);bb=np.asarray([x[field] for x in tb]);err=float(np.abs(aa-bb).max());maximum=max(maximum,err);assert err==0.,(index,case,field,err)
    hist.append(np.asarray([x['s'] for x in ta]));rows.append(dict(case=case,outcome=a['outcome'],steps=a['episode_steps'],maximum_difference=maximum));save(OUT/'fast_preflight_progress.json',dict(policy=index,completed_cases=len(rows),maximum_difference=maximum,seconds=time.time()-start));print('FAST_WHOLE_FLIGHT',index,case['seed'],a['outcome'],maximum,flush=True)
   flags=[]
   for cls in [EntryPreserveOptions if spec.get('version')==3 else PreserveOptions,FastFactory]:
    flags.append(certify(cls.from_spec(spec).cuda().double(),hist))
   assert np.array_equal(*flags);checks.append(dict(spec=spec,cases=rows,whole_trajectory_observation_and_executed_action_difference=maximum,off_proof_flags_identical=True))
  close('h8_fast_execution_equivalence','Shared GTrXL context and post-option neural skip preserve full physical trajectories and actions, including nominal flights, successful rescue and known bad interventions; OFF-proof identity checked.',checks=checks,passed=True,seconds=time.time()-start)
if __name__=='__main__':main()
