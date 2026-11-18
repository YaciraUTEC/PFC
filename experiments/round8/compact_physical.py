"""Matched full physical flights for the representation-only option compaction."""
import time,hashlib
from pathlib import Path
import numpy as np
import torch
from .common import OUT,read,save,sha,close
from .loading_compact import Policy
from ..round7.common import config,silent
from ..round7.evaluation import key
from ..round7.fast_gtrxl import install
from ..round6.env import LandingEnv
from ..round7.power import awake

def main():
 torch.set_num_threads(1);start=time.time();source=read(OUT/'h8_checkpoint_selection_conclusion.json');root=Path(source['development_folder']);base={key(x):x for x in read(root/'baseline.json')['episodes']};jobs=[]
 for entry in source['entries']:
  r=read(root/f'seed_{entry["seed"]}/results.json');wanted=[x for x in r['episodes'] if x.get('policy_summary',{}).get('actual_motor_intervention_steps',0)>0 and (not base[key(x)]['stable_landing'] and x['stable_landing'] if entry['seed'] in [501,505] else base[key(x)]['stable_landing'] and not x['stable_landing'])];assert wanted
  row=min(wanted,key=lambda x:(not 812000<=x['seed']<812128,x['noise']!=1.,x['clock'] is not None,x['level']!=.295,x['seed']));jobs.append((entry,{k:row[k] for k in ['seed','level','condition','noise','clock']}))
 op=next(x for x in source['entries'] if x['seed']==505)
 for seed,level,noise,clock in [(760006,.296,2.,4.),(810177,.296,1.,None),(810085,.296,1.,None)]:jobs.append((op,dict(seed=seed,level=level,condition='estimated',noise=noise,clock=clock)))
 checks=[]
 with awake(),torch.no_grad():
  for entry,case in jobs:
   a=install(Policy.from_spec(entry['spec']));b=install(Policy.from_spec(dict(entry['spec'],compact_options=True)));worlds=[]
   try:
    with silent():
     for _ in range(2):
      e=LandingEnv(condition=case['condition'],config=config(noise_multiplier=case['noise'],fault_clock=case['clock']));o,_=e.reset(seed=case['seed'],options=dict(level=case['level']));worlds.append((e,o))
    sa=sb=None;tick=0;maximum=0.;commands=hashlib.sha256();starts=[[],[]]
    while True:
     ea,oa=worlds[0];eb,ob=worlds[1];assert np.array_equal(oa,ob);aa,sa=a.predict_actions(torch.as_tensor(oa[-1:],dtype=torch.float64),sa);bb,sb=b.predict_actions(torch.as_tensor(ob[-1:],dtype=torch.float64),sb);assert torch.equal(aa.float(),bb.float()),('Action changed',entry['seed'],case,tick);commands.update(aa.float().numpy().tobytes())
     for j,m in enumerate([a,b]):
      if bool(m.last_diagnostics['option_started'][0]):starts[j].append(tick)
     na,_,da,ta,ia=ea.step(aa.numpy()[0]);nb,_,db,tb,ib=eb.step(bb.numpy()[0]);assert da==db and ta==tb;err=float(np.abs(ea.core.obs_raw-eb.core.obs_raw).max());maximum=max(maximum,err);assert err==0.;assert np.array_equal(na,nb);worlds=[(ea,na),(eb,nb)];tick+=1
     if da or ta:
      assert ia['outcome']==ib['outcome'] and ia['first_contact']==ib['first_contact'] and ia['stable_seconds']==ib['stable_seconds'];break
    assert starts[0]==starts[1];record=dict(gate_seed=entry['seed'],case=case,steps=tick,outcome=ia['outcome'],option_starts=starts[0],maximum_physical_state_difference=maximum,commands_sha256=commands.hexdigest());checks.append(record);save(OUT/'execution_optimization/compact_physical_progress.json',dict(checks=checks));print('COMPACT_PHYSICAL',record,flush=True)
   finally:
    for e,_ in worlds:e.close()
 close('h10_compact_physical','Eight complete paired physical flights have identical float32 motor requests, observations, physical states, first contacts, settling times and termination; all five gates, actual rescues and losses included.',passed=True,checks=checks,maximum_physical_state_difference=max(x['maximum_physical_state_difference'] for x in checks),source_sha256=sha(__file__),loader_sha256=sha(Path(__file__).with_name('loading_compact.py')),seconds=time.time()-start)
if __name__=='__main__':main()
