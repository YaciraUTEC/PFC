"""Advance a zero-loss pilot to five paired actor/gate replicas and wider cases."""
import time
import numpy as np
import torch
from .common import OUT,R7,read,save,close
from .acquire import stage
from .cache import run,certify,baseline
from .policy import PreservePolicy
from ..round7.evaluation import cases,key,compare
from ..round7.statistics import paired
from ..round7.power import awake

def subset(result,base):
 kk={key(r) for r in base['episodes']};return dict(result,episodes=[r for r in result['episodes'] if key(r) in kk])

def main():
 torch.set_num_threads(1);start=time.time();pilot=read(OUT/'h2_flight_pilot_conclusion.json');winner=next(r for r in pilot['results'] if r['name']=='all_members_both');assert winner['metrics']['losses']==0 and winner['metrics']['gains']>0
 # Raw-only options yielded no gain; choose the smaller supported action family.
 selected=dict(winner['spec'],option_mode='native')
 close('h2_review','Two pilot rescues without lost baseline landings. Conditional risk alone was insufficient on old-state data; natural-base data and coherent native options advance. Native-only wins the simplicity tie because raw-only options added no pilot rescues.',selected=selected,pilot=winner['metrics'],acceptance_pending=True)
 with awake():
  for seed in [504,505]:stage('h3_extra_fit_'+str(seed),['-m','experiments.round8.fit','--seed',str(seed),'--folder',str(OUT/'data/base_options_v1'),'--tag','base_options_v1'])
  stage('h3_extra_calibration',['-m','experiments.round8.recalibrate','--seeds','501','502','503','504','505'])
  primary=read(R7/'final/primary/baseline.json');shifted=read(R7/'final/shifted/baseline.json');new=baseline(cases(range(810000,810032),levels=[0.,.25,.295,.296]),OUT/'development/new_routes',workers=16)
  base=dict(episodes=primary['episodes']+shifted['episodes']+new['episodes']);folder=OUT/'development/natural_native_v1';save(folder/'baseline.json',base)
  specs=[]
  for i in range(5):specs.append(dict(selected,proposer=read(R7/'selected_candidate.json')['entries'][i]['proposer'],gate=str(OUT/'factor_gates/base_options_calibrated'/f'seed_{501+i}.pt')))
  save(folder/'prospective_specs.json',dict(entries=specs,cases_per_model=len(base['episodes']),selection='Development only; all five paired replicas are assessed before opening reserved840/850/860 routes.',old_primary_routes=64,old_shifted_routes=32,new_development_routes=32))
  histories=[]
  picks=[r for r in primary['episodes'] if r['seed'] in [740029,740035] and r['level'] in [.295,.296]]+[r for r in shifted['episodes'] if r['seed']==760000]
  for r in picks:
   with np.load(r['trace_path']) as z:histories.append(z['observation'].copy())
  entries=[];results=[]
  for i,spec in enumerate(specs):
   save(OUT/'status.json',dict(stage='h3_development_'+str(501+i),status='running'))
   flags=[]
   for device in ['cpu','cuda']:flags.append(certify(PreservePolicy.from_spec(spec).to(device).double(),histories))
   assert np.array_equal(*flags);save(folder/f'seed_{501+i}_device_check.json',dict(cases=len(histories),equal=True,flags=flags[0]))
   r=run(spec,base,folder/f'seed_{501+i}');results.append(r);m=compare(r,base);entries.append(dict(seed=501+i,spec=spec,metrics=m));print('WIDER_DEVELOPMENT',501+i,m,flush=True)
  groups={}
  for name,b in [('old_primary',primary),('old_shifted',shifted),('new_routes',new)]:groups[name]=paired([subset(r,b) for r in results],b)
  all_losses=sum(r['metrics']['losses'] for r in entries);all_gains=sum(r['metrics']['gains'] for r in entries)
  close('h3_development_validation','All five paired replicas completed old-curve, shifted-time and new-route development; empirical preservation is assessed separately from mean improvement.',entries=entries,statistics=groups,gains=all_gains,losses=all_losses,cases_per_model=len(base['episodes']),seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='h3_development_ready_for_review',status='running'))
if __name__=='__main__':main()
