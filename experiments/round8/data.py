"""Counterfactual preservation/rescue on states visited by the unmodified base.
All options end in the same base continuation. Keep strict boundary examples.
"""
import argparse,time,copy
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .common import OUT,save,read,close,sha
from .options import OPTIONS,option_feature,rollout
from ..round7.common import config,silent,old_model
from ..round7.episode import load_policy
from ..round7.policy import propose,canonical_risk,risk_features,controller_descriptor
from ..round7.replay_fork import ReplayFork,checked_state
from ..round7.branches import continuation,outcome_class
from ..round7.gate_data import compact
from ..round7.fast_gtrxl import install
from ..round6.env import LandingEnv,ZERO
from ..round7.power import awake

def cell(index):
 b=index//5
 return dict(proposer=201+index%5,condition=['exact','estimated'][b%2],noise=[1.,2.][(b//2)%2],clock=[None,4.,5.,6.,7.,8.][(b//4)%6])

def collect_one(index,split,tag='base_options_v1'):
 torch.set_num_threads(1);started=time.time();folder=OUT/'data'/tag/split;path=folder/f'group_{index:03d}.npz'
 source_hash=sha(__file__)+sha(Path(__file__).with_name('options.py'))
 if path.exists():
  cached=read(path.with_suffix('.json'));assert cached['source_hash']==source_hash;return cached
 settings=cell(index);seed=(820000 if split=='train' else 830000)+index;proposer=str(old_model(settings['proposer']));model=install(load_policy(proposer));risk=install(canonical_risk().double().eval());identity=controller_descriptor(proposer)
 with silent():e=LandingEnv(condition=settings['condition'],config=config(noise_multiplier=settings['noise'],fault_clock=settings['clock']))
 xx=[];yy=[];records=[];history_obs=[];trace_indices=[];trace_starts=[];bounds=[];prefix_checks=0;max_error=0.;projection_error=0.;reused=0;natural_checks=[]
 try:
  for level in [.295,.296]+([0.,.25] if index%12==0 else []):
   with silent():obs,_=e.reset(seed=seed,options=dict(level=level))
   pm=rm=None;history=[];query_kinds=set();last_query=-999;begin=len(history_obs);bounds.append(begin)
   while True:
    history_obs.append(obs[-1].copy());qindex=len(history_obs)-1;x=torch.as_tensor(obs[-1:],dtype=torch.float64)
    with torch.no_grad():raw,native,pm=propose(model,x,pm);context,rm=risk_features(risk,x,rm)
    native_on=bool(native[0]);initial_raw=raw[0].numpy();height=float(e.measured[2]);eligible=e.t>=24 and not e.touchdown_seen and (e.estimated_eta.min()<.98 or e.route_reached)
    kind=None
    if eligible and 'onset' not in query_kinds and e.estimated_eta.min()<.98:kind='onset'
    elif eligible and e.route_reached and 'descent' not in query_kinds:kind='descent'
    elif eligible and e.route_reached and 'braking' not in query_kinds and e.t-last_query>=48 and (height<.42 or e.t>720):kind='braking'
    if kind is not None and e.t-last_query>=24:
     query_kinds.add(kind);last_query=e.t;tick=e.t;caller=checked_state(e)
     with ReplayFork(e,history) as fork:
      for future in range(2):
       future_seed=18000000+seed*31+tick*7+future;branch=fork.restore();branch.sensor_rng=np.random.default_rng(future_seed);base=compact(continuation(branch));single=None
       for option,(budget,alpha,use_native) in enumerate(OPTIONS):
        duration=min(budget,1440-tick)
        if duration==1 and use_native and not native_on:candidate=dict(base);reused+=1
        elif duration==1 and not use_native and native_on and single is not None:candidate=dict(single);reused+=1
        else:
         branch=fork.restore();branch.sensor_rng=np.random.default_rng(future_seed);full=rollout(branch,model,pm,initial_raw,native_on,alpha,duration,use_native);projection_error=max(projection_error,full['projection_feature_error']);candidate=compact(full)
        if option==0 and native_on:single=candidate
        label=outcome_class(base['stable_landing'],candidate['stable_landing']);xx.append(option_feature(obs[-1],context[0].numpy(),initial_raw,identity,duration,alpha,use_native,native_on));yy.append(label);trace_indices.append(qindex);trace_starts.append(begin)
        records.append(dict(seed=seed,level=level,condition=settings['condition'],noise=settings['noise'],clock=settings['clock'],tick=tick,phase=kind,future=future,option=option,duration=duration,alpha=alpha,native_veto=use_native,base=base,candidate=candidate,label=label,proposer=proposer))
      prefix_checks+=fork.checks;max_error=max(max_error,fork.maximum_error)
     actual=checked_state(e)
     for name,value in caller.items():
      if isinstance(value,np.ndarray):np.testing.assert_array_equal(actual[name],value)
      else:assert actual[name]==value
    history.append(ZERO.copy());obs,_,done,truncated,info=e.step(ZERO)
    if done or truncated:break
   # Independent no-query flight checks the caller, not two restored worlds.
   with silent():independent=LandingEnv(condition=settings['condition'],config=config(noise_multiplier=settings['noise'],fault_clock=settings['clock']));independent.reset(seed=seed,options=dict(level=level))
   try:
    untouched=continuation(independent);assert untouched['outcome']==info['outcome'] and untouched['episode_steps']==info['episode_steps'];err=float(np.max(np.abs(np.asarray(untouched['terminal_position'])-info['terminal_position'])));assert err==0.;natural_checks.append(dict(level=level,error=err))
   finally:independent.close()
 finally:e.close()
 folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,x=np.asarray(xx,np.float32).reshape(-1,201),y=np.asarray(yy,np.int64),seed=np.full(len(yy),seed,np.int64),trace=np.asarray(history_obs,np.float32),trace_index=np.asarray(trace_indices,np.int64),trace_start=np.asarray(trace_starts,np.int64),episode_boundaries=np.asarray(bounds+[len(history_obs)],np.int64),future=np.asarray([r['future'] for r in records],np.int64))
 result=dict(source_hash=source_hash,seed=seed,index=index,split=split,settings=settings,records=records,samples=len(yy),labels=dict(Counter(yy)),seconds=time.time()-started,prefix_replay_checks=prefix_checks,maximum_prefix_error=max_error,maximum_projection_error=projection_error,natural_caller_checks=natural_checks,identical_single_step_reuse=reused,input_dimension=201,behavior='DSLPID only before first option',fork_method='natural_prefix_replay',near_boundary_examples_included=True,options=OPTIONS)
 save(path.with_suffix('.json'),result);return result

def main(split,groups=120,workers=8,tag='base_options_v1'):
 start=time.time();folder=OUT/'data'/tag/split;existing=[i for i in range(groups) if (folder/f'group_{i:03d}.json').exists()];todo=[i for i in range(groups) if i not in existing];results=[read(folder/f'group_{i:03d}.json') for i in existing]
 with awake(),ProcessPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(collect_one,i,split,tag):i for i in todo}
  for future in as_completed(futures):
   r=future.result();results.append(r);summary=dict(split=split,groups_complete=len(results),target=groups,samples=sum(v['samples'] for v in results),seconds=time.time()-start);save(OUT/f'data_{tag}_{split}_progress.json',summary);print('BASE_OPTION_DATA',summary,flush=True)
 labels=Counter()
 for r in results:labels.update({int(k):v for k,v in r['labels'].items()})
 close('h2_'+tag+'_'+split,'Natural-base states paired with explicit native/raw options; strict boundary examples retained.',groups=groups,samples=sum(r['samples'] for r in results),labels=dict(labels),prefix_checks=sum(r['prefix_replay_checks'] for r in results),maximum_prefix_error=max(r['maximum_prefix_error'] for r in results),maximum_projection_error=max(r['maximum_projection_error'] for r in results),seconds=time.time()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','calibration'],required=True);p.add_argument('--groups',type=int,default=120);p.add_argument('--workers',type=int,default=8);p.add_argument('--tag',default='base_options_v1');a=p.parse_args();main(a.split,a.groups,a.workers,a.tag)
