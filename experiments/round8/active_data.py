"""Four paired futures at naturally replayed first-decision states, frozen proposer201."""
import argparse,time,hashlib,copy
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .common import OUT,read,save,sha,close
from .options import option_feature,rollout
from ..round7.common import config,silent,old_model
from ..round7.episode import load_policy
from ..round7.policy import propose,controller_descriptor
from ..round7.replay_fork import ReplayFork,checked_state
from ..round7.branches import continuation,outcome_class
from ..round7.gate_data import compact
from ..round7.fast_gtrxl import install
from ..round6.env import LandingEnv,ZERO
from ..round7.power import awake

def shared_proposal_context(model,x,state=None):
 assert model.method=='flashsac'
 memory=None if state is None else state[0];clock=torch.zeros(len(x),dtype=x.dtype,device=x.device) if state is None else state[1]
 features,new=model.features(x,memory);raw,_=model.proposal(features,deterministic=True,noise=None);on=model.select(features,raw).abs().amax(-1)>.002
 return raw,on,[new,clock+1],features

def collect(entry):
 torch.set_num_threads(1);start=time.time();case=entry['case'];split=entry['split'];folder=OUT/'data/active_v1'/split;dest=folder/(entry['tag']+'.npz');source_hash=sha(__file__)+sha(Path(__file__).with_name('options.py'))+sha(old_model(201))+sha(Path(__file__).parents[1]/'round7/replay_fork.py')+sha(OUT/'data/active_v1/allocation.json')+sha(OUT/'data/active_v1/collection_plan.json')
 if dest.exists():
  r=read(dest.with_suffix('.json'));assert r['source_hash']==source_hash;return r
 proposer=str(old_model(201));model=install(load_policy(proposer));identity=controller_descriptor(proposer)
 from ..round7.policy import canonical_risk
 canonical=canonical_risk().double().eval();a=model.prior.encoder.state_dict();b=canonical.encoder.state_dict();assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a);del canonical;reference=entry['baseline_reference']
 with np.load(entry['baseline_trace_path']) as z:expected=z['observation'].copy()
 with silent():e=LandingEnv(condition=case['condition'],config=config(noise_multiplier=case['noise'],fault_clock=case['clock']));obs,_=e.reset(seed=case['seed'],options=dict(level=case['level']))
 history=[];trace=[];pm=None;xx=[];yy=[];records=[];indexes=[];prefix_checks=0;max_error=0.;projection=0.;history_error=0.;queries=set(entry['ticks']);done_queries=[];explored=False
 try:
  while True:
   tick=e.t;assert tick<len(expected);error=float(np.abs(obs[-1]-expected[tick]).max());history_error=max(history_error,error);assert error==0.,('Natural base history differs from stored reference',case,tick,error);trace.append(obs[-1].copy());x=torch.as_tensor(obs[-1:],dtype=torch.float64)
   with torch.no_grad():raw,on,pm,context=shared_proposal_context(model,x,pm)
   exploration=entry.get('explore_observed_fault',False) and not explored and float(obs[-1,31:35].max())>.08 and bool(on[0]) and float(raw[0].abs().max())*.5>.002 and obs[-1,57]<.5
   if exploration:queries.add(tick);explored=True
   if tick in queries:
    assert bool(on[0]) and float(raw[0].abs().max())*.5>.002,('Query must be actionable',case,tick);before=checked_state(e);initial_raw=raw[0].numpy();duration=min(24,1440-tick)
    with ReplayFork(e,history) as fork:
     for future in range(4):
      material=repr((case,tick,future,'active_v1')).encode();future_seed=int.from_bytes(hashlib.sha256(material).digest()[:8],'little');branch=fork.restore()
      if future:branch.sensor_rng=np.random.default_rng(future_seed)
      base_full=continuation(branch);base=compact(base_full);assert future!=0 or (base['outcome']==reference['outcome'] and base['episode_steps']==reference['episode_steps'] and float(np.abs(np.asarray(base_full['terminal_position'])-reference['terminal_position']).max())==0.)
      for option,native_veto in [(1,True),(5,False)]:
       branch=fork.restore()
       if future:branch.sensor_rng=np.random.default_rng(future_seed)
       full=rollout(branch,model,pm,initial_raw,True,.5,duration,native_veto);candidate=compact(full);projection=max(projection,full['projection_feature_error']);label=outcome_class(base['stable_landing'],candidate['stable_landing']);xx.append(option_feature(obs[-1],context[0].numpy(),initial_raw,identity,duration,.5,native_veto,True));yy.append(label);indexes.append(tick);records.append(dict(**case,tick=tick,phase='actual_first_decision_distribution',future=future,option=option,duration=duration,alpha=.5,native_veto=native_veto,future_kind='natural_factual' if future==0 else 'resampled_paired',base=base,candidate=candidate,label=label,proposer=proposer))
     prefix_checks+=fork.checks;max_error=max(max_error,fork.maximum_error)
    actual=checked_state(e)
    for name,value in before.items():
     if isinstance(value,np.ndarray):np.testing.assert_array_equal(actual[name],value)
     else:assert actual[name]==value
    done_queries.append(tick)
   history.append(ZERO.copy());obs,_,d,t,info=e.step(ZERO)
   if d or t:break
  assert sorted(done_queries)==sorted(queries);assert info['outcome']==reference['outcome'] and info['episode_steps']==reference['episode_steps'];terminal_error=float(np.abs(np.asarray(info['terminal_position'])-reference['terminal_position']).max());assert terminal_error==0.;assert len(trace)==len(expected)
 finally:e.close()
 folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,x=np.asarray(xx,np.float32).reshape(-1,201),y=np.asarray(yy,np.int64),seed=np.full(len(yy),case['seed'],np.int64),trace=np.asarray(trace,np.float32),trace_index=np.asarray(indexes,np.int64),trace_start=np.zeros(len(yy),np.int64),episode_boundaries=np.asarray([0,len(trace)]),future=np.asarray([r['future'] for r in records]))
 result=dict(source_hash=source_hash,case=case,split=split,records=records,samples=len(yy),labels=dict(Counter(yy)),query_states=len(queries),actual_query_ticks=sorted(queries),prefix_replay_checks=prefix_checks,maximum_prefix_error=max_error,maximum_projection_error=projection,entire_caller_history_error=history_error,caller_terminal_position_error=terminal_error,seconds=time.time()-start,options=[1,5],futures=4,selection='Allocation uses native activity and prior first authorizations, never the base terminal outcome.')
 save(dest.with_suffix('.json'),result);return result

def collection_plan(allocation):
 dest=OUT/'data/active_v1/collection_plan.json'
 if dest.exists():return read(dest)
 entries=[]
 for entry in allocation['entries']:
  first=sorted({int(t) for t in entry['first_authorizations'] if t is not None});seed=entry['case']['seed'];sample=hashlib.sha256(('active-explore-v1:'+str(seed)).encode()).digest()[0]%4==0
  if first or sample:entries.append(dict(entry,ticks=first,explore_observed_fault=True,selection_role='all_actual_first_decisions' if first else 'route_sampled_observed_fault_exploration'))
 result=dict(entries=entries,source_allocation_sha=sha(OUT/'data/active_v1/allocation.json'),cases=len(entries),routes=len({r['case']['seed'] for r in entries}),known_first_decision_states=sum(len(r['ticks']) for r in entries),rule='Keep every actual first authorization of all10 prior gates. Also sample25% of remaining routes, across regimes, by independent deterministic hash. Add one native-on exploration query after an OBSERVED effectiveness loss>2%; no true fault flag or terminal outcome used. Avoid spending most new data on prefault native false alarms. Query states can be correlated; only routes are independent scenario clusters.',created_before_counterfactual_acquisition=True)
 save(dest,result);return result

def main(workers=16):
 torch.set_num_threads(1);start=time.time();allocation=read(OUT/'data/active_v1/allocation.json');entries=collection_plan(allocation)['entries'];pilots=[x for x in entries if (x['case']['seed']==810085 and x['case']['level']==.296 and x['case']['condition']=='estimated' and x['case']['noise']==1. and x['case']['clock'] is None) or (x['case']['seed']==760023 and x['case']['level']==.295 and x['case']['noise']==2. and x['case']['clock']==4.)]
 if not pilots:pilots=entries[:1]
 pilot_results=[collect(entry) for entry in pilots]
 close('h6_active_data_pilot','Actionable first-decision states replay naturally; native/raw24half options and four paired future continuations preserve caller and actuator semantics.',cases=[x['case'] for x in pilots],pilot_labels=[dict(case=x['case'],query_states=x['query_states'],labels=x['labels'],prefix_error=x['maximum_prefix_error'],caller_error=x['entire_caller_history_error']) for x in pilot_results])
 with awake(),ProcessPoolExecutor(max_workers=workers) as pool:
  tasks={pool.submit(collect,e):e['tag'] for e in entries};results=[]
  for job in as_completed(tasks):
   results.append(job.result());save(OUT/'data/active_v1/acquisition_progress.json',dict(cases_complete=len(results),target=len(entries),samples=sum(r['samples'] for r in results),seconds=time.time()-start))
  support={}
  for split in ['train','calibration']:
   rr=[r for x in results if x['split']==split for r in x['records']];support[split]=dict(routes=len({r['seed'] for r in rr}),query_states=len({(r['seed'],r['level'],r['condition'],r['noise'],r['clock'],r['tick']) for r in rr}),records=len(rr),classes={label:dict(records=sum(r['label']==label for r in rr),routes=len({r['seed'] for r in rr if r['label']==label})) for label in range(4)})
  close('h6_active_data','New fixed201 short-option support acquired at actual first-decision distributions; four paired futures and exact natural histories, train/calibration split by route.',support=support,cases=len(results),prefix_checks=sum(x['prefix_replay_checks'] for x in results),maximum_prefix_error=max(x['maximum_prefix_error'] for x in results),maximum_caller_history_error=max(x['entire_caller_history_error'] for x in results),maximum_projection_error=max(x['maximum_projection_error'] for x in results),seconds=time.time()-start)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args();main(a.workers)
