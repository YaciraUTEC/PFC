"""Development selection, independent execution checks, complete-chain timing and regression."""
import argparse,time,subprocess,sys,gc
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,read,save,sha,close
from .study import path
from .cache import baseline,certify
from .execution_compact import run
from .evaluation import evaluate
from .loading_compact import Policy as PreservePolicy
from ..round7.evaluation import key
from ..round7.fast_gtrxl import install
from ..round7.episode import infer
from ..round7.common import config,silent
from ..round6.env import LandingEnv
from ..round7.power import awake

def select(attempt,conclusion):
 folder=path(attempt);dest=folder/'selection.json'
 if dest.exists():return read(dest)
 source=read(OUT/conclusion);assert source['closed'] and source.get('selected',True) is not None,'No development checkpoint selected'
 entries=source['entries'];assert len(entries)==5
 # Choose one real deployment checkpoint solely on development: default critical
 # gains, shifted gains, fewer actual interventions, then smaller seed.
 dev=Path(source.get('development_folder',OUT/'development/short_margin_v1'));default_range=source.get('selection_default_range',[810000,810256]);shift_range=source.get('selection_shifted_range',[811000,811064]);table=[];base=read(dev/'baseline.json');old={key(x):x for x in base['episodes']}
 for row in entries:
  seed=row['seed'];r=read(dev/f'seed_{seed}/results.json');assert r['spec']==row['spec'];default=shift=active=0
  for x in r['episodes']:
   gain=int(not old[key(x)]['stable_landing'] and x['stable_landing']);default+=gain and default_range[0]<=x['seed']<default_range[1];shift+=gain and shift_range[0]<=x['seed']<shift_range[1];active+=x.get('policy_summary',{}).get('actual_motor_intervention_steps',0)>0
  table.append(dict(seed=seed,new_default_gains=int(default),new_shifted_gains=int(shift),active_cases=int(active),development_losses=row['metrics']['losses']))
 feasible=[r for r in table if r['development_losses']==0];assert feasible;selected=source.get('recommended_operational_gate') or max(feasible,key=lambda r:(r['new_default_gains'],r['new_shifted_gains'],-r['active_cases'],-r['seed']))['seed'];assert next(r for r in table if r['seed']==selected)['development_losses']==0
 value=dict(entries=entries,operational_gate=selected,selection_table=table,development_losses=0,development_family_losses=source['losses'],development_cases_per_gate=source['cases_per_model'],source=str(OUT/conclusion),source_sha=sha(OUT/conclusion),development_folder=str(dev),scope='Five independently fitted authorization gates, one frozen GTrXL proposer201. Checkpoint selection occurs on development only, not formal test. Gate replicas do not represent independent actor retrainings.')
 save(dest,value)
 # The previous prospective mixture assignment was ambiguous for deployment.
 # This is declared before generating/reading any formal evaluation outcome.
 save(folder/'AMENDMENT_OPERATIONAL_CHECKPOINT.json',dict(created_before_final_allocation=True,operational_gate=selected,change='Independent assigned statistical stream uses the actual development-selected operational checkpoint; random level and noise only. All five fits remain evaluated. Crossed bootstrap and family losses are secondary sensitivity diagnostics, as declared in AMENDMENT_FIXED_CHECKPOINT_ESTIMAND_BEFORE_TEST.json; operational checkpoint has the primary zero-loss and gain requirements.',reason='Family-mixture evidence should not be advertised as checkpoint-specific deployment evidence.',prior_design=str(OUT/'AMENDMENT_FINAL_POWER_AND_SEQUENTIAL_EVIDENCE.json')))
 return value

def preflight(attempt,chosen):
 folder=path(attempt);dest=folder/'preflight.json'
 if dest.exists():return read(dest)
 union=read(Path(chosen['development_folder'])/'baseline.json');b=dict(episodes=[x for x in union['episodes'] if x['seed'] in [810085,810150] and x['level'] in [.295,.296]])
 assert len(b['episodes'])==16;hist=[]
 for r in b['episodes']:
  with np.load(r['trace_path']) as z:hist.append(z['observation'].copy())
 checks=[]
 for row in chosen['entries']:
  seed=row['seed'];spec=row['spec'];flags=[]
  for device in ['cpu','cuda']:flags.append(certify(PreservePolicy.from_spec(spec).to(device).double(),hist))
  assert np.array_equal(*flags);torch.cuda.empty_cache();cached=run(spec,b,folder/'preflight'/f'gate_{seed}');fresh=evaluate(install(PreservePolicy.from_spec(spec)),[{k:r[k] for k in ['seed','level','condition','noise','clock']} for r in b['episodes']],workers=16);aa={key(r):r for r in cached['episodes']};bb={key(r):r for r in fresh['episodes']};largest=0.
  for case,a in aa.items():
   c=bb[case];assert a['outcome']==c['outcome'] and a['stable_landing']==c['stable_landing'] and a['episode_steps']==c['episode_steps'];error=float(np.max(np.abs(np.asarray(a['terminal_position'])-c['terminal_position'])));largest=max(largest,error);assert error<1e-8
  checks.append(dict(seed=seed,cases=16,cpu_gpu_off_flags_equal=True,fresh_vs_identity_max_terminal_position_error=largest));print('R8_PREFLIGHT',checks[-1],flush=True)
 save(dest,dict(passed=True,checks=checks,scope='Development-only routes include active rescue and the rejected late-intervention diagnosis; all exact/estimated and noise1/2 combinations.'))
 return read(dest)

def latency(attempt,chosen):
 folder=path(attempt);dest=folder/'latency.json'
 if dest.exists():return read(dest)
 results=[]
 for name,spec in [('DSLPID',None)]+[('gate_'+str(x['seed']),x['spec']) for x in chosen['entries']]:
  model=None if spec is None else install(PreservePolicy.from_spec(spec));values=[];inference=[];control=[];strata=[];flights=[]
  plans=[(810085,None,1.,.296),(811059,4.,2.,.296)]
  if model is not None:
   seed_id=int(name.split('_')[-1]);rr=read(Path(chosen['development_folder'])/f'seed_{seed_id}/results.json');active=[x for x in rr['episodes'] if x['condition']=='estimated' and x.get('policy_summary',{}).get('actual_motor_intervention_steps',0)>0]
   if active:
    worst=max(active,key=lambda x:(x['policy_summary']['actual_motor_intervention_steps'],-x['seed']));plan=(worst['seed'],worst['clock'],worst['noise'],worst['level'])
    if plan not in plans:plans.append(plan)
  for seed,clock,noise,level in plans:
   with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=noise,fault_clock=clock))
   physics=[0.];original=e._impact_step
   def timed(rpm):
    before=time.perf_counter_ns()
    try:return original(rpm)
    finally:physics[0]+=(time.perf_counter_ns()-before)/1e6
   e._impact_step=timed
   try:
    with silent():obs,_=e.reset(seed=seed,options=dict(level=level))
    state=None;tick=0
    while True:
     postfault=e.fault_step is not None and not e.touchdown_seen;physics[0]=0.;before=time.perf_counter_ns();a,state=infer(model,obs,state);mid=time.perf_counter_ns();dd=getattr(model,'last_diagnostics',{}) if model is not None else {};attempted=bool(dd.get('authorized_delta',torch.zeros(1,4)).abs().max()>.002);obs,_,d,t,info=e.step(a);after=time.perf_counter_ns();tick+=1
     if tick>24:
      neural=(mid-before)/1e6;base=(after-mid)/1e6-physics[0];assert base>=0;inference.append(neural);control.append(base);values.append(neural+base);strata.append((postfault,attempted,bool(a[5]>0),bool(dd.get('proposal_evaluated',dd.get('neural_evaluated',torch.zeros(1,dtype=torch.bool))))))
     if d or t:flights.append(dict(seed=seed,clock=clock,noise=noise,level=level,outcome=info['outcome'],steps=tick));break
   finally:e.close()
  vv=np.asarray(values);mask=np.asarray(strata,bool);groups={}
  for j,label in enumerate(['postfault_flight','authorized_nonzero','requested_intervention','decision_network_evaluated']):
   v=vv[mask[:,j]];groups[label]=dict(samples=len(v),p95_ms=float(np.quantile(v,.95)) if len(v) else None,p99_ms=float(np.quantile(v,.99)) if len(v) else None)
  result=dict(name=name,samples=len(vv),p50_ms=float(np.quantile(vv,.5)),p95_ms=float(np.quantile(vv,.95)),p99_ms=float(np.quantile(vv,.99)),max_ms=float(vv.max()),inference_p95_ms=float(np.quantile(inference,.95)),base_p95_ms=float(np.quantile(control,.95)),deadline_ms=1000/48,miss_fraction=float((vv>1000/48).mean()),conditional_latency=groups,flights=flights,scope='Complete inference + controller/observer/supervisor/observation bookkeeping. Only plant integration and first24warmup ticks excluded; measured in one process without training or parallel flight pools. Empirical timing, not hard real-time guarantee.')
  result['guarded_p95_ms']=max([result['p95_ms']]+[g['p95_ms'] for g in groups.values() if g['samples']>=24]);results.append(result);save(folder/'latency_progress.json',dict(results=results));print('R8_LATENCY',name,result['guarded_p95_ms'],flush=True)
 save(dest,dict(results=results));return read(dest)

def tests(attempt):
 folder=path(attempt);dest=folder/'tests.json'
 if dest.exists():assert read(dest)['exit_code']==0;return
 args=[sys.executable,'-B','-m','pytest','experiments/tests','-q','-p','no:cacheprovider'];started=time.time()
 with (folder/'tests.log').open('w',encoding='utf-8') as handle:r=subprocess.run(args,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
 save(dest,dict(exit_code=r.returncode,seconds=time.time()-started,args=args));assert r.returncode==0,'Regression failed; inspect '+str(folder/'tests.log')
 print('R8_REGRESSION_PASSED',round(time.time()-started),flush=True)

def main(attempt=1,conclusion='h5_conservative_validation_conclusion.json'):
 torch.set_num_threads(1)
 while not (OUT/conclusion).exists():time.sleep(5)
 with awake():
  chosen=select(attempt,conclusion);save(OUT/'status.json',dict(stage='prepare_execution_preflight',status='running'));preflight(attempt,chosen);gc.collect();torch.cuda.empty_cache();time.sleep(10);save(OUT/'status.json',dict(stage='prepare_latency',status='running'));latency(attempt,chosen);save(OUT/'status.json',dict(stage='prepare_regression',status='running'));tests(attempt)
  close('attempt_'+str(attempt)+'_prepared','Development selection, execution equivalence, CPU/GPU proof checks, complete-chain timing and regression completed before formal tests.',operational_gate=chosen['operational_gate']);save(OUT/'status.json',dict(stage='attempt_'+str(attempt)+'_prepared',status='running'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);p.add_argument('--conclusion',default='h5_conservative_validation_conclusion.json');a=p.parse_args();main(a.attempt,a.conclusion)
