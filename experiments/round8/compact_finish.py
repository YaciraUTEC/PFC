"""Finish representation-only equivalence, paired timing and honest result migration."""
import time,copy,gc,shutil
from pathlib import Path
import numpy as np
import torch
from .common import OUT,ROOT,TASK,read,save,sha,close
from .loading_compact import Policy
from .cache_compact import fingerprint_for
from .evaluation import evaluate
from ..round7.evaluation import key,compare
from ..round7.fast_gtrxl import install
from ..round6.models import physics_observation
from ..round7.power import awake

def main():
 torch.set_num_threads(1);start=time.time()
 for name in ['h10_compact_first_decisions','h10_compact_physical']:
  p=OUT/(name+'_conclusion.json')
  while not p.exists():time.sleep(5)
  assert read(p)['passed']
 source=read(OUT/'h8_checkpoint_selection_conclusion.json');oldroot=Path(source['development_folder']);base=read(oldroot/'baseline.json');entries=source['entries'];newroot=oldroot.parent/'compact_guarded_raw';newroot.mkdir(exist_ok=True);save(newroot/'baseline.json',base)
 with np.load(OUT/'execution_optimization/compact_first_decisions.npz') as z:ambiguous=z['ambiguous'].copy()
 checks=[];migrated=[]
 with awake(),torch.no_grad():
  for mi,entry in enumerate(entries):
   spec=dict(entry['spec'],compact_options=True);oldpath=oldroot/f'seed_{entry["seed"]}/results.json';r=read(oldpath);lookup={key(x):x for x in r['episodes']};cases=[{k:x[k] for k in ['seed','level','condition','noise','clock']} for j,x in enumerate(base['episodes']) if ambiguous[mi,j]];fresh=None
   if cases:
    folder=OUT/'execution_optimization'/f'near_threshold_{entry["seed"]}';marker=folder/'results.json'
    fresh=read(marker) if marker.exists() else evaluate(install(Policy.from_spec(spec)),cases,workers=min(16,len(cases)),path=marker,capture_directory=folder/'traces',capture_baseline=base)
    for a in fresh['episodes']:
     b=lookup[key(a)];assert a['outcome']==b['outcome'] and a['episode_steps']==b['episode_steps'] and a['first_contact']==b['first_contact'] and a['stable_seconds']==b['stable_seconds'];assert a['terminal_position']==b['terminal_position'];assert a['policy_summary']['option_start_seconds']==b['policy_summary']['option_start_seconds']
   checks.append(dict(seed=entry['seed'],near_threshold_physical_replays=len(cases),passed=True))
   outfolder=newroot/f'seed_{entry["seed"]}';rows=[];actual={key(x):x for x in fresh['episodes']} if fresh else {}
   for x in r['episodes']:
    if key(x) in actual:rows.append(dict(actual[key(x)],fresh_simulation=True,baseline_reused_verified_off=False,identical_policy_result_reused=False))
    else:rows.append(dict(x,original_fresh_simulation=x.get('fresh_simulation',False),fresh_simulation=False,identical_policy_result_reused=not x.get('baseline_reused_verified_off',False)))
   off=sum(x.get('baseline_reused_verified_off',False) for x in rows);same=sum(x.get('identical_policy_result_reused',False) for x in rows)
   result=dict(r,episodes=rows,spec=spec,fingerprint=fingerprint_for(spec),fresh_simulations=len(cases),verified_off_reuse=off,identical_policy_reuse=same,seconds=fresh['seconds'] if fresh else 0.,execution_backend='compact_admissible_options_shared_encoder_completed_option_skip',representation_equivalence_source=str(oldpath),representation_equivalence_source_sha256=sha(oldpath),representation_equivalence_proofs={name:sha(OUT/(name+'_conclusion.json')) for name in ['h10_compact_first_decisions','h10_compact_physical']},scope='Identical compact representation: all actual first decisions checked on8064 histories; selected actor actions, feedback duration and amplitude unchanged. Reused physical results are explicitly counted, never advertised as fresh flights. Near-threshold cases replayed physically.')
   assert len(rows)==len(base['episodes'])==off+same+len(cases);save(outfolder/'results.json',result);m=compare(result,base);assert m==entry['metrics'];migrated.append(dict(entry,spec=spec,metrics=m));print('COMPACT_MIGRATED',checks[-1],flush=True)
  close('h10_compact_near_threshold','Direct CPU physical replay of every numerically near-boundary first decision agrees with original full flights.',passed=True,checks=checks)
  # Fixed recorded history and context, alternating dense/compact order on one CPU.
  entry=next(x for x in entries if x['seed']==505);dense=Policy.from_spec(entry['spec']);compact=Policy.from_spec(dict(entry['spec'],compact_options=True));case=next(x for x in base['episodes'] if x['seed']==810085 and x['level']==.296 and x['condition']=='estimated' and x['noise']==1 and x['clock'] is None)
  with np.load(case['trace_path']) as z:h=z['observation'][:69].copy()
  obs=torch.as_tensor(h[None],dtype=torch.float64);clean=physics_observation(obs);enc,_=dense.proposer.prior.encoder(clean);context=torch.cat([clean,enc],-1)[:,-1];last=obs[:,-1];raw,_=dense.proposer.proposal(context,deterministic=True,noise=None);native=dense.proposer.select(context,raw).abs().amax(-1)>.002
  def call(m):
   aa,bb,x=m.build_options(last,context,raw,native);return m.assess(x,aa)
  for _ in range(40):call(dense);call(compact)
  times=[[],[]]
  for i in range(1000):
   for j in ([0,1] if i%2==0 else [1,0]):
    before=time.perf_counter_ns();call([dense,compact][j]);times[j].append((time.perf_counter_ns()-before)/1e6)
  timing={name:dict(samples=len(v),p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99))) for name,v in zip(['dense8','compact1'],times)}
  close('h10_compact_paired_timing','Same recorded context, same CPU, alternating order; this isolates option-table gate computation. Whole-chain timing is measured separately and includes hardware variability.',passed=True,timing=timing,options=[8,1],scope='Inference arithmetic timing only, not full deadline evidence; old and new whole-chain timings were not simultaneous.')
 # Archive the old failed preparation, preserving all receipts. No final allocations exist.
 old=OUT/'final/attempt_1';archive=OUT/'execution_optimization/attempt_1_before_compaction'
 if old.exists():
  assert not (old/'frozen.json').exists() and not (old/'allocations.json').exists()
  assert not archive.exists()
  assert old.resolve().is_relative_to(TASK.resolve()) and archive.resolve().is_relative_to(TASK.resolve())
  shutil.move(str(old),str(archive))
  p=OUT/'attempt_1_prepared_conclusion.json'
  if p.exists():shutil.move(str(p),str(OUT/'execution_optimization/attempt_1_prepared_before_compaction.json'))
 close('h10_compact_execution','Only admissible option rows retained: same first decisions on all8064cases and all5gates, matched full physical flights, near-threshold replays and explicitly reused equivalent development outcomes. Actor, thresholds, amplitude, history and intervention semantics unchanged.',passed=True,selected_family='compact_guarded_raw',recommended_operational_gate=505,operational_development_losses=0,entries=migrated,losses=sum(x['metrics']['losses'] for x in migrated),cases_per_model=8064,development_folder=str(newroot),selection_default_range=source['selection_default_range'],selection_shifted_range=source['selection_shifted_range'],prior_conclusion_sha256=sha(OUT/'h8_checkpoint_selection_conclusion.json'),proofs={name:sha(OUT/(name+'_conclusion.json')) for name in ['h10_compact_first_decisions','h10_compact_physical','h10_compact_near_threshold','h10_compact_paired_timing']},seconds=time.time()-start)
 gc.collect();torch.cuda.empty_cache()
 print('COMPACT_READY_FOR_NEW_PREFLIGHT',flush=True)
if __name__=='__main__':main()