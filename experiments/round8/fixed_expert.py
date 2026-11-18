"""Five independently fitted gates, one development-selected frozen GTrXL prior.
This isolates learning of authorization from variability of the old proposal.
It is explicitly not five independent retrainings of the proposal network.
"""
import time
from pathlib import Path
import numpy as np
import torch
from .common import OUT,R7,read,save,close
from .acquire import stage
from .cache import run,certify
from .policy import PreservePolicy
from ..round7.evaluation import key,compare
from ..round7.statistics import paired
from ..round7.power import awake

def sub(r,b):
 k={key(x) for x in b['episodes']};return dict(r,episodes=[x for x in r['episodes'] if key(x) in k])

def main():
 torch.set_num_threads(1);start=time.time();review=read(OUT/'h4_new_route_generalization_conclusion.json');families=review['families'];assert families['native']['losses']>0
 close('h4_review','Larger development rejects both paired-proposer families: sparse improvements with one lost baseline landing, and extra baseline-prognosis training introduces additional losses despite better marginal forecasts. Separate fixed-proposal authorization fitting next.',families=families,selected_frozen_proposer=201,selection_role='Development only; final840/850/860 routes not opened.')
 with awake():
  stage('h4_matched_proposal_probe',['-m','experiments.round8.probe_loss'],'h4_matched_proposal_probe_conclusion.json')
  old=read(OUT/'development/natural_native_v1/baseline.json');new=read(OUT/'generalization/baseline.json');union={key(r):r for r in old['episodes']};union.update({key(r):r for r in new['episodes']});base=dict(episodes=list(union.values()));folder=OUT/'development/fixed_expert201';save(folder/'baseline.json',base)
  template=read(OUT/'h2_review_conclusion.json')['selected'];prior=read(R7/'selected_candidate.json')['entries'][0]['proposer'];specs=[dict(template,proposer=prior,gate=str(OUT/'factor_gates/base_options_calibrated'/f'seed_{seed}.pt')) for seed in range(501,506)]
  # Same recurrent representation is intentional: the canonical risk encoder is
  # precisely the frozen encoder of proposer201, not a rotated expert's memory.
  model=PreservePolicy.from_spec(specs[0]);a=model.proposer.prior.encoder.state_dict();b=model.risk.encoder.state_dict();assert a.keys()==b.keys();assert all(torch.equal(a[k],b[k]) for k in a)
  save(folder/'prospective_specs.json',dict(entries=specs,cases_per_gate=len(base['episodes']),fixed_proposer=201,gate_training_seeds=list(range(501,506)),same_proposal_and_risk_encoder_weights=True,scope='Five gate-fitting replicas on one development-selected frozen proposal; earlier paired-proposer results remain separate robustness diagnostics.'))
  picks=[r for r in base['episodes'] if r['seed'] in [740029,810239] and r['level'] in [.295,.296]]+[r for r in base['episodes'] if r['seed']==811000];hist=[]
  for r in picks:
   with np.load(r['trace_path']) as z:hist.append(z['observation'].copy())
  results=[];entries=[]
  for seed,spec in zip(range(501,506),specs):
   save(OUT/'status.json',dict(stage='h5_fixed_expert_'+str(seed),status='running'));flags=[]
   for device in ['cpu','cuda']:flags.append(certify(PreservePolicy.from_spec(spec).to(device).double(),hist))
   assert np.array_equal(*flags);target=folder/f'seed_{seed}';save(target/'device_check.json',dict(cases=len(hist),equal=True,flags=flags[0]))
   if seed==501:
    # Strict identity reuse: same complete current inference sources, checkpoint
    # bytes, policy specification and case. No relabeling or model approximation.
    previous_path=OUT/'generalization/native/seed_501/results.json';previous=read(previous_path);assert previous['spec']==spec;known={key(r):r for r in previous['episodes']};missing=dict(episodes=[r for r in base['episodes'] if key(r) not in known]);rest=run(spec,missing,target/'pending');assert rest['fingerprint']==previous['fingerprint'];reuse=[dict(known[key(r)],identical_policy_result_reused=True,baseline_reused_verified_off=False,fresh_simulation=False) for r in base['episodes'] if key(r) in known];r=dict(rest,episodes=rest['episodes']+reuse,logical_cases=len(base['episodes']),identical_policy_reuse=len(reuse),identical_policy_source=str(previous_path));save(target/'results.json',r)
   else:r=run(spec,base,target)
   results.append(r);m=compare(r,base);entries.append(dict(seed=seed,spec=spec,metrics=m));print('FIXED_EXPERT_GATE',seed,m,flush=True)
  groups={}
  for name,b in [('old_primary',dict(episodes=[r for r in base['episodes'] if 740000<=r['seed']<740064])),('new_default',dict(episodes=[r for r in base['episodes'] if 810000<=r['seed']<810256])),('new_shifted',dict(episodes=[r for r in base['episodes'] if 811000<=r['seed']<811064]))]:groups[name]=paired([sub(r,b) for r in results],b)
  close('h5_fixed_expert_validation','Fixed-proposer gate replicas evaluated on the entire accumulated development union. Previous variable-proposal results are not overwritten or represented as five full independent actor retrainings.',entries=entries,statistics=groups,gains=sum(r['metrics']['gains'] for r in entries),losses=sum(r['metrics']['losses'] for r in entries),cases_per_model=len(base['episodes']),seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='h5_fixed_expert_ready_for_review',status='running'))
if __name__=='__main__':main()
