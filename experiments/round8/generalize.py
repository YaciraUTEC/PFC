"""Isolate improved baseline prognosis, then evaluate both families on new routes."""
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

def subset(r,b):
 keys={key(v) for v in b['episodes']};return dict(r,episodes=[v for v in r['episodes'] if key(v) in keys])

def main():
 torch.set_num_threads(1);start=time.time();old=read(OUT/'h3_development_validation_conclusion.json');assert old['losses']==0
 close('h3_review','Preservation held across all five paired replicas and15360 development comparisons, but seven gains occur only in previous routes and new32-route subset has no gain. Improve baseline prognosis in isolation before final testing.',gains=old['gains'],losses=old['losses'],statistics=old['statistics'])
 with awake():
  stage('h4_logging_equivalence',['-m','experiments.round8.logging_check'],'h4_logging_equivalence_conclusion.json')
  for seed in range(502,506):
   if not (OUT/'factor_gates/base_forecast_augmented'/f'seed_{seed}_complete.json').exists():stage('h4_base_extra_'+str(seed),['-m','experiments.round8.base_extra','--seed',str(seed)])
  close('h4_baseline_forecast_training','Only baseline-success heads updated; preservation/rescue weights and policy continuation fixed for a causal ablation.',seeds=list(range(501,506)),fits=[read(OUT/'factor_gates/base_forecast_augmented'/f'seed_{s}_complete.json') for s in range(501,506)])
  # This is development, intentionally distinct from the reserved840/850/860.
  cc=cases(range(810000,810256),levels=[.295,.296])+cases(range(811000,811064),levels=[.295,.296],conditions=['estimated'],noises=[2.],clocks=[4.,5.,7.,8.])
  retained=read(OUT/'development/new_routes/baseline.json');known={key(r):r for r in retained['episodes']};pending=[r for r in cc if key(r) not in known];new=baseline(pending,OUT/'generalization/new_baseline',workers=16);lookup=dict(known);lookup.update({key(r):r for r in new['episodes']});base=dict(episodes=[lookup[key(r)] for r in cc]);save(OUT/'generalization/baseline.json',base)
  specs={}
  for family in ['native','forecast_augmented']:
   tag='base_options_calibrated' if family=='native' else 'base_forecast_augmented';specs[family]=[dict(row['spec'],gate=str(OUT/'factor_gates'/tag/f"seed_{row['seed']}.pt")) for row in old['entries']]
  save(OUT/'generalization/prospective_specs.json',dict(entries=specs,new_default_routes=256,new_clock_routes=64,cases_per_model=len(cc),scope='Development generalization and isolated base-prognosis ablation; reserved final-test routes still untouched.'))
  picks=[r for r in base['episodes'] if r['seed'] in [810000,810001] and r['level']==.296]+[r for r in base['episodes'] if r['seed'] in [811000,811001]];hist=[]
  for row in picks:
   with np.load(row['trace_path']) as z:hist.append(z['observation'].copy())
  families={}
  for family,entries in specs.items():
   results=[];rows=[]
   for i,spec in enumerate(entries):
    seed=501+i;save(OUT/'status.json',dict(stage='h4_generalization_'+family+'_'+str(seed),status='running'));flags=[]
    for device in ['cpu','cuda']:flags.append(certify(PreservePolicy.from_spec(spec).to(device).double(),hist))
    assert np.array_equal(*flags)
    folder=OUT/'generalization'/family/f'seed_{seed}';save(folder/'device_check.json',dict(cases=len(hist),equal=True,flags=flags[0]));r=run(spec,base,folder);results.append(r);m=compare(r,base);rows.append(dict(seed=seed,spec=spec,metrics=m));print('NEW_ROUTE_GENERALIZATION',family,seed,m,flush=True)
   stats={}
   for name,prefix in [('default',810),('shifted',811)]:
    b=dict(episodes=[r for r in base['episodes'] if r['seed']//1000==prefix]);stats[name]=paired([subset(r,b) for r in results],b)
   families[family]=dict(entries=rows,statistics=stats,gains=sum(r['metrics']['gains'] for r in rows),losses=sum(r['metrics']['losses'] for r in rows));save(OUT/'generalization'/family/'complete.json',families[family])
  close('h4_new_route_generalization','Both unchanged and improved-baseline-prognosis families evaluated in five paired replicas on larger new-route development; no final route consumed.',families=families,cases_per_model=len(cc),seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='h4_ready_for_review',status='running'))
if __name__=='__main__':main()
