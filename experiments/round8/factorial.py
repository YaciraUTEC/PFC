"""Data addition x internal-veto semantics on exactly the same fresh development.
This explanatory ablation does not silently select a new final controller.
"""
import time
import torch
from .common import OUT,read,save,close
from .cache_fast import run
from .study import subset
from ..round7.evaluation import compare
from ..round7.statistics import paired
from ..round7.power import awake

def main():
 torch.set_num_threads(1)
 # Avoid another physical16-worker pool during committed-family validation.
 while not (OUT/'h7_committed_review_conclusion.json').exists():time.sleep(5)
 start=time.time();new=read(OUT/'development/active_v1/new_baseline/baseline.json');source=read(OUT/'h5_conservative_validation_conclusion.json')['entries'];folder=OUT/'development/factorial_v1';specs=[dict(x['spec'],gate=str(OUT/f'factor_gates/full_support_v1/seed_{x["seed"]}.pt'),version=3,fast_execution=True,option_mode='both',allowed_options=[5]) for x in source];save(folder/'prospective_specs.json',dict(entries=specs,role='Complete2x2 attribution: original/expanded data x native-veto/committed-raw. Models and inference thresholds unchanged within each contrast; baseline exactly same new812/813cases.',selection='Explanatory ablation only; no unchecked legacy raw variant is deployed.'))
 with awake():
  results=[];entries=[]
  for seed,spec in zip(range(501,506),specs):
   save(OUT/'status.json',dict(stage='h7_factorial_old_raw_'+str(seed),status='running'));r=run(spec,new,folder/f'seed_{seed}');results.append(r);entries.append(dict(seed=seed,metrics=compare(r,new)));print('FACTORIAL_OLD_RAW',seed,entries[-1]['metrics'],flush=True)
  cells={};paths=[('original_native',OUT/'development/active_v1/references/complete_support','seed_{}'),('expanded_native',OUT/'development/active_v1','seed_{}'),('expanded_raw',OUT/'development/committed_v1/raw_short','new_seed_{}')]
  for label,root,pattern in paths:
   rr=[subset(read(root/pattern.format(seed)/'results.json'),new) for seed in range(501,506)];rows=[dict(seed=501+i,metrics=compare(r,new)) for i,r in enumerate(rr)];cells[label]=dict(entries=rows,statistics=paired(rr,new),gains=sum(x['metrics']['gains'] for x in rows),losses=sum(x['metrics']['losses'] for x in rows))
  cells['original_raw']=dict(entries=entries,statistics=paired(results,new),gains=sum(x['metrics']['gains'] for x in entries),losses=sum(x['metrics']['losses'] for x in entries));close('h7_data_veto_factorial','Complete2x2 matched development ablation distinguishes data expansion from internal cancellation semantics; final test still separate.',cells=cells,cases_per_model=len(new['episodes']),seconds=time.time()-start);save(OUT/'status.json',dict(stage='h7_factorial_ready_for_review',status='running'))
if __name__=='__main__':main()
