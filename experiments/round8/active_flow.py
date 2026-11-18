"""Close the support ablation, aggregate real stopping states, fit and validate."""
import time,os
from pathlib import Path
import torch
from .common import OUT,read,save,sha,close
from .acquire import stage
from .fit import main as fit
from .cache import baseline,run
from .study import subset
from ..round7.evaluation import cases,key,compare
from ..round7.statistics import paired
from ..round7.power import awake

def merged():
 folder=OUT/'data/active_merged_v1'
 for tag,prefix in [('base_options_v1','legacy_'),('active_v1','active_')]:
  for split in ['train','calibration']:
   target=folder/split;target.mkdir(parents=True,exist_ok=True)
   for src in (OUT/'data'/tag/split).glob('*'):
    if src.suffix not in ['.json','.npz']:continue
    dst=target/(prefix+src.name)
    if dst.exists():assert sha(src)==sha(dst)
    else:os.link(src,dst)
 close('h6_merged_support','Original counterfactual data preserved and augmented with disjoint route-split first-decision queries. Hardlinks preserve exact original bytes.',folder=str(folder),scope='No final test routes used. Old810/740 development now contributes training/calibration labels and is not advertised as an independent held-out validation.')
 return folder

def validation():
 torch.set_num_threads(1);start=time.time();folder=OUT/'development/active_v1';oldbase=read(OUT/'development/short_margin_v1/baseline.json');cc=cases(range(812000,812128),levels=[.295,.296])+cases(range(812000,812016),levels=[0.,.25])+cases(range(813000,813032),levels=[.295,.296],conditions=['estimated'],noises=[2.],clocks=[4.,5.,7.,8.]);fresh=baseline(cc,folder/'new_baseline',workers=16);base=dict(episodes=oldbase['episodes']+fresh['episodes']);save(folder/'baseline.json',base);previous=read(OUT/'h5_conservative_validation_conclusion.json')['entries'];specs=[dict(x['spec'],gate=str(OUT/f'factor_gates/active_v1/seed_{x["seed"]}.pt')) for x in previous];save(folder/'prospective_specs.json',dict(entries=specs,new_cases=len(cc),known_cases=len(oldbase['episodes']),selection_only_on_new_ranges=[[812000,812128],[813000,813032]],source_training='active_merged_v1',scope='Fresh812/813 development for generalization; old acquisition-source union is a consistency/stress check, not independent validation.'))
 models=[];entries=[]
 for seed,spec in zip(range(501,506),specs):
  save(OUT/'status.json',dict(stage='h6_active_validation_'+str(seed),status='running'));r=run(spec,base,folder/f'seed_{seed}');models.append(r);m=compare(r,base);entries.append(dict(seed=seed,spec=spec,metrics=m));print('ACTIVE_GATE',seed,m,flush=True)
 groups={}
 for name,b in [('new_default',dict(episodes=[r for r in fresh['episodes'] if 812000<=r['seed']<812128])),('new_shifted',dict(episodes=[r for r in fresh['episodes'] if 813000<=r['seed']<813032]))]:groups[name]=paired([subset(r,b) for r in models],b)
 # The unaugmented family is evaluated on exactly the same genuinely new cases.
 references={}
 for family,tag in [('bootstrap_short','base_options_calibrated'),('complete_support','full_support_v1')]:
  rr=[];rows=[]
  for x in previous:
   seed=x['seed'];spec=dict(x['spec'],gate=str(OUT/f'factor_gates/{tag}/seed_{seed}.pt'));save(OUT/'status.json',dict(stage='h6_reference_'+family+'_'+str(seed),status='running'));r=run(spec,fresh,folder/'references'/family/f'seed_{seed}');rr.append(r);rows.append(dict(seed=seed,metrics=compare(r,fresh)));print('ACTIVE_DATA_ABLATION',family,seed,rows[-1]['metrics'],flush=True)
  references[family]=dict(entries=rows,statistics=paired(rr,fresh),gains=sum(x['metrics']['gains'] for x in rows),losses=sum(x['metrics']['losses'] for x in rows))
 close('h6_active_validation','Expanded actual stopping-state support evaluated on fresh development and old factual consistency union; unchanged options/thresholds isolate data addition from inference-rule changes.',entries=entries,statistics=paired(models,base),new_statistics=groups,references=references,gains=sum(r['metrics']['gains'] for r in entries),losses=sum(r['metrics']['losses'] for r in entries),cases_per_model=len(base['episodes']),development_folder=str(folder),selection_default_range=[812000,812128],selection_shifted_range=[813000,813032],seconds=time.time()-start)
 save(OUT/'status.json',dict(stage='h6_active_ready_for_review',status='running'))

def main():
 torch.set_num_threads(1)
 while not (OUT/'h5_full_support_validation_conclusion.json').exists():time.sleep(5)
 with awake():
  old=read(OUT/'h5_full_support_validation_conclusion.json');close('h5_support_review','Complete-data ensemble ablation closed before targeted expansion. Audit finds only three own short-option rescue routes and one harm route in initial training; increase independent counterfactual support at actual authorization times next.',previous_results=dict(gains=old['gains'],losses=old['losses'],statistics=old['statistics']),effective_support=read(OUT/'option1_support_audit.json'))
  stage('h6_active_scan',['-m','experiments.round8.active_scan'],'h6_active_query_scan_conclusion.json')
  stage('h6_active_acquisition',['-m','experiments.round8.active_data','--workers','16'],'h6_active_data_conclusion.json')
  data=merged()
  for seed in range(501,506):
   marker=OUT/f'factor_gates/active_v1/seed_{seed}_complete.json'
   if not marker.exists():fit(seed=seed,folder=data,tag='active_v1',route_bootstrap=False)
  close('h6_active_fits','Five complete-support conditional gate fits trained/calibrated after paired active query acquisition; frozen GTrXL actor and inference rule unchanged.',seeds=list(range(501,506)))
  validation()
if __name__=='__main__':main()
