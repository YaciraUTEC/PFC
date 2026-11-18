"""Restrict first decisions to observed-degradation support, then select a checkpoint."""
import time
import torch
from .common import OUT,read,save,close
from .cache_guarded import run
from .study import subset
from ..round7.evaluation import compare
from ..round7.statistics import paired
from ..round7.power import awake

def main():
 torch.set_num_threads(1)
 while not (OUT/'h7_data_veto_factorial_conclusion.json').exists():time.sleep(5)
 start=time.time();source=read(OUT/'h6_active_validation_conclusion.json');base=read(OUT/'development/active_v1/new_baseline/baseline.json');folder=OUT/'development/observed_support_v1';families={};specs={}
 for name,allowed,mode in [('raw',[5],'both'),('native',[1],'native'),('both',[1,5],'both')]:specs[name]=[dict(x['spec'],version=4,fast_execution=True,allowed_options=allowed,option_mode=mode,observed_loss_min=.02,sigma_multiplier=0.) for x in source['entries']]
 for prefix,conclusion in [('bootstrap','h5_conservative_validation_conclusion.json'),('original','h5_full_support_validation_conclusion.json')]:
  for menu,allowed,mode in [('native',[1],'native'),('raw',[5],'both')]:specs[prefix+'_'+menu]=[dict(x['spec'],version=4,fast_execution=True,allowed_options=allowed,option_mode=mode,observed_loss_min=.02,sigma_multiplier=0.) for x in read(OUT/conclusion)['entries']]
 save(folder/'prospective_specs.json',dict(families=specs,reason='Expanded-model501 lost760006 after intervention1.1875s before actual fault4s with observed loss0 and eta error0, predicted harm.001284. First-decision distribution moved outside acquisition support. Use the already-declared acquisition observable loss>2% condition at entry, no true fault state or schedule.',learning_unchanged=True,selection_before_any_final_case=True,threshold_origin='The same .02 observed loss used in collection_plan before counterfactual acquisition; no new ground-truth motor indicator.'))
 close('h9_support_review','Completed data/veto ablation is followed by an observable entry support restriction; low predicted risk alone allowed an unsupported healthy-state correction.',previous=read(OUT/'h7_data_veto_factorial_conclusion.json'),diagnostic=read(OUT/'h6_early_loss_diagnostics.json'))
 with awake():
  for name,entries in specs.items():
   rr=[];rows=[]
   for seed,spec in zip(range(501,506),entries):
    save(OUT/'status.json',dict(stage='h9_observed_'+name+'_'+str(seed),status='running'));r=run(spec,base,folder/name/f'seed_{seed}');rr.append(r);rows.append(dict(seed=seed,spec=spec,metrics=compare(r,base)));print('OBSERVED_SUPPORT',name,seed,rows[-1]['metrics'],flush=True)
   families[name]=dict(entries=rows,statistics=paired(rr,base),gains=sum(x['metrics']['gains'] for x in rows),losses=sum(x['metrics']['losses'] for x in rows))
  close('h9_observed_support_validation','Observed entry support compared on exactly the same fresh development cases for native/raw/both menus, with unchanged learned weights and preservation/gain thresholds.',families=families,cases_per_model=len(base['episodes']),seconds=time.time()-start)
  from .select_checkpoint import main as select
  select()
if __name__=='__main__':main()
