"""Isolate internal cancellation from complete short-option terminal authorization."""
import time
import torch
from .common import OUT,read,save,close
from .cache_fast import run,certify
from .loading import Policy
from .study import subset
from ..round7.evaluation import compare,key
from ..round7.statistics import paired
from ..round7.power import awake
import numpy as np

def main():
 torch.set_num_threads(1)
 while not (OUT/'h6_active_validation_conclusion.json').exists() or not (OUT/'h8_fast_execution_equivalence_conclusion.json').exists():time.sleep(5)
 start=time.time();source=read(OUT/'h6_active_validation_conclusion.json');known_folder=OUT/'development/active_v1';base=read(known_folder/'baseline.json');new=read(known_folder/'new_baseline/baseline.json');new_keys={key(x) for x in new['episodes']};old=dict(episodes=[x for x in base['episodes'] if key(x) not in new_keys]);folder=OUT/'development/committed_v1';families={};ranking=[];all_specs={}
 for name,allowed in [('raw_short',[5]),('both_short',[1,5])]:all_specs[name]=[dict(row['spec'],version=3,fast_execution=True,option_mode='both',allowed_options=allowed) for row in source['entries']]
 save(folder/'prospective_specs.json',dict(families=all_specs,reason='Matched factual pilot760023: native4actualsteps leads to timeout1440; raw24half stable737 versus base653. Compare committed raw correction with explicit native alternative; do not generalize one case.',entry_rule='Native proposal ON at entry, aligned with acquired raw labels. Subsequent raw feedback has no internal Q veto; one option then permanent DSLPID.',primary_development_ranges=[[812000,812128],[813000,813032]],source_data='active_merged_v1',labels_before_any_final_case=True))
 with awake():
  # Existing native result is an exactly unchanged reference, not a new rollout.
  native=[read(known_folder/f'seed_{s}/results.json') for s in range(501,506)];entries=[dict(seed=501+i,spec=source['entries'][i]['spec'],metrics=compare(subset(r,new),new)) for i,r in enumerate(native)];families['native_short']=dict(entries=entries,statistics=paired([subset(r,new) for r in native],new),gains=sum(x['metrics']['gains'] for x in entries),losses=sum(x['metrics']['losses'] for x in entries));ranking.append(('native_short',families['native_short']))
  for name,specs in all_specs.items():
   entries=[];rr=[]
   for seed,spec in zip(range(501,506),specs):
    save(OUT/'status.json',dict(stage='h7_'+name+'_'+str(seed),status='running'));r=run(spec,new,folder/name/f'new_seed_{seed}');rr.append(r);entries.append(dict(seed=seed,spec=spec,metrics=compare(r,new)));print('COMMITTED_NEW',name,seed,entries[-1]['metrics'],flush=True)
   family=dict(entries=entries,statistics=paired(rr,new),gains=sum(x['metrics']['gains'] for x in entries),losses=sum(x['metrics']['losses'] for x in entries));families[name]=family;ranking.append((name,family))
  close('h7_committed_new_ablation','Internal-veto semantics isolated with identical expanded training, thresholds, initial support and fresh development cases.',families=families)
  rank=sorted([(n,f) for n,f in ranking if f['losses']==0],key=lambda x:(-x[1]['gains'],{'raw_short':0,'native_short':1,'both_short':2}[x[0]]));selected=None
  for name,family in rank:
   if name=='native_short':
    if source['losses']>0:continue
    selected=dict(source,selected_family=name);break
   rr=[];entries=[];target=folder/name/'complete';save(target/'baseline.json',base)
   for seed,row in zip(range(501,506),family['entries']):
    spec=row['spec'];save(OUT/'status.json',dict(stage='h7_'+name+'_known_'+str(seed),status='running'));earlier=read(folder/name/f'new_seed_{seed}/results.json');rest=run(spec,old,target/f'known_seed_{seed}');assert earlier['fingerprint']==rest['fingerprint'];r=dict(rest,episodes=earlier['episodes']+rest['episodes'],logical_cases=len(base['episodes']),fresh_simulations=earlier['fresh_simulations']+rest['fresh_simulations'],verified_off_reuse=earlier['verified_off_reuse']+rest['verified_off_reuse'],seconds=earlier['seconds']+rest['seconds'],combined_disjoint_case_sets=True);save(target/f'seed_{seed}/results.json',r);rr.append(r);entries.append(dict(seed=seed,spec=spec,metrics=compare(r,base)));print('COMMITTED_ALL',name,seed,entries[-1]['metrics'],flush=True)
   losses=sum(x['metrics']['losses'] for x in entries);save(target/'family_result.json',dict(entries=entries,gains=sum(x['metrics']['gains'] for x in entries),losses=losses))
   if losses==0:
    selected=dict(closed=True,entries=entries,selected_family=name,gains=sum(x['metrics']['gains'] for x in entries),losses=0,cases_per_model=len(base['episodes']),development_folder=str(target),selection_default_range=[812000,812128],selection_shifted_range=[813000,813032],statistics=paired(rr,base),source_training='active_merged_v1');break
  if selected is None:
   close('h7_committed_review','No expanded-support option family preserves all development baseline landings; continue development without opening final routes.',selected=None,families=families,seconds=time.time()-start);save(OUT/'status.json',dict(stage='h7_requires_revision',status='running'));return
  close('h7_committed_review','Selected family uses expanded counterfactual support and passes fresh-development and old factual consistency checks before formal tests.',**{k:v for k,v in selected.items() if k not in ['closed','conclusion','finished_epoch','seconds','new_ablation']},new_ablation=families,seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='h7_selected_ready_for_review',status='running',selected_family=selected['selected_family']))
if __name__=='__main__':main()
