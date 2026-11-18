"""Select the concrete safe-improving checkpoint on development only.
All other fits remain explicit sensitivity diagnostics; no held-out case is read.
"""
import time
from pathlib import Path
import torch
from .common import OUT,read,save,sha,close
from .execution_guarded import run
from .study import subset
from ..round7.evaluation import key,compare
from ..round7.power import awake

def semantic(spec):return {k:v for k,v in spec.items() if k!='fast_execution'}
def checked_reference(r,spec,b):
 assert semantic(r['spec'])==semantic(spec);assert {key(x) for x in r['episodes']}=={key(x) for x in b['episodes']}
 for name,digest in r['fingerprint'].items():assert sha(spec[name] if name in ['gate','proposer'] else Path(__file__).parent/name)==digest
 return ('policy_guarded.py' in r['fingerprint']) if spec.get('version')==4 else ('fast_policy.py' in r['fingerprint'])
def reference_rows(rows):return [dict(x,fresh_simulation=False,baseline_reused_verified_off=False,identical_policy_result_reused=True) for x in rows]

def complete(candidate,base,new,old,root):
 seed=candidate['seed'];folder=root/candidate['family'];dest=folder/f'seed_{seed}/results.json';spec=candidate['spec'];save(folder/'baseline.json',base)
 if dest.exists():r=read(dest);checked_reference(r,spec,base);return r
 full=Path(candidate['full']) if candidate.get('full') else None
 if full is not None and full.exists():
  original=read(full)
  if checked_reference(original,spec,base):
   r=dict(original,spec=spec,episodes=reference_rows(original['episodes']),fresh_simulations=0,verified_off_reuse=0,identical_policy_reuse=len(base['episodes']),seconds=0.,identity_source=str(full),identity_source_sha=sha(full),identity_scope='Same implementation and bytes; fast_execution is an explicit loader selector, not an input used by FastFactory. No new physical flight.');save(dest,r);return r
 nr=read(candidate['new']);nr=subset(nr,new)
 if not checked_reference(nr,spec,new):nr=run(spec,new,folder/f'new_seed_{seed}')
 oldpath=Path(candidate['old']) if candidate.get('old') else None;old_reference=False
 if oldpath is not None and oldpath.exists():
  original=read(oldpath);original=subset(original,old)
  if checked_reference(original,spec,old):orr=original;old_reference=True
  else:orr=run(spec,old,folder/f'known_seed_{seed}')
 else:orr=run(spec,old,folder/f'known_seed_{seed}')
 assert nr['fingerprint']==orr['fingerprint'];rows=reference_rows(orr['episodes']) if old_reference else orr['episodes'];r=dict(orr,spec=spec,episodes=rows+reference_rows(nr['episodes']),logical_cases=len(base['episodes']),fresh_simulations=0 if old_reference else orr['fresh_simulations'],verified_off_reuse=0 if old_reference else orr['verified_off_reuse'],identical_policy_reuse=len(new['episodes'])+(len(old['episodes']) if old_reference else 0),seconds=0. if old_reference else orr['seconds'],identity_new_source=candidate['new'],identity_old_source=str(oldpath) if old_reference else None,scope='Disjoint known-consistency and fresh-development cases, actual fast implementation; exact result references counted separately from new simulations.');save(dest,r);return r

def qualify(candidate,result,sweep,base,root):
 seed=candidate['seed'];folder=root/'with_severity_sweep'/candidate['family'];dest=folder/f'seed_{seed}/results.json';union=dict(episodes=base['episodes']+sweep['episodes']);save(folder/'baseline.json',union)
 if dest.exists():r=read(dest);checked_reference(r,candidate['spec'],union);return r
 extra=run(candidate['spec'],sweep,folder/f'sweep_seed_{seed}');assert result['fingerprint']==extra['fingerprint'];r=dict(extra,episodes=reference_rows(result['episodes'])+extra['episodes'],logical_cases=len(union['episodes']),identical_policy_reuse=len(result['episodes']),seconds=extra['seconds'],previous_complete_source=str(root/candidate['family']/f'seed_{seed}/results.json'),scope='Complete prior development plus prospectively declared noncritical/severity30 development sweep; no formal case used.');save(dest,r);return r

def main():
 torch.set_num_threads(1)
 while not (OUT/'h7_data_veto_factorial_conclusion.json').exists() or not (OUT/'h9_observed_support_validation_conclusion.json').exists():time.sleep(5)
 assert read(OUT/'h8_fast_execution_equivalence_conclusion.json')['passed'];start=time.time();base=read(OUT/'development/active_v1/baseline.json');new=read(OUT/'development/active_v1/new_baseline/baseline.json');old=read(OUT/'development/short_margin_v1/baseline.json');root=OUT/'development/checkpoint_selection';definitions=[('expanded_native',OUT/'development/active_v1','seed_{}',OUT/'development/active_v1','seed_{}',True),('expanded_raw',OUT/'development/committed_v1/raw_short','new_seed_{}',OUT/'development/committed_v1/raw_short/complete','known_seed_{}',False),('expanded_both',OUT/'development/committed_v1/both_short','new_seed_{}',OUT/'development/committed_v1/both_short/complete','known_seed_{}',False),('bootstrap_native',OUT/'development/active_v1/references/bootstrap_short','seed_{}',OUT/'development/short_margin_v1','seed_{}',False),('complete_native',OUT/'development/active_v1/references/complete_support','seed_{}',OUT/'development/full_support_v1','seed_{}',False),('complete_raw',OUT/'development/factorial_v1','seed_{}',None,None,False)];definitions += [('guarded_'+name,OUT/'development/observed_support_v1'/name,'seed_{}',None,None,False) for name in ['native','raw','both','bootstrap_native','bootstrap_raw','original_native','original_raw']];candidates=[]
 for family,nroot,npattern,oroot,opattern,isfull in definitions:
  for seed in range(501,506):
   npth=nroot/npattern.format(seed)/'results.json';nr=read(npth);spec=dict(nr['spec'],fast_execution=True);rr=subset(nr,new);nm=compare(rr,new);default=dict(episodes=[x for x in new['episodes'] if 812000<=x['seed']<812128 and x['condition']=='estimated' and x['level'] in [.295,.296]]);shift=dict(episodes=[x for x in new['episodes'] if 813000<=x['seed']<813032]);dm=compare(subset(rr,default),default);sm=compare(subset(rr,shift),shift);oldpath=oroot/opattern.format(seed)/'results.json' if oroot is not None else None;fullpath=npth if isfull else (nroot/'complete'/f'seed_{seed}/results.json' if family.startswith('expanded_') else None);known_losses=None
   if fullpath is not None and fullpath.exists():known_losses=compare(subset(read(fullpath),old),old)['losses']
   elif oldpath is not None and oldpath.exists():known_losses=compare(subset(read(oldpath),old),old)['losses']
   active=sum(x.get('policy_summary',{}).get('actual_motor_intervention_steps',0)>0 for x in rr['episodes']);candidates.append(dict(family=family,seed=seed,spec=spec,new=str(npth),old=str(oldpath) if oldpath is not None else None,full=str(fullpath) if fullpath is not None else None,new_metrics=nm,new_default_gains=dm['gains'],new_shift_gains=sm['gains'],known_losses=known_losses,new_active_cases=active))
 save(root/'prospective_checkpoint_registry.json',dict(candidates=candidates,selection='Observable support is a compared ablation, not a categorical delivery requirement; full actual-policy8064 qualification must show zero baseline losses and positive new-default rescues. Rank new-default gains, new-clock gains, fewer actual interventions, smaller option menu, deterministic name/seed. No final outcome used.',estimand_amendment_sha=sha(OUT/'AMENDMENT_FIXED_CHECKPOINT_ESTIMAND_BEFORE_TEST.json')))
 eligible=[c for c in candidates if c['new_metrics']['losses']==0 and c['known_losses'] in [None,0] and c['new_default_gains']>0];eligible.sort(key=lambda c:(-c['new_default_gains'],-c['new_shift_gains'],c['new_active_cases'],len(c['spec']['allowed_options']),c['family'],c['seed']));selected=None
 with awake():
  from .cache import baseline
  from ..round7.evaluation import cases
  sweep=baseline(cases(range(814000,814064),levels=[.02,.10,.28,.29,.30]),root/'severity_sweep',workers=16);union=dict(episodes=base['episodes']+sweep['episodes'])
  for c in eligible:
   save(OUT/'status.json',dict(stage='h8_checkpoint_'+c['family']+'_'+str(c['seed']),status='running'));r=qualify(c,complete(c,base,new,old,root),sweep,base,root);m=compare(r,union);print('CHECKPOINT_CONSISTENCY',c['family'],c['seed'],m,flush=True)
   if m['losses']==0:selected=c;break
  if selected is None:
   close('h8_checkpoint_selection','No individual checkpoint satisfies development preservation and rescue; continue with targeted refinement before opening final routes.',selected=None,registry=candidates,seconds=time.time()-start);save(OUT/'status.json',dict(stage='h8_requires_refinement',status='running'));return
  entries=[]
  for c in [x for x in candidates if x['family']==selected['family']]:
   save(OUT/'status.json',dict(stage='h8_family_diagnostic_'+str(c['seed']),status='running'));r=qualify(c,complete(c,base,new,old,root),sweep,base,root);entries.append(dict(seed=c['seed'],spec=c['spec'],metrics=compare(r,union)))
  op=selected['seed'];assert next(x['metrics']['losses'] for x in entries if x['seed']==op)==0;losses=sum(x['metrics']['losses'] for x in entries);close('h8_checkpoint_selection','Concrete operational checkpoint selected exclusively on matched development and complete known-flight preservation. Other replicas remain explicit sensitivity diagnostics.',selected_family=selected['family'],recommended_operational_gate=op,operational_development_losses=0,entries=entries,gains=sum(x['metrics']['gains'] for x in entries),losses=losses,eligible_gate_seeds=[x['seed'] for x in entries if x['metrics']['losses']==0],cases_per_model=len(union['episodes']),development_severity_sweep=dict(seeds=[814000,814064],levels=[.02,.10,.28,.29,.30],cases=1280),development_folder=str(root/'with_severity_sweep'/selected['family']),selection_default_range=[812000,812128],selection_shifted_range=[813000,813032],registry=candidates,seconds=time.time()-start);save(OUT/'status.json',dict(stage='h8_checkpoint_selected',status='running',family=selected['family'],gate=op))
if __name__=='__main__':main()
