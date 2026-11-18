"""Prospective frozen-candidate study with fresh routes and revision-wise spending."""
import argparse,time,secrets,sys
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,TASK,OUT,LEVELS,read,save,sha,close
from .cache import baseline
from .execution_compact import run
from .sequential import assignment,evidence,spending
from ..round7.evaluation import cases,key,compare
from ..round7.statistics import paired
from ..round7.power import awake

def path(attempt):return OUT/'final'/f'attempt_{attempt}'
def subset(r,b):
 kk={key(x) for x in b['episodes']};return dict(r,episodes=[x for x in r['episodes'] if key(x) in kk])
def point_statistics(r,b,replicates=20000):
 old={key(x):x for x in b['episodes']};routes=sorted({x['seed'] for x in b['episodes']});values=[]
 for seed in routes:
  rows=[x for x in r['episodes'] if x['seed']==seed and x['condition']=='estimated' and x['level'] in [.295,.296]]
  assert rows;values.append(np.mean([int(x['stable_landing'])-int(old[key(x)]['stable_landing']) for x in rows]))
 values=np.asarray(values);rng=np.random.default_rng(772);means=[]
 for first in range(0,replicates,250):means.extend(values[rng.integers(len(values),size=(min(250,replicates-first),len(values)))].mean(1))
 return dict(mean=float(values.mean()),lower95=float(np.quantile(means,.025)),upper95=float(np.quantile(means,.975)),route_clusters=len(routes),replicates=replicates,scope='Route-cluster paired bootstrap for the development-selected operational checkpoint; repeated noise and severity settings remain clustered.')
def verify(attempt):
 f=read(path(attempt)/'frozen.json')
 for rel,d in f['sources'].items():assert sha(ROOT/rel)==d,'Frozen source changed: '+rel
 for p,d in f['artifacts'].items():assert sha(p)==d,'Frozen artifact changed: '+p
 assert sha(path(attempt)/'selection.json')==f['selection_sha'];return f

def freeze(attempt):
 folder=path(attempt)
 if (folder/'frozen.json').exists():return verify(attempt)
 chosen=read(folder/'selection.json');assert len(chosen['entries'])==5;assert chosen['development_losses']==0
 assert read(folder/'preflight.json')['passed'];assert read(folder/'tests.json')['exit_code']==0
 lat=read(folder/'latency.json');assert all(x['guarded_p95_ms']<1000/48 for x in lat['results'])
 sources={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(v in p.relative_to(ROOT).parts for v in ['runs','.git','__pycache__']) and str(p.relative_to(ROOT)).replace('\\','/') not in ['experiments/round8/report.py','experiments/round8/handoff.py']}
 # Reporting files are intentionally excluded; reporting cannot change inference.
 files={str(TASK/'PFC/runs/intervention_round6/phase1_conclusion.json'),str(ROOT/'runs/gtrxl_round7_replay/execution_settings.json')}
 for row in chosen['entries']:
  files.update([row['spec']['gate'],row['spec']['proposer']])
 files.update(str(p) for p in (ROOT/'gym_pybullet_drones/assets').rglob('*') if p.is_file())
 files.update(str(p) for p in OUT.glob('AMENDMENT*.json'))
 from ..round4.landing import LandingCriteria
 save(folder/'frozen.json',dict(created_epoch=time.time(),attempt=attempt,sources=sources,artifacts={p:sha(p) for p in sorted(files)},selection_sha=sha(folder/'selection.json'),strict_criteria=LandingCriteria().dictionary(),python=sys.version,torch=torch.__version__,numpy=np.__version__,spending=spending(attempt),operational_gate=chosen['operational_gate'],assigned_mixture='Selected operational gate, uniformly sampled critical level29.5/29.6 and noise1/2 with estimated effectiveness.',acceptance='Zero losses for actual operational checkpoint in all development and prospective flights; positive lower95 operational route bootstrap in primary AND confirmation; anytime benefit and harm evidence for actual selected checkpoint; guarded latency p95 below20.833ms. All five fits still evaluated; family robustness reported separately.'))
 return verify(attempt)

def allocation(attempt):
 folder=path(attempt);dest=folder/'allocations.json'
 if dest.exists():return read(dest)
 frozen=verify(attempt);offset=(attempt-1)*1000000;op=frozen['operational_gate'];random_seed=int.from_bytes(secrets.token_bytes(8),'little')
 primary=cases(range(840000+offset,840384+offset),levels=[.295,.296])+cases(range(840000+offset,840128+offset),levels=[l for l in LEVELS if l not in [.295,.296]])
 confirm=cases(range(850000+offset,851024+offset),levels=[.295,.296]);shift=cases(range(860000+offset,860064+offset),levels=[.295,.296],conditions=['estimated'],noises=[2.],clocks=[4.,5.,7.,8.])
 assigned=assignment(range(850000+offset,851024+offset),random_seed,[op])+assignment(range(852000+offset,853024+offset),random_seed+1,[op]);extension=assignment(range(854000+offset,856048+offset),random_seed+2,[op])
 assert len(primary)==6656 and len(confirm)==8192 and len(shift)==512 and len(assigned)==2048
 save(dest,dict(primary=primary,confirmation=confirm,shifted=shift,assigned=assigned,extension=extension,random_seed=random_seed,generated_epoch=time.time(),frozen_sha=sha(folder/'frozen.json'),independence='One assigned episode per independent scenario, actual preselected deployment checkpoint. Other full-grid settings are stress tests, not independent trials.'))
 return read(dest)

def check_zero(attempt,name,entries):
 operational=read(path(attempt)/'selection.json')['operational_gate'];losses=sum(r['metrics']['losses'] for r in entries if r.get('seed',operational)==operational)
 if losses:
  close('attempt_'+str(attempt)+'_rejected','Prospective safety futility: at least one lost baseline landing. Candidate rejected; any revision needs fresh cases and next alpha allocation.',phase=name,entries=entries,losses=losses)
  save(OUT/'status.json',dict(stage='attempt_'+str(attempt)+'_rejected',status='running',reason='Continue development; do not tune this candidate on its final routes and reuse them as untouched.'));return False
 return True

def phase(attempt,label,cc):
 folder=path(attempt)/label;marker=folder/'complete.json'
 if marker.exists():return read(marker)
 verify(attempt);save(OUT/'status.json',dict(stage=f'final_{attempt}_{label}_baseline',status='running'));b=baseline(cc,folder,workers=16);selected=read(path(attempt)/'selection.json');entries=[];results=[]
 for row in selected['entries']:
  seed=row['seed'];save(OUT/'status.json',dict(stage=f'final_{attempt}_{label}_gate_{seed}',status='running'));r=run(row['spec'],b,folder/f'gate_{seed}');verify(attempt);m=compare(r,b);entries.append(dict(seed=seed,metrics=m));results.append(r);print('FROZEN_GATE',label,seed,m,flush=True)
 # All outcomes in a phase are collected before testing safety futility.
 reference=None
 if label=='primary':
  from ..round7.cache import run as old_run
  curve=dict(episodes=[r for r in b['episodes'] if r['seed']<840128+(attempt-1)*1000000]);refpath=folder/'previous_native/results.json'
  save(OUT/'status.json',dict(stage=f'final_{attempt}_reference_curve',status='running'))
  ref=read(refpath) if refpath.exists() else old_run(dict(proposer=selected['entries'][0]['spec']['proposer'],mode='native'),curve,folder/'previous_native',workers=16)
  verify(attempt);reference=compare(ref,curve)
 metrics=dict(complete=True,entries=entries,cases_per_model=len(cc),reference_curve=reference,family_statistics=paired(results,b),operational_statistics=point_statistics(results[[r['seed'] for r in selected['entries']].index(selected['operational_gate'])],b))
 save(marker,metrics);print('FROZEN_PHASE',label,metrics,flush=True);return metrics

def assigned_job(attempt,label,cc):
 folder=path(attempt)/label;marker=folder/'complete.json'
 if marker.exists():return read(marker)
 selected=read(path(attempt)/'selection.json');op=selected['operational_gate'];spec=next(x['spec'] for x in selected['entries'] if x['seed']==op);cc=[{k:r[k] for k in ['seed','level','condition','noise','clock']} for r in cc]
 save(OUT/'status.json',dict(stage=f'final_{attempt}_{label}',status='running'));b=baseline(cc,folder,workers=16);r=run(spec,b,folder/f'gate_{op}');verify(attempt);m=compare(r,b);result=dict(complete=True,metrics=m,cases=len(cc));save(marker,result);return result

def assigned_evidence(attempt,alloc,extended=False):
 chosen=read(path(attempt)/'selection.json');op=chosen['operational_gate'];bb=[];rr=[]
 for label in ['confirmation','assigned_initial']+(['assigned_extension'] if extended else []):
  bb.extend(read(path(attempt)/label/'baseline.json')['episodes']);rr.extend(read(path(attempt)/label/f'gate_{op}/results.json')['episodes'])
 b={key(x):x for x in bb};r={key(x):x for x in rr};assigned=alloc['assigned']+(alloc['extension'] if extended else [])
 value=evidence([b[key(x)]['stable_landing'] for x in assigned],[r[key(x)]['stable_landing'] for x in assigned],attempt=attempt);save(path(attempt)/('sequential_extension.json' if extended else 'sequential_initial.json'),value);print('ASSIGNED_EVIDENCE',value,flush=True);return value

def main(attempt=1):
 torch.set_num_threads(1);start=time.time();freeze(attempt);alloc=allocation(attempt)
 with awake():
  phases={}
  for label in ['primary','confirmation','shifted']:
   phases[label]=phase(attempt,label,alloc[label])
   if not check_zero(attempt,label,phases[label]['entries']):return
  extra=[r for r in alloc['assigned'] if r['seed']>=852000+(attempt-1)*1000000];initial=assigned_job(attempt,'assigned_initial',extra)
  if not check_zero(attempt,'assigned_initial',[dict(metrics=initial['metrics'])]):return
  e=assigned_evidence(attempt,alloc);extended=False
  if not e['both_statistical_conditions']:
   ext=assigned_job(attempt,'assigned_extension',alloc['extension']);extended=True
   if not check_zero(attempt,'assigned_extension',[dict(metrics=ext['metrics'])]):return
   e=assigned_evidence(attempt,alloc,True)
  bootstrap=all(phases[x]['operational_statistics']['lower95']>0 for x in ['primary','confirmation']);family_robustness=all(phases[x]['family_statistics']['lower95']>0 for x in ['primary','confirmation']) and all(row['metrics']['losses']==0 for phase in phases.values() for row in phase['entries']) and read(path(attempt)/'selection.json').get('development_family_losses',0)==0;accepted=bootstrap and e['both_statistical_conditions']
  save(path(attempt)/'decision.json',dict(accepted=bool(accepted),bootstrap_conditions=bool(bootstrap),family_robustness_supported=bool(family_robustness),sequential=e,extended=extended,phases=phases,seconds=time.time()-start,scope='Independent assigned evidence and zero-loss tests address the actual preselected operational checkpoint. Five-fit family robustness is reported separately. Simulator-specific evidence only.'))
  close('attempt_'+str(attempt)+'_evaluation','Prospective allocated evaluation closed with frozen policy and strict landing contract.',accepted=bool(accepted),seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='attempt_'+str(attempt)+'_evaluated',status='running',accepted=bool(accepted)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);a=p.parse_args();main(a.attempt)
