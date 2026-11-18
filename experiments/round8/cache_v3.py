"""Conservative OFF identity reuse for the complete one-option policy.
CPU and GPU checks are compared before use. Uncertain cases are fully simulated.
"""
import time
from pathlib import Path
import torch
import numpy as np
from ..round6.models import physics_observation
from ..round6.offpolicy import OffPolicy
from ..round7.evaluation import key
from .evaluation import evaluate
from ..round7.fast_gtrxl import install
from ..round7.cache import baseline
from .loading import Policy as PreservePolicy
from .common import save,read,sha
MARGIN=1e-7

def certify(model,histories):
 lengths=np.asarray([len(h) for h in histories]);n=len(histories);parameter=next(model.parameters());pm=rm=None;off=np.ones(n,bool)
 with torch.no_grad():
  for first in range(0,int(lengths.max()),64):
   width=min(64,int(lengths.max())-first);data=np.zeros((n,width,88),np.float32);valid=np.arange(first,first+width)[None]<lengths[:,None]
   for j,h in enumerate(histories):
    end=min(len(h),first+width)
    if end>first:data[j,:end-first]=h[first:end]
   obs=torch.as_tensor(data,dtype=parameter.dtype,device=parameter.device);clean=physics_observation(obs);flat=obs.reshape(-1,88);source=model.proposer
   if isinstance(source,OffPolicy):
    z,pm=source.prior.encoder(clean,pm);features=torch.cat([clean,z],-1).reshape(n*width,-1);raw,_=source.proposal(features,deterministic=True,noise=None)
    native=(source.benefit_q(features,raw)-source.benefit_q(features,torch.zeros_like(raw))).amin(0)>=source.threshold-MARGIN if source.gate else torch.ones(len(flat),dtype=torch.bool,device=flat.device)
   else:
    mu,_,pm=source(obs,pm);mu=mu.reshape(n*width,-1);raw=source.physical_delta(mu);native=mu[:,5]>=-MARGIN
   delta=torch.where(native[:,None],raw,torch.zeros_like(raw)) if model.native else raw
   rz,rm=model.risk.encoder(clean,rm);context=torch.cat([clean,rz],-1).reshape(n*width,-1)
   aa,budgets,x=model.build_options(flat,context,raw,native) if hasattr(model,'build_options') else model.option_features(flat,context,delta)
   allowed,metrics=model.assess(x,aa,margin=MARGIN)
   bad=~torch.isfinite(raw).all(-1)|~torch.isfinite(metrics['lower_gain']).all(-1)|~torch.isfinite(metrics['conditional_harm']).all(-1)
   possible=((allowed.any(-1)|bad)&(flat[:,57]<.5+MARGIN)).reshape(n,width).cpu().numpy()&valid;off &=~possible.any(1)
 return off

def run(spec,base,folder,workers=16,proof_device='cuda',batch=16):
 folder=Path(folder);path=folder/'results.json'
 source_dir=Path(__file__).parent;fingerprint={name:sha(source_dir/name) for name in ['policy.py','policy_v2.py','factor_gate.py','options.py','cache.py','evaluation.py','policy_v3.py','loading.py','cache_v3.py','execution.py']};fingerprint.update(gate=sha(spec['gate']),proposer=sha(spec['proposer']))
 if path.exists():
  r=read(path);assert r['spec']==spec;assert r.get('fingerprint')==fingerprint,'Completed inference inputs changed; use a new revision folder';assert {key(x) for x in r['episodes']}=={key(x) for x in base['episodes']};return r
 start=time.time();model=PreservePolicy.from_spec(spec).to(proof_device).double();torch.set_num_threads(1);rows=[];pending=[];proof=[]
 for first in range(0,len(base['episodes']),batch):
  group=base['episodes'][first:first+batch];histories=[]
  for row in group:
   with np.load(row['trace_path']) as z:histories.append(z['observation'].copy())
  off=certify(model,histories)
  for row,flag in zip(group,off):
   if flag:rows.append(dict(row,baseline_reused_verified_off=True,fresh_simulation=False,policy_summary=dict(options_started=0,actual_motor_intervention_steps=0,first_actual_intervention_seconds=None,option_start_seconds=None,scope='Complete-policy OFF identity verified on entire base history')));proof.append(list(key(row)))
   else:pending.append({k:row[k] for k in ['seed','level','condition','noise','clock']})
 del model
 if torch.cuda.is_available():torch.cuda.empty_cache()
 if pending:
  model=install(PreservePolicy.from_spec(spec));fresh=evaluate(model,pending,workers=workers,capture_directory=folder/'changed_outcome_traces',capture_baseline=base)
  rows.extend(dict(r,baseline_reused_verified_off=False,fresh_simulation=True) for r in fresh['episodes'])
 assert len(rows)==len(base['episodes']);result=dict(episodes=rows,spec=spec,fingerprint=fingerprint,logical_cases=len(rows),fresh_simulations=len(pending),verified_off_reuse=len(proof),seconds=time.time()-start,proof_device=proof_device,workers=workers,scope='Only numerically conservative all-history OFF cases reuse the base; no physical safety certificate for nonzero intervention.')
 save(folder/'off_identity_proof.json',proof);save(path,result);print('MODEL_JOB',folder.name,len(pending),len(proof),round(result['seconds'],2),flush=True);return result
