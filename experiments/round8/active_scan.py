"""Select counterfactual queries from actual first-authorization states on base histories.
No baseline terminal outcome is read to choose cases or ticks. This is development.
"""
import time,hashlib
from pathlib import Path
import numpy as np
import torch
from .common import OUT,read,save,sha,close
from .policy import PreservePolicy
from ..round6.models import physics_observation
from ..round7.evaluation import key

def choose_ticks(native,first,maximum=12):
 native=np.asarray(native,bool);valid=np.flatnonzero(native);first=sorted(set(int(t) for t in first if t is not None))
 if not len(valid):return []
 # Earliest and latest first decision cover observed stopping-state variation.
 candidates=([first[0],first[-1]]+first[1:-1] if first else [])+[int(valid[0])]
 if first:
  nearby=valid[(valid>=first[0]+8)&(valid<=first[0]+24)]
  if len(nearby):candidates.append(int(nearby[len(nearby)//2]))
 result=[]
 for t in candidates:
  if t in valid and t not in result:result.append(t)
  if len(result)>=maximum:break
 return sorted(result)

def partition(seed):return 'calibration' if hashlib.sha256(('gtrxl201-active-v1:'+str(seed)).encode()).digest()[0]%4==0 else 'train'

def main():
 torch.set_num_threads(1);start=time.time();folder=OUT/'data/active_v1';dest=folder/'allocation.json'
 if dest.exists():return read(dest)
 base=read(OUT/'development/fixed_expert201/baseline.json');template=read(OUT/'h5_conservative_validation_conclusion.json')['entries'];specs=[dict(x['spec'],gate=str(OUT/f'factor_gates/{tag}/seed_{x["seed"]}.pt')) for tag in ['base_options_calibrated','full_support_v1'] for x in template]
 models=[PreservePolicy.from_spec(s).cuda().double() for s in specs];source=models[0].proposer;assert source.method=='flashsac';query=[];native_cases=0;model_files={str(p):sha(p) for s in specs for p in [s['gate'],s['proposer']]}
 for first_case in range(0,len(base['episodes']),16):
  rows=base['episodes'][first_case:first_case+16];hist=[]
  for row in rows:
   with np.load(row['trace_path']) as z:hist.append(z['observation'].copy())
  lengths=np.asarray([len(x) for x in hist]);n=len(rows);pm=None;native_hist=[np.zeros(len(x),bool) for x in hist];firsts=[[None]*len(models) for _ in rows]
  with torch.no_grad():
   for first_tick in range(0,int(lengths.max()),64):
    width=min(64,int(lengths.max())-first_tick);data=np.zeros((n,width,88),np.float32);valid=np.arange(first_tick,first_tick+width)[None]<lengths[:,None]
    for j,h in enumerate(hist):
     end=min(len(h),first_tick+width)
     if end>first_tick:data[j,:end-first_tick]=h[first_tick:end]
    obs=torch.as_tensor(data,device='cuda',dtype=torch.float64);clean=physics_observation(obs);z,pm=source.prior.encoder(clean,pm);features=torch.cat([clean,z],-1).reshape(n*width,-1);flat=obs.reshape(-1,88);raw,_=source.proposal(features,deterministic=True,noise=None);native=source.select(features,raw).abs().amax(-1)>.002
    eligible=native&(raw.abs().amax(-1)*.5>.002)&(flat[:,57]<.5);native_np=eligible.reshape(n,width).cpu().numpy()&valid
    aa,budgets,x=models[0].build_options(flat,features,raw,native)
    for j in range(n):
     end=min(width,len(native_hist[j])-first_tick)
     if end>0:native_hist[j][first_tick:first_tick+end]=native_np[j,:end]
    for mi,model in enumerate(models):
     allowed,_=model.assess(x,aa);on=(allowed[:,1]&eligible).reshape(n,width).cpu().numpy()&valid
     for j in range(n):
      tt=np.flatnonzero(on[j])
      if firsts[j][mi] is None and len(tt):firsts[j][mi]=first_tick+int(tt[0])
  for j,row in enumerate(rows):
   ticks=choose_ticks(native_hist[j],firsts[j]);native_cases+=bool(ticks)
   if ticks:
    case={k:row[k] for k in ['seed','level','condition','noise','clock']};tag=hashlib.sha256(repr(key(row)).encode()).hexdigest()[:16];query.append(dict(case=case,ticks=ticks,first_authorizations=firsts[j],split=partition(row['seed']),tag=tag,baseline_trace_path=row['trace_path'],baseline_reference=dict(outcome=row['outcome'],episode_steps=row['episode_steps'],terminal_position=row['terminal_position'])))
  if (first_case//16)%16==0:save(folder/'scan_progress.json',dict(histories_scanned=min(first_case+16,len(base['episodes'])),queries=len(query),states=sum(len(x['ticks']) for x in query),seconds=time.time()-start))
 del models;torch.cuda.empty_cache();result=dict(entries=query,models=specs,model_sha256=model_files,base_cases=len(base['episodes']),queried_cases=len(query),query_states=sum(len(x['ticks']) for x in query),independent_routes=len({x['case']['seed'] for x in query}),routes_per_split={split:len({x['case']['seed'] for x in query if x['split']==split}) for split in ['train','calibration']},futures_per_state=4,option=1,selection='No baseline terminal outcome used; native onset and earliest/latest first authorizations of five bootstrap and five full-support gates, all distinct first-authorization ticks retained, maximum12queries per case including native onset and one nearby exploration.',encoder='Canonical risk encoder is exactly proposal201 encoder; single shared forward computes both representations.',source_sha256=sha(__file__),seconds=time.time()-start)
 save(dest,result);close('h6_active_query_scan','Prospective development query states chosen from actual first-decision distributions without using terminal labels to sample.',cases=len(query),states=result['query_states'],routes=result['independent_routes'],routes_per_split=result['routes_per_split'],seconds=time.time()-start);return result
if __name__=='__main__':main()
