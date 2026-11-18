"""Validate compaction on every actual first decision, before formal test allocation."""
import time,hashlib
from pathlib import Path
import numpy as np
import torch
from .common import OUT,read,save,sha,close
from .loading_compact import Policy
from ..round6.models import physics_observation
from ..round7.evaluation import key
from ..round7.power import awake
MARGIN=1e-7

def main():
 torch.set_num_threads(1);start=time.time();source=read(OUT/'h8_checkpoint_selection_conclusion.json');base=read(Path(source['development_folder'])/'baseline.json');entries=source['entries'];dense=[Policy.from_spec(x['spec']).cuda().double() for x in entries];compact=[Policy.from_spec(dict(x['spec'],compact_options=True)).cuda().double() for x in entries];actor=dense[0].proposer;assert actor.method=='flashsac';count=len(base['episodes']);before=np.full((5,count),-1,np.int64);after=before.copy();expected=before.copy();ambiguous=np.zeros((5,count),bool);errors=torch.zeros(5,device='cuda',dtype=torch.float64);feature_error=errors.clone();inputs=[];refs=[];metrics=['lower_gain','decision_conditional_harm','base_success','preservation','mean_rescue','mean_harm']
 for mi,row in enumerate(entries):
  p=Path(source['development_folder'])/f'seed_{row["seed"]}/results.json';r=read(p);lookup={key(x):x for x in r['episodes']};assert {key(x) for x in base['episodes']}==set(lookup);refs.append(dict(path=str(p),sha256=sha(p)))
  for j,case in enumerate(base['episodes']):
   tick=lookup[key(case)].get('policy_summary',{}).get('option_start_seconds');expected[mi,j]=-1 if tick is None else int(round(tick*48))
 with awake(),torch.no_grad():
  for first in range(0,count,16):
   rows=base['episodes'][first:first+16];hist=[]
   for row in rows:
    with np.load(row['trace_path']) as z:h=z['observation'].copy()
    hist.append(h);inputs.append(dict(case=list(key(row)),observation_sha256=hashlib.sha256(h.tobytes()).hexdigest(),frames=len(h)))
   lengths=np.asarray([len(x) for x in hist]);n=len(rows);memory=None
   for tick in range(0,int(lengths.max()),64):
    width=min(64,int(lengths.max())-tick);data=np.zeros((n,width,88),np.float32);times=np.arange(tick,tick+width);valid=times[None]<lengths[:,None]
    for j,h in enumerate(hist):
     end=min(len(h),tick+width)
     if end>tick:data[j,:end-tick]=h[tick:end]
    obs=torch.as_tensor(data,device='cuda',dtype=torch.float64);clean=physics_observation(obs);z,memory=actor.prior.encoder(clean,memory);context=torch.cat([clean,z],-1).reshape(n*width,-1);flat=obs.reshape(-1,88);raw,_=actor.proposal(context,deterministic=True,noise=None);native=actor.select(context,raw).abs().amax(-1)>.002
    for mi,(a,b) in enumerate(zip(dense,compact)):
     aa,ab,ax=a.build_options(flat,context,raw,native);ba,bb,bx=b.build_options(flat,context,raw,native);al,am=a.assess(ax,aa);bl,bm=b.assess(bx,ba);ai=a.option_ids.index(5);bi=b.option_ids.index(5);feature_error[mi]=torch.maximum(feature_error[mi],torch.stack([(ax[:,ai]-bx[:,bi]).abs().max(),(aa[:,ai]-ba[:,bi]).abs().max(),(ab[:,ai]-bb[:,bi]).abs().max()]).max());errors[mi]=torch.maximum(errors[mi],torch.stack([(am[k][:,ai]-bm[k][:,bi]).abs().max() for k in metrics]).max());eligible=native&(raw.abs().amax(-1)*.5>.002)&(flat[:,57]<.5)&(flat[:,31:35].amax(-1)*.25>.02)
     allowed_a=(al[:,ai]&(flat[:,57]<.5)).reshape(n,width).cpu().numpy()&valid;allowed_b=(bl[:,bi]&(flat[:,57]<.5)).reshape(n,width).cpu().numpy()&valid;assert np.array_equal(allowed_a,allowed_b),'Dense/compact authorization changed'
     near=(((am['lower_gain'][:,ai]-.01).abs()<MARGIN)|((am['decision_conditional_harm'][:,ai]-.02).abs()<MARGIN))&eligible;near=near.reshape(n,width).cpu().numpy()&valid
     for j in range(n):
      pos=first+j;stop=expected[mi,pos];prefix=times<=stop if stop>=0 else np.ones(width,bool);ambiguous[mi,pos]|=bool((near[j]&prefix).any())
      for array,on in [(before,allowed_a),(after,allowed_b)]:
       hits=np.flatnonzero(on[j])
       if array[mi,pos]<0 and len(hits):array[mi,pos]=tick+int(hits[0])
   if (first//16)%32==0:
    progress=dict(cases=min(first+16,count),total=count,seconds=time.time()-start);save(OUT/'execution_optimization/compact_scan_progress.json',progress);print('COMPACT_SCAN',progress,flush=True)
 e=errors.cpu().tolist();f=feature_error.cpu().tolist();assert max(f)==0.;assert max(e)<1e-10;assert np.array_equal(before,after);assert np.array_equal(before,expected),'Reconstructed dense first decisions do not match actual prior CPU flights'
 folder=OUT/'execution_optimization';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'compact_first_decisions.npz',dense=before,compact=after,actual_original=expected,ambiguous=ambiguous);save(folder/'compact_scan_inputs.json',inputs);close('h10_compact_first_decisions','Every dense and compact first authorization matches the actual original policy on all8064development cases for all5gates; actor/feedback/budget are unchanged. Near-threshold cases are flagged for direct CPU physical replay.',passed=True,cases_per_model=count,models=5,first_decisions_compared=count*5,maximum_selected_feature_difference=max(f),maximum_selected_forecast_difference=max(e),source_baseline_sha=sha(Path(source['development_folder'])/'baseline.json'),original_references=refs,source_sha256=sha(__file__),loader_sha256=sha(Path(__file__).with_name('loading_compact.py')),arrays_sha256=sha(folder/'compact_first_decisions.npz'),inputs_sha256=sha(folder/'compact_scan_inputs.json'),near_threshold_cases={str(x['seed']):int(ambiguous[i].sum()) for i,x in enumerate(entries)},seconds=time.time()-start)
if __name__=='__main__':main()
