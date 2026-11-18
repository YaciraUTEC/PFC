"""Keep prognosis strictly before first contact and validate feature streaming."""
import time
import numpy as np
import torch
from .common import OUT,read,save,close
from ..round7.policy import canonical_risk,risk_features

def main():
 torch.set_num_threads(1);start=time.time();folder=OUT/'base_forecast';meta=read(folder/'sources.json');limits=[]
 for path in meta['paths']:
  with np.load(path) as z:obs=z['observation'];first=np.flatnonzero(obs[:,57]>=.5);limits.append(int(first[0]) if len(first) else len(obs))
 limits=np.asarray(limits);summary={};kept={};dest=folder/'precontact';dest.mkdir(exist_ok=True)
 for split in ['train','calibration']:
  with np.load(folder/(split+'.npz')) as z:data={k:z[k].copy() for k in z.files}
  mask=data['tick']<limits[data['case']];new={k:v[mask] for k,v in data.items()};np.savez_compressed(dest/(split+'.npz'),**new);kept[split]=new;summary[split]=dict(before=len(mask),retained=int(mask.sum()),postcontact_removed=int((~mask).sum()),routes=len(np.unique(new['seed'])))
 data=kept['train'];case=next(i for i,path in enumerate(meta['paths']) if len(np.flatnonzero((data['case']==i)&(data['estimated_eta'].min(-1)<.98)))>2);use=np.flatnonzero(data['case']==case);wanted={int(data['tick'][i]):i for i in use}
 with np.load(meta['paths'][case]) as z:history=z['observation'].copy()
 model=canonical_risk().double().eval();state=None;largest=0.
 with torch.no_grad():
  for tick in range(max(wanted)+1):
   context,state=risk_features(model,torch.as_tensor(history[tick:tick+1],dtype=torch.float64),state)
   if tick in wanted:largest=max(largest,float(np.abs(context[0].float().numpy()-data['x'][wanted[tick]]).max()))
 save(dest/'feature_streaming_comparison.json',dict(case=case,queries=len(wanted),maximum_float32_context_difference=largest,source='CPU single-step vs GPU chunk64 extraction'))
 assert largest<1e-5,('Prognosis feature streaming differs',largest)
 save(dest/'sources.json',dict(**meta,first_contact_limits=limits,precontact_only=True,filter_summary=summary,feature_validation_maximum_difference=largest))
 close('h2_baseline_forecast_precontact','Post-contact states removed from additional prognosis data; CPU runtime feature equivalence verified.',summary=summary,feature_maximum_difference=largest,seconds=time.time()-start)
if __name__=='__main__':main()
