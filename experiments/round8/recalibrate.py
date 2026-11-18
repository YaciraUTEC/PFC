"""Align temperature fitting with the conditional probability of the mixture.
Keep the fitted weights and route roles; only held-out temperature parameters
change. The pre-correction checkpoints remain intact for the audit trail.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import OUT,read,save,sha,close
from .factor_gate import FactorGate
from .fit import load,logits,diagnostic

def main(seeds=(501,502,503)):
 assert (OUT/'h2_factor_fits_conclusion.json').exists();torch.set_num_threads(1);cal=load('calibration',OUT/'data/base_options_v1');rows=[];folder=OUT/'factor_gates/base_options_calibrated';folder.mkdir(parents=True,exist_ok=True)
 for seed in seeds:
  src=OUT/'factor_gates/base_options_v1'/f'seed_{seed}.pt'
  if (folder/f'seed_{seed}_complete.json').exists():
   cached=read(folder/f'seed_{seed}_complete.json');assert cached['prior_sha']==sha(src);rows.append(dict(seed=seed,before=cached['temperature_before'],after=cached['temperatures'],assessment=cached['assessment']));continue
  meta=read(src.with_name(f'seed_{seed}_complete.json'));m=FactorGate.load(src).cuda();start=time.time();before=m.temperature.detach().cpu().tolist();roles=meta['calibration_route_roles'];use=np.isin(cal['seed'],roles['temperature']);assess=np.isin(cal['seed'],roles['assessment']);m.temperature.fill_(1.);raw=logits(m,cal['x'][use]);y=cal['y'][use];base=np.isin(y,[0,2]);candidate=np.isin(y,[0,1]).astype(np.float64);tt=[base.astype(np.float64),candidate,candidate];masks=[np.ones(len(y),bool),base,~base];temps=[]
  for head in range(3):
   choices=[]
   for temp in [.5,.75,1.,1.5,2.,3.]:
    v=(raw[:,:,head]/temp).sigmoid()
    if head==0:p=v.mean(0)
    else:
     weights=F.logsigmoid(raw[:,:,0]/temps[0]*(1. if head==1 else -1.)).softmax(0);p=(weights*v).sum(0)
    pp=p.numpy()[masks[head]].clip(1e-7,1-1e-7);target=tt[head][masks[head]];loss=float(np.mean(-target*np.log(pp)-(1-target)*np.log1p(-pp)));choices.append((loss,temp))
   temps.append(min(choices)[1])
  m.temperature.copy_(torch.as_tensor(temps,device='cuda'));d=diagnostic(m,cal['x'][assess],cal['y'][assess]);new=dict({k:v for k,v in meta.items() if k!='path'},prior=str(src),prior_sha=sha(src),temperatures=temps,assessment=d,temperature_before=before,temperature_scope='Conditional probability of the joint ensemble, weighted by baseline success/failure in log space; original held-out route roles unchanged.',recalibration_seconds=time.time()-start);m.cpu().save(folder/f'seed_{seed}.pt',**new);save(folder/f'seed_{seed}_complete.json',new);rows.append(dict(seed=seed,before=before,after=temps,assessment=d));print('CONDITIONAL_CALIBRATION',seed,before,temps,d['joint_nll'],flush=True)
 close('h2_conditional_calibration','Conditional mixture temperatures aligned before any new-policy flight pilot; original weights and raw checkpoints preserved.',results=rows)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[501,502,503]);a=p.parse_args();main(a.seeds)
