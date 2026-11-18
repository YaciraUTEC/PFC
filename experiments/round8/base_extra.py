"""Ablation: improve only the base-success prognosis, keep option heads fixed."""
import argparse,time,math,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import OUT,read,save,close,sha
from .factor_gate import FactorGate
from .fit import load

def main(seed=501,updates=1600):
 assert (OUT/'h2_flight_pilot_conclusion.json').exists(),'Close the original new-data pilot first'
 torch.set_num_threads(1);torch.manual_seed(seed);start=time.time();rng=np.random.default_rng(seed+91);prior=OUT/'factor_gates/base_options_calibrated'/f'seed_{seed}.pt';model=FactorGate.load(prior).cuda();old_option={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('option_members')};original_temperature=float(model.temperature[0]);model.temperature[0]=1.
 train=load('train',OUT/'data/base_options_v1');cal=load('calibration',OUT/'data/base_options_v1');bs=train['base_rows'];vb=cal['base_rows']
 with np.load(OUT/'base_forecast/precontact/train.npz') as z:extra={k:z[k].copy() for k in ['x','y','seed']}
 with np.load(OUT/'base_forecast/precontact/calibration.npz') as z:valid={k:z[k].copy() for k in ['x','y','seed']}
 cx=train['x'][bs,:184];cy=np.isin(train['y'][bs],[0,2]).astype(np.float32);cs=train['seed'][bs];vx=cal['x'][vb,:184];vy=np.isin(cal['y'][vb],[0,2]).astype(np.float32);vs=cal['seed'][vb]
 assert set(extra['seed']).isdisjoint(set(valid['seed'])) and set(cs).isdisjoint(set(vs));roles=read(OUT/'factor_gates/base_options_v1'/f'seed_{seed}_complete.json')['calibration_route_roles'];er=np.unique(valid['seed']);er_rng=np.random.default_rng(70719);er_rng.shuffle(er);n=len(er);masks={role:(np.isin(vs,roles[role]),np.isin(valid['seed'],er[lo:hi])) for role,lo,hi in [('tune',0,n//2),('temperature',n//2,3*n//4),('assessment',3*n//4,n)]}
 opts=[torch.optim.AdamW(member.parameters(),lr=1e-4,weight_decay=1e-4) for member in model.base_members];pools=[]
 for _ in opts:
  a=np.unique(cs);b=np.unique(extra['seed']);pa=np.concatenate([np.flatnonzero(cs==s) for s in rng.choice(a,len(a),replace=True)]);pb=np.concatenate([np.flatnonzero(extra['seed']==s) for s in rng.choice(b,len(b),replace=True)]);pools.append((pa,pb))
 def probabilities(x,temp=1.):
  values=[]
  with torch.no_grad():
   for first in range(0,len(x),1024):
    z=((torch.as_tensor(x[first:first+1024],device='cuda',dtype=torch.float32)-model.center[:184])/model.scale[:184]).clamp(-12,12);values.append(torch.stack([b(z)[:,0]/temp for b in model.base_members]).sigmoid().mean(0).cpu().numpy())
  return np.concatenate(values)
 def metric(role,temp=1.):
  a,b=masks[role];values=[]
  for x,y,mask in [(vx,vy,a),(valid['x'],valid['y'],b)]:
   p=probabilities(x[mask],temp).clip(1e-7,1-1e-7);t=y[mask].astype(float);values.append(dict(n=len(t),nll=float(np.mean(-t*np.log(p)-(1-t)*np.log1p(-p))),brier=float(np.mean((p-t)**2))))
  return dict(mean_nll=float(np.mean([v['nll'] for v in values])),counterfactual_base=values[0],additional_flights=values[1])
 before=metric('assessment',original_temperature);best=(float('inf'),None,0);logs=[]
 for it in range(updates):
  lr=1e-4*min(1.,(it+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*it/updates)));loss=[]
  for j,opt in enumerate(opts):
   aa=rng.choice(pools[j][0],256);bb=rng.choice(pools[j][1],256);xx=np.concatenate([cx[aa],extra['x'][bb]]);yy=np.r_[cy[aa],extra['y'][bb]].astype(np.float32);tx=torch.as_tensor(xx,device='cuda');ty=torch.as_tensor(yy,device='cuda');z=((tx-model.center[:184])/model.scale[:184]).clamp(-12,12);value=F.binary_cross_entropy_with_logits(model.base_members[j](z)[:,0],ty)
   for g in opt.param_groups:g['lr']=lr
   opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.base_members[j].parameters(),1.);opt.step();loss.append(float(value))
  if (it+1)%400==0:
   result=metric('tune');row=dict(update=it+1,loss=float(np.mean(loss)),tune=result,seconds=time.time()-start);logs.append(row);print('BASE_ONLY_FIT',seed,row,flush=True)
   if result['mean_nll']<best[0]:best=(result['mean_nll'],copy.deepcopy(model.base_members.state_dict()),it+1)
 model.base_members.load_state_dict(best[1]);temperature=min([.5,.75,1.,1.5,2.,3.],key=lambda t:metric('temperature',t)['mean_nll']);model.temperature[0]=temperature;after=metric('assessment',temperature)
 for name,value in old_option.items():assert torch.equal(model.state_dict()[name].cpu(),value),'Option head unexpectedly changed'
 folder=OUT/'factor_gates/base_forecast_augmented';folder.mkdir(parents=True,exist_ok=True);meta=dict(seed=seed,prior=str(prior),prior_sha=sha(prior),updates=updates,selected_update=best[2],base_temperature=temperature,before=before,after=after,option_weights_unchanged=True,additional_train_routes=len(np.unique(extra['seed'])),additional_calibration_routes=len(er),seconds=time.time()-start)
 model.cpu().save(folder/f'seed_{seed}.pt',**meta);save(folder/f'seed_{seed}_complete.json',dict(**meta,progress=logs));print('BASE_ONLY_FIT_COMPLETE',seed,before,after,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=501);p.add_argument('--updates',type=int,default=1600);a=p.parse_args();main(a.seed,a.updates)
