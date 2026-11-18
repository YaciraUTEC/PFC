"""Conditional terminal likelihoods, route-separated tuning and calibration."""
import argparse,time,math,copy
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import OUT,R7,read,save,sha,close
from .factor_gate import FactorGate

def load(split,folder=None):
 folder=Path(folder) if folder else R7/'gate_data'
 xx=[];yy=[];ss=[];groups=[];first=[];offset=0
 paths=sorted((folder/split).glob('*.npz'))
 for path in paths:
  with np.load(path) as z:x=z['x'].copy();y=z['y'].copy();seeds=z['seed'].copy()
  records=read(path.with_suffix('.json'))['records'];assert len(records)==len(y)
  seen={}
  for i,r in enumerate(records):
   key=(r['seed'],r['level'],r['tick'],r['future'])
   b=int(y[i] in [0,2])
   if key not in seen:seen[key]=(i,b);first.append(offset+i)
   else:
    j,target=seen[key];assert b==target,'Contradictory baseline counterfactual';np.testing.assert_array_equal(x[i,:184],x[j,:184])
   groups.append(key)
  xx.append(x);yy.append(y);ss.append(seeds);offset+=len(y)
 return dict(x=np.concatenate(xx),y=np.concatenate(yy),seed=np.concatenate(ss),base_rows=np.array(first,np.int64),paths=paths,groups=groups)

def logits(model,x,batch=1024):
 result=[]
 with torch.no_grad():
  for first in range(0,len(x),batch):result.append(model(torch.as_tensor(x[first:first+batch],device='cuda',dtype=torch.float32)).cpu())
 return torch.cat(result,1)

def diagnostic(model,x,y):
 raw=logits(model,x);parts=raw.sigmoid().mean(0).numpy();p,s,r=parts.T
 q=np.stack([p*s,(1-p)*r,p*(1-s),(1-p)*(1-r)],-1)
 # The joint ensemble is an average of per-member factorizations.
 z=raw.sigmoid().numpy();pb,ss,rr=np.moveaxis(z,-1,0);q=np.stack([pb*ss,(1-pb)*rr,pb*(1-ss),(1-pb)*(1-rr)],-1).mean(0)
 p=q[:,0]+q[:,2];base_weights=torch.nn.functional.logsigmoid(raw[:,:,0]).softmax(0).numpy();fail_weights=torch.nn.functional.logsigmoid(-raw[:,:,0]).softmax(0).numpy();s=(base_weights*ss).sum(0);r=(fail_weights*rr).sum(0)
 b=np.isin(y,[0,2]);c=np.isin(y,[0,1]);metrics={}
 for name,estimate,target,mask in [('baseline',p,b,np.ones(len(y),bool)),('preservation',s,c,b),('rescue',r,c,~b)]:
  target=target[mask];estimate=estimate[mask].clip(1e-7,1-1e-7)
  metrics[name]=dict(samples=int(mask.sum()),positive=int(target.sum()),mean_prediction=float(estimate.mean()),brier=float(np.mean((estimate-target)**2)),nll=float(np.mean(-target.astype(float)*np.log(estimate)-(~target)*np.log1p(-estimate))))
 return dict(samples=len(y),labels=dict(Counter(map(int,y))),joint_nll=float(-np.log(q[np.arange(len(y)),y].clip(1e-8)).mean()),joint_brier=float(np.mean(np.sum((q-np.eye(4)[y])**2,-1))),components=metrics,scope='Marginal prediction diagnostics on held-out route groups, not a safety certificate.')

def main(seed=501,updates=3200,folder=None,tag='initial',route_bootstrap=True):
 start=time.time();torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 train=load('train',folder);cal=load('calibration',folder);assert set(train['seed']).isdisjoint(set(cal['seed']))
 x,y,seeds=train['x'],train['y'],train['seed'];vx,vy,vs=cal['x'],cal['y'],cal['seed'];rng=np.random.default_rng(seed)
 ids=np.unique(vs);split_rng=np.random.default_rng(17028);split_rng.shuffle(ids);n=len(ids);tune=np.isin(vs,ids[:n//2]);temp=np.isin(vs,ids[n//2:3*n//4]);assess=~(tune|temp)
 assert all(mask.any() for mask in [tune,temp,assess])
 model=FactorGate(dim=x.shape[-1]).cuda();model.center.copy_(torch.as_tensor(x.mean(0),device='cuda'));model.scale.copy_(torch.as_tensor(np.maximum(x.std(0),.05),device='cuda'))
 optim=[torch.optim.AdamW(list(b.parameters())+list(o.parameters()),lr=2e-4,weight_decay=1e-4) for b,o in zip(model.base_members,model.option_members)]
 routes=np.unique(seeds);base_set=set(train['base_rows']);pools=[];base_pools=[]
 for _ in optim:
  boot=rng.choice(routes,len(routes),replace=True) if route_bootstrap else routes;pools.append(np.concatenate([np.flatnonzero(seeds==s) for s in boot]));base_pools.append(np.concatenate([np.array([i for i in np.flatnonzero(seeds==s) if i in base_set],int) for s in boot]))
 dest=OUT/'factor_gates'/tag;dest.mkdir(parents=True,exist_ok=True);best=(float('inf'),None,0);progress=[]
 for it in range(updates):
  lr=2e-4*min(1.,(it+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*it/updates)));losses=[]
  for j,opt in enumerate(optim):
   ii=rng.choice(pools[j],512);bb=rng.choice(base_pools[j],512);tx=torch.as_tensor(x[ii],device='cuda');bx=torch.as_tensor(x[bb,:184],device='cuda');ty=torch.as_tensor(y[ii],device='cuda');by=torch.as_tensor(np.isin(y[bb],[0,2]),device='cuda',dtype=torch.float32)
   z=((tx-model.center)/model.scale).clamp(-12,12);bz=((bx-model.center[:184])/model.scale[:184]).clamp(-12,12)
   bl=model.base_members[j](bz)[:,0];ol=model.option_members[j](z);base=(ty==0)|(ty==2);candidate=((ty==0)|(ty==1)).float()
   keep=F.binary_cross_entropy_with_logits(ol[:,0],candidate,reduction='none');rescue=F.binary_cross_entropy_with_logits(ol[:,1],candidate,reduction='none')
   loss=F.binary_cross_entropy_with_logits(bl,by)+(keep*base).sum()/base.sum().clamp_min(1)+(rescue*~base).sum()/(~base).sum().clamp_min(1)
   for group in opt.param_groups:group['lr']=lr
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(model.base_members[j].parameters())+list(model.option_members[j].parameters()),1.);opt.step();losses.append(float(loss))
  if (it+1)%400==0 or it==updates-1:
   metric=diagnostic(model,vx[tune],vy[tune]);value=metric['joint_nll'];row=dict(update=it+1,loss=float(np.mean(losses)),tune_joint_nll=value,seconds=time.time()-start);progress.append(row);save(dest/f'seed_{seed}_progress.json',row);print('FACTOR_FIT',seed,row,flush=True)
   if value<best[0]:best=(value,copy.deepcopy(model.state_dict()),it+1)
 model.load_state_dict(best[1]);raw=logits(model,vx[temp]);truth=[np.isin(vy[temp],[0,2]),np.isin(vy[temp],[0,1]),np.isin(vy[temp],[0,1])];masks=[np.ones(temp.sum(),bool),truth[0],~truth[0]];temperatures=[]
 for head in range(3):
  besttemp=(float('inf'),1.)
  for t in [.5,.75,1.,1.5,2.,3.]:
   pred=torch.sigmoid(raw[:,:,head]/t)
   if head==0:pp=pred.mean(0)
   else:
    weights=F.logsigmoid(raw[:,:,0]/temperatures[0]*(1. if head==1 else -1.)).softmax(0);pp=(weights*pred).sum(0)
   pp=pp.numpy()[masks[head]].clip(1e-7,1-1e-7);tt=truth[head][masks[head]];nll=float(np.mean(-tt.astype(float)*np.log(pp)-(~tt)*np.log1p(-pp)))
   if nll<besttemp[0]:besttemp=(nll,t)
  temperatures.append(besttemp[1])
 model.temperature.copy_(torch.as_tensor(temperatures,device='cuda'));metrics=diagnostic(model,vx[assess],vy[assess]);path=dest/f'seed_{seed}.pt'
 source={str(path):sha(path) for path in train['paths']+cal['paths']};meta=dict(seed=seed,updates=updates,selected_update=best[2],training_samples=len(y),baseline_unique_future_queries=len(train['base_rows']),train_routes=len(routes),calibration_route_roles=dict(tune=ids[:n//2].tolist(),temperature=ids[n//2:3*n//4].tolist(),assessment=ids[3*n//4:].tolist()),temperatures=temperatures,assessment=metrics,seconds=time.time()-start,source_fingerprint=source,baseline_head_independent_of_option=True,route_bootstrap=route_bootstrap,member_training_unique_routes=[len(np.unique(seeds[p])) for p in pools])
 model.cpu().save(path,**meta);save(dest/f'seed_{seed}_complete.json',dict(path=str(path),**meta,progress=progress));print('FACTOR_FIT_COMPLETE',seed,round(time.time()-start,2),metrics,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=501);p.add_argument('--updates',type=int,default=3200);p.add_argument('--folder');p.add_argument('--tag',default='initial');p.add_argument('--full-support',action='store_true');a=p.parse_args();main(a.seed,a.updates,a.folder,a.tag,not a.full_support)
