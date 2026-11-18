"""Additional baseline-only supervision from explicitly reused round7 flights.
No new final-test route is read. Fault truth is retained only as optional targets,
never included in the deployable history representation.
"""
import time
import numpy as np
import torch
from .common import OUT,R7,read,save,sha,close
from ..round7.policy import canonical_risk
from ..round6.models import physics_observation

def main():
 torch.set_num_threads(1);start=time.time();folder=OUT/'base_forecast'
 if (OUT/'h2_baseline_forecast_features_conclusion.json').exists():return
 data=read(R7/'final/confirmation/baseline.json');rows=data['episodes'];model=canonical_risk().cuda().double().eval();collected={k:[] for k in ['x','y','seed','tick','case','true_eta','estimated_eta']};paths=[];maximum_feature_error=0.
 with torch.no_grad():
  for first in range(0,len(rows),8):
   group=rows[first:first+8];hist=[];truth=[];estimate=[];queries=[]
   for row in group:
    with np.load(row['trace_path']) as z:obs=z['observation'].copy();truth.append(z['true_eta_before'].copy());estimate.append(z['estimated_eta_before'].copy())
    hist.append(obs);eligible=np.flatnonzero(((obs[:,31:35].max(-1)>.08)|(obs[:,56]>.5))&(obs[:,57]<.5))
    query=np.unique(eligible[np.linspace(0,len(eligible)-1,min(16,len(eligible))).astype(int)]) if len(eligible) else np.array([],int);queries.append(set(map(int,query)));paths.append(row['trace_path'])
   state=None;n=len(group);lengths=[len(h) for h in hist]
   for offset in range(0,max(lengths),64):
    width=min(64,max(lengths)-offset);batch=np.zeros((n,width,88),np.float32)
    for j,h in enumerate(hist):
     end=min(len(h),offset+width)
     if end>offset:batch[j,:end-offset]=h[offset:end]
    obs=torch.as_tensor(batch,device='cuda',dtype=torch.float64);clean=physics_observation(obs);z,state=model.encoder(clean,state);features=torch.cat([clean,z],-1).float().cpu().numpy()
    for j,row in enumerate(group):
     for tick in sorted(t for t in queries[j] if offset<=t<offset+width):
      collected['x'].append(features[j,tick-offset]);collected['y'].append(int(row['stable_landing']));collected['seed'].append(row['seed']);collected['tick'].append(tick);collected['case'].append(first+j);collected['true_eta'].append(truth[j][tick]);collected['estimated_eta'].append(estimate[j][tick])
   if first%128==0:
    print('BASE_FORECAST_FEATURES',first,len(rows),len(collected['y']),round(time.time()-start),flush=True);save(OUT/'base_forecast_progress.json',dict(cases=first+len(group),total=len(rows),query_states=len(collected['y']),seconds=time.time()-start))
 allrows={k:np.asarray(v,np.float32 if k in ['x','true_eta','estimated_eta'] else np.int64) for k,v in collected.items()};seed=allrows['seed'];folder.mkdir(parents=True,exist_ok=True)
 for split,mask in [('train',seed<750224),('calibration',seed>=750224)]:np.savez_compressed(folder/(split+'.npz'),**{k:v[mask] for k,v in allrows.items()})
 save(folder/'sources.json',dict(paths=paths,source=str(R7/'final/confirmation/baseline.json'),source_sha=sha(R7/'final/confirmation/baseline.json'),encoder='Frozen canonical round6 GTrXL201',truth_scope='true_eta is optional supervised target only; x contains observable physics and its GTrXL encoding, never fault truth, route ID, or outcome.',route_split='750000..750223 train;750224..750319 calibration; all were old inspected tests and now development. New840/850/860 tests untouched.'))
 close('h2_baseline_forecast_features','Additional base-only prognosis features extracted without new simulation; route split preserved and fault truth excluded from decision input.',cases=len(rows),states=len(seed),train_routes=len(np.unique(seed[seed<750224])),calibration_routes=len(np.unique(seed[seed>=750224])),seconds=time.time()-start)
if __name__=='__main__':main()
