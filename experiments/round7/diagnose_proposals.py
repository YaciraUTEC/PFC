"""Diagnose actual proposal learning before attributing failure to data quantity."""
import time
from collections import defaultdict
import numpy as np
import torch
from .common import OUT,save,read,old_model
from .episode import load_policy
from .proposal_validation import samples
from ..round4.models import detach


def evaluate(path,data):
    m=load_policy(path);routes=defaultdict(list);zeros=defaultdict(list);window=defaultdict(list);full_sample=[];amp=[];pred_norm=[];target_norm=[];lower_saturated=[]
    with torch.no_grad():
        for r,s,a,valid in data:
            state=None;pred=[];mus=[]
            for k in range(0,len(s),128):
                x=torch.as_tensor(s[k:k+128][None],dtype=torch.float64);hidden,state=m.encoder(x,state);mu=m.actor(hidden)[0];mus.append(mu.numpy());cc=mu[:,:5].clamp(-5,5).tanh()
                pred.append((cc[:,:4]*(cc[:,4:5]+1)/2).numpy());state=detach(state)
            pred=np.concatenate(pred);mu=np.concatenate(mus);actual=a[:,:4]*(a[:,4:5]+1)/2;use=valid&(a[:,5]>0);ids=np.flatnonzero(use)
            if not len(ids):continue
            routes[r['seed']].append(float(np.square(pred[use]-actual[use]).mean()));zeros[r['seed']].append(float(np.square(actual[use]).mean()))
            amp.extend(mu[use,4]);lower_saturated.extend(mu[use,4]<=-5);pred_norm.extend(np.max(np.abs(pred[use]),-1));target_norm.extend(np.max(np.abs(actual[use]),-1))
            selected=ids[np.linspace(0,len(ids)-1,8).astype(int)]
            full_sample.extend(np.square(pred[selected]-actual[selected]).mean(-1))
            for burn in [32,128]:
                for first in selected:
                    x=torch.as_tensor(s[max(0,int(first)-burn):int(first)+1][None],dtype=torch.float64);h,_=m.encoder(x);mm=m.actor(h)[0,-1];cc=mm[:5].clamp(-5,5).tanh();p=(cc[:4]*(cc[4]+1)/2).numpy()
                    window[burn].append(float(np.square(p-actual[first]).mean()))
    mse=float(np.mean([np.mean(v) for v in routes.values()]));zero=float(np.mean([np.mean(v) for v in zeros.values()]))
    return dict(route_mean_active_mse=mse,zero_proposal_mse=zero,skill_over_zero=1-mse/zero,raw_amplitude_logit_quantiles=np.quantile(amp,[.05,.5,.95]).tolist(),
        fraction_amplitude_below_gradient_clamp=float(np.mean(lower_saturated)),predicted_delta_norm_quantiles=np.quantile(pred_norm,[.05,.5,.95]).tolist(),target_delta_norm_quantiles=np.quantile(target_norm,[.05,.5,.95]).tolist(),
        sampled_full_history_mse=float(np.mean(full_sample)),sampled_reset32_mse=float(np.mean(window[32])),sampled_reset128_mse=float(np.mean(window[128])))

if __name__=='__main__':
    torch.set_num_threads(1);start=time.time();data=samples();rows=[]
    for name,path in [('round6',old_model(201,'bc'))]+[(str(g),OUT/'bc_curve'/f'groups_{g}'/'seed_301/final.pt') for g in [0,16,64,256]]:
        result=evaluate(path,data);rows.append(dict(name=name,path=str(path),**result));print('PROPOSAL_DIAGNOSTIC',name,result,flush=True)
    save(OUT/'h1_proposal_diagnostic.json',dict(results=rows,seconds=time.time()-start,scope='Reserved teacher trajectories, proposal regression independent of gate execution.'))
