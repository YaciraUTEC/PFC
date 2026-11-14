"""Exact OFF replay on captured baseline observations, with conservative margins.
Non-OFF or borderline cases always receive a fresh closed-loop simulation.
"""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from .models import Agent,physics_observation
from .offpolicy import OffPolicy
from .evaluate import OUT,evaluate,save

MARGIN=1e-7

def cache_baseline(label,config,condition,noise,seeds,levels,role='test'):
    folder=OUT/'cache'/role/label/f'{condition}_noise{noise:g}';folder.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for level in levels:
        marker=folder/f'level_{level:.6f}.json'
        if not marker.exists():
            result,traces=evaluate(dict(config,noise_multiplier=noise),condition,seeds,[level],workers=8,capture=True)
            for r in result['episodes']:
                np.savez_compressed(folder/f"{level:.6f}_{r['seed']}.npz",observations=traces[(level,r['seed'])])
            save(marker,result)
        allrows.extend(json.loads(marker.read_text())['episodes'])
    return allrows,folder

def certify(model,observations,ungated=False):
    """One batched recurrent scan. No privileged state or new physics."""
    lengths=np.array([len(x) for x in observations]);n=len(lengths);state=None;off=np.ones(n,bool);max_signal=np.full(n,-np.inf)
    p=next(model.parameters());dtype=p.dtype;device=p.device
    for first in range(0,int(lengths.max()),64):
        width=min(64,int(lengths.max())-first);raw=np.zeros((n,width,88),np.float32)
        valid=np.arange(first,first+width)[None]<lengths[:,None]
        for j,x in enumerate(observations):
            end=min(len(x),first+width)
            if end>first:raw[j,:end-first]=x[first:end]
        x=torch.as_tensor(raw,device=device,dtype=dtype)
        with torch.no_grad():
            if isinstance(model,Agent):
                mu,_,state=model(x,state)
                if ungated:
                    # A forced proposal only certifies OFF if amplitude/direction
                    # is exactly zero. Numerical near-zeros are simulated.
                    signal=torch.ones_like(mu[...,5]);safe=torch.zeros_like(signal,dtype=torch.bool)
                else:signal=mu[...,5];safe=signal < -MARGIN
            else:
                z,state=model.prior.encoder(x,state);features=torch.cat([physics_observation(x),z],-1).reshape(n*width,-1)
                clocks=torch.arange(first,first+width,device=device).repeat(n)
                noise=model.evaluation_noise(clocks) if model.method=='fql' else None
                action,_=model.proposal(features,deterministic=True,noise=noise)
                amp=action.abs().amax(-1)
                if not ungated and model.gate:
                    diff=(model.benefit_q(features,action)-model.benefit_q(features,torch.zeros_like(action))).amin(0)
                    safe=(diff<model.threshold-MARGIN)|(amp<.002-MARGIN);signal=diff-model.threshold
                else:safe=amp<.002-MARGIN;signal=amp-.002
                safe=safe.reshape(n,width);signal=signal.reshape(n,width)
        safe=safe.cpu().numpy();signal=signal.cpu().numpy()
        off &= np.all(safe|~valid,axis=1)
        for j in range(n):
            if valid[j].any():max_signal[j]=max(max_signal[j],float(signal[j,valid[j]].max()))
    return off,max_signal
class ForcedProposal(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def predict_actions(self,obs,state=None):
        _,proposal,st=self.model.propose_actions(obs,state);return proposal,st

def evaluate_cached(model,rows,folder,config,condition,noise,ungated=False,certification_batch=8):
    start=time.time();out=[];pending=[];proof=[];model=model.cpu().double().eval()
    for first in range(0,len(rows),certification_batch):
        batch=rows[first:first+certification_batch];observations=[np.load(folder/f"{r['level']:.6f}_{r['seed']}.npz")['observations'] for r in batch]
        flags,signals=certify(model,observations,ungated)
        for r,x,flag,signal in zip(batch,observations,flags,signals):
            if flag:
                out.append(dict(r,zero_replay_certified=True,checked_observations=len(x),max_gate_signal=float(signal)))
            else:pending.append((r['seed'],r['level']))
    if pending:
        if isinstance(model,OffPolicy):model.gate=not ungated
        deploy=ForcedProposal(model) if isinstance(model,Agent) and ungated else model
        result=evaluate(dict(config,noise_multiplier=noise),condition,model=deploy,case_pairs=pending,workers=8)
        out.extend(dict(r,zero_replay_certified=False) for r in result['episodes'])
    if isinstance(model,OffPolicy):model.gate=True
    assert len(out)==len(rows)
    return dict(episodes=sorted(out,key=lambda r:(r['level'],r['seed'])),seconds=time.time()-start,logical_cases=len(out),fresh_simulations=len(pending),certified_zero_cases=len(out)-len(pending))
