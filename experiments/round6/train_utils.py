"""Round6 sequence helpers; no mutation of prior-round module globals."""
import numpy as np
import torch
from .env import DIM
from .models import tree_map,detach,select,concatenate
def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)

def sequences(dones,length=96):
    T,N=dones.shape;items=[]
    for j in range(N):
        episode_start=0;start=0
        while start<T:
            end=min(T,start+length);hits=np.flatnonzero(dones[start:end,j])
            if len(hits):end=start+int(hits[0])+1
            pre=min(32,start-episode_start);items.append((start,end,j,start-pre,pre))
            if dones[end-1,j]:episode_start=end
            start=end
    return items

def seq_batch(items,obs,states):
    length=max(b-a for a,b,*_ in items);x=np.zeros((len(items),length,DIM),np.float32);mask=np.zeros((len(items),length),np.float32);initial=[];groups={}
    for k,(a,b,j,p,pre) in enumerate(items):
        x[k,:b-a]=obs[a:b,j];mask[k,:b-a]=1.;initial.append(tree_map(lambda t:t[j:j+1].cuda(),states[p]));groups.setdefault(pre,[]).append(k)
    initial=concatenate(initial);state=tree_map(lambda t:t.clone(),initial)
    return tensor(x),tensor(mask),state,groups

def advance_burnin(m,items,obs,state,groups):
    with torch.no_grad():
        for pre,indices in groups.items():
            if pre==0:continue
            prefix=np.stack([obs[items[k][3]:items[k][0],items[k][2]] for k in indices]);_,new=m.encoder(tensor(prefix),select(state,indices))
            def put(dst,src):
                if isinstance(dst,(list,tuple)):
                    for a,b in zip(dst,src):put(a,b)
                else:dst[indices]=src
            put(state,new)
    return detach(state)

def padded(items,array,length,canonical_actions=False):
    out=np.zeros((len(items),length,*array.shape[2:]),np.float32)
    if canonical_actions:out[...,4:]=-1.
    for k,(a,b,j,*_) in enumerate(items):out[k,:b-a]=array[a:b,j]
    return tensor(out)

def mixed_kl(m,mu,reference_mu,reference_std):
    p,q=m.distribution(mu);rp=torch.distributions.Normal(reference_mu[...,:5],reference_std.exp())
    rq=torch.distributions.Bernoulli(logits=reference_mu[...,5])
    return torch.distributions.kl_divergence(q,rq)+q.probs*torch.distributions.kl_divergence(p,rp).sum(-1)

