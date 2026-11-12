"""Predeclared paired statistics. No test-driven thresholds or model changes."""
import numpy as np
from scipy.stats import beta

ALPHA=.05/36

def harm_upper(losses,baseline_successes,alpha=ALPHA):
    k=int(losses);n=int(baseline_successes)
    if n<=0 or k>=n:return 1.
    if k==0:return float(-np.expm1(np.log(alpha)/n))
    return float(beta.ppf(1-alpha,k+1,n-k))

def route_interval(differences,replicates=100000,seed=197,alpha=ALPHA):
    d=np.asarray(differences,float)
    if d.ndim>1:d=d.mean(tuple(range(1,d.ndim)))
    rng=np.random.default_rng(seed);values=[];n=len(d)
    for first in range(0,replicates,500):
        indices=rng.integers(n,size=(min(500,replicates-first),n));values.extend(d[indices].mean(1))
    values=np.asarray(values)
    return dict(mean=float(d.mean()),lower=float(np.quantile(values,alpha/2)),upper=float(np.quantile(values,1-alpha/2)),routes=n,replicates=replicates,alpha=alpha)

def crossed_simultaneous(differences,replicates=10000,seed=193):
    """Kcontrasts x training seeds x route IDs x paired severities(optional)."""
    d=np.asarray(differences,float)
    if d.ndim==4:d=d.mean(-1)
    K,S,R=d.shape;point=d.mean((1,2));rng=np.random.default_rng(seed);samples=[]
    for first in range(0,replicates,500):
        count=min(500,replicates-first);sw=rng.multinomial(S,np.full(S,1/S),count)/S;rw=rng.multinomial(R,np.full(R,1/R),count)/R
        samples.append(np.einsum('bs,ksr,br->bk',sw,d,rw,optimize=True))
    draws=np.concatenate(samples);sd=draws.std(0,ddof=1);standardised=np.abs(draws-point)/np.where(sd>0,sd,1.)
    quantile=float(np.quantile(standardised.max(1),.95));bounds=[]
    for k in range(K):bounds.append(dict(mean=float(point[k]),lower=float(max(-1,point[k]-quantile*sd[k])),upper=float(min(1,point[k]+quantile*sd[k])),bootstrap_sd=float(sd[k])))
    return dict(contrasts=bounds,simultaneous_critical=quantile,replicates=replicates,training_seeds=S,routes=R,interpretation='Crossed training-seed/route bootstrap with simultaneous standardised band;not a rare-harm confidence interval.')

def compare_confirmation(candidate,reference):
    key=lambda r:(r['role'],r['seed'],r['level'],r['condition'])
    cm={key(r):r for r in candidate};rm={key(r):r for r in reference};assert set(cm)==set(rm)
    route_ids=sorted({r['seed'] for r in reference if r['role']=='primary'})
    differences=[]
    for seed in route_ids:
        keys=sorted(k for k in rm if k[0]=='primary' and k[1]==seed)
        assert len(keys)==3
        differences.append([int(cm[k]['stable_landing'])-int(rm[k]['stable_landing']) for k in keys])
    primary=route_interval(differences);risk={}
    for role in ['mild','exact']:
        keys=[k for k in rm if k[0]==role];n=sum(rm[k]['stable_landing'] for k in keys)
        loss=sum(rm[k]['stable_landing'] and not cm[k]['stable_landing'] for k in keys)
        risk[role]=dict(cases=len(keys),baseline_successes=int(n),losses=int(loss),upper=harm_upper(loss,n))
    statistical_pass=primary['lower']>0 and all(v['upper']<.01 for v in risk.values())
    return dict(primary=primary,risk=risk,statistical_pass=bool(statistical_pass),success=float(np.mean([r['stable_landing'] for r in candidate if r['role']=='primary'])),reference_success=float(np.mean([r['stable_landing'] for r in reference if r['role']=='primary'])))
