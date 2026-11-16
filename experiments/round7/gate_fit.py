"""Route-separated terminal-outcome modeling with held-out calibration diagnostics.
Ensemble disagreement is a heuristic penalty, not a calibrated confidence bound.
"""
import time,math,argparse,hashlib
from collections import Counter
import numpy as np
import torch
from torch.nn import functional as F
from .common import OUT,save,read,close_milestone,sha
from ..round5.bc import setup
from .policy import OutcomeGate


def load(split):
    xx=[];yy=[];ss=[];calibration_flags=[]
    for p in sorted((OUT/'gate_data'/split).glob('*.npz')):
        with np.load(p) as z:
            xx.append(z['x']);yy.append(z['y']);ss.append(z['seed']);n=len(z['y'])
        records=read(p.with_suffix('.json'))['records'];r=records[0] if records else dict(condition='exact',noise=1.,clock=None)
        cell=int(r['condition']=='estimated')+int(r['noise']-1)+[None,4.,6.,8.].index(r['clock'])
        calibration_flags.append(np.full(n,cell%2==0,bool))
    return np.concatenate(xx),np.concatenate(yy),np.concatenate(ss),np.concatenate(calibration_flags)

def diagnostics(model,x,y):
    with torch.no_grad():
        probs=torch.cat([model.probabilities(torch.as_tensor(a,dtype=torch.float32,device='cuda')).mean(0).cpu() for a in np.array_split(x,max(1,math.ceil(len(x)/1024))) if len(a)]).numpy()
    truth=np.eye(4)[y];pred=probs.argmax(-1)
    bins=[]
    advantage=probs[:,1]-probs[:,2]
    for lo,hi in [(-1.,0.),(0.,.025),(.025,.05),(.05,.10),(.10,.25),(.25,1.01)]:
        use=(advantage>=lo)&(advantage<hi)
        bins.append(dict(lower=lo,upper=hi,n=int(use.sum()),predicted_advantage=float(advantage[use].mean()) if use.any() else None,
            observed_advantage=float(((y[use]==1).astype(float)-(y[use]==2)).mean()) if use.any() else None,harm_rate=float((y[use]==2).mean()) if use.any() else None))
    confusion=np.zeros((4,4),int);np.add.at(confusion,(y,pred),1)
    class_metrics={str(c):dict(truth=int((y==c).sum()),predicted=int((pred==c).sum()),recall=float(confusion[c,c]/max(1,(y==c).sum())),precision=float(confusion[c,c]/max(1,(pred==c).sum()))) for c in range(4)}
    return dict(confusion_matrix=confusion.tolist(),per_class=class_metrics,samples=len(y),brier=float(np.mean(np.sum((probs-truth)**2,-1))),nll=float(-np.log(probs[np.arange(len(y)),y].clip(1e-8)).mean()),accuracy=float(np.mean(pred==y)),
        labels=dict(Counter(map(int,y))),benefit_bins=bins,scope='Marginal validation diagnostics on paired queries; not a per-flight safety certificate')

def main(output="outcome_gate.pt",milestone="h2_gate_model",seed=307):
    if milestone=='h2_gate_model':
        while not (OUT/'h2_natural_labels_conclusion.json').exists():time.sleep(5)
    start=time.time();setup(seed)
    x,y,seeds,_=load('train');vx,vy,vs,calibration=load('calibration');assert set(seeds).isdisjoint(set(vs))
    if len(x)<100:raise RuntimeError('Insufficient paired labels')
    m=OutcomeGate().cuda();m.center.copy_(torch.as_tensor(x.mean(0),device='cuda'));m.scale.copy_(torch.as_tensor(np.maximum(x.std(0),.05),device='cuda'))
    opts=[torch.optim.AdamW(member.parameters(),lr=3e-4,weight_decay=1e-4) for member in m.members];rng=np.random.default_rng(seed);logs=[]
    route_ids=np.unique(seeds);bootstraps=[rng.choice(route_ids,len(route_ids),replace=True) for _ in m.members]
    pools=[np.concatenate([np.flatnonzero(seeds==s) for s in b]) for b in bootstraps]
    for it in range(2400):
        lr=3e-4*(.1+.9*.5*(1+math.cos(math.pi*it/2400)));losses=[]
        for j,opt in enumerate(opts):
            ids=rng.choice(pools[j],512);tx=torch.as_tensor(x[ids],device='cuda');ty=torch.as_tensor(y[ids],device='cuda')
            z=((tx-m.center)/m.scale).clamp(-12,12);logits=m.members[j](z);loss=F.cross_entropy(logits,ty)
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.members[j].parameters(),1.);opt.step();losses.append(float(loss))
        if it%200==0:
            row=dict(update=it,loss=float(np.mean(losses)),seconds=time.time()-start);logs.append(row);save(OUT/(milestone+'_training_progress.json'),row);print('OUTCOME_GATE',row,flush=True)
    # Temperature is selected on the first route half; diagnostics use the other.
    assessment=~calibration
    # Balanced across controllers, noise, effectiveness condition and clock.
    assert len(np.unique(vs[calibration])) and len(np.unique(vs[assessment]))
    best=(float('inf'),1.)
    for temp in [.5,.75,1.,1.5,2.,3.]:
        m.temperature.fill_(temp);d=diagnostics(m,vx[calibration],vy[calibration])
        if d['nll']<best[0]:best=(d['nll'],temp)
    m.temperature.fill_(best[1]);metric=diagnostics(m,vx[assessment],vy[assessment]);priors=np.bincount(y,minlength=4)/len(y);truth=np.eye(4)[vy[assessment]]
    null_brier=float(np.mean(np.sum((priors-truth)**2,-1)));metric['constant_training_prior_reference']=dict(probabilities=priors.tolist(),brier=null_brier,nll=float(-np.log(priors[vy[assessment]].clip(1e-8)).mean()),brier_skill=1-metric['brier']/null_brier if null_brier else None)
    fingerprint={str(p.relative_to(OUT)):sha(p) for split in ['train','calibration'] for p in sorted((OUT/'gate_data'/split).glob('*.npz'))}
    m.cpu().save(OUT/output,seed=seed,data_fingerprint=hashlib.sha256(str(sorted(fingerprint.items())).encode()).hexdigest(),temperature=best[1],training_samples=len(y),train_seed_count=len(route_ids),calibration=metric)
    save(OUT/(milestone+'_prediction_diagnostics.json'),dict(temperature=best[1],training_samples=len(y),training_labels=dict(Counter(map(int,y))),heldout=metric,seconds=time.time()-start,ensemble_is_not_a_confidence_interval=True))
    close_milestone(milestone,'The authorizer predicts paired terminal outcomes; full-flight threshold selection and filter ablation remain separate.',diagnostics=metric,seconds=time.time()-start)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outcome_gate.pt');p.add_argument('--milestone',default='h2_gate_model');p.add_argument('--seed',type=int,default=307);a=p.parse_args();main(a.output,a.milestone,a.seed)
