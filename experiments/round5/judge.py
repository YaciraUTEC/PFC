"""Ensemble gain prediction with held-out conservative residual calibration.
Calibrated lower bounds are empirical diagnostics, not formal control guarantees.
Unsupported proposals abstain; acceptance never uses true fault effectiveness.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from ..run import save
from .evaluate import OUT

class BenefitJudge(nn.Module):
    def __init__(self,dim=86,width=96):
        super().__init__();self.dim=dim;self.width=width
        self.register_buffer('mean',torch.zeros(dim));self.register_buffer('scale',torch.ones(dim))
        self.models=nn.ModuleList([nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,1)) for _ in range(5)])
        self.register_buffer('calibration',torch.tensor(1.));self.register_buffer('minimum',torch.tensor(0.))
        self.register_buffer('lower_support',torch.full((dim,),-10.));self.register_buffer('upper_support',torch.full((dim,),10.))
    def forward(self,x):
        xx=((x-self.mean)/self.scale).clamp(-12,12)
        return 100.*torch.stack([m(xx).squeeze(-1) for m in self.models])
    @torch.no_grad()
    def accept(self,features):
        x=torch.as_tensor(features,dtype=torch.float32).reshape(1,-1);pred=self(x)
        lower=float(pred.mean(0)-2*pred.std(0,unbiased=False)-self.calibration)
        support=bool(((x>=self.lower_support)&(x<=self.upper_support)).all())
        return support and lower>float(self.minimum),lower
    def save(self,path,metadata):
        torch.save(dict(dim=self.dim,width=self.width,state=self.state_dict(),metadata=metadata),path)
    @classmethod
    def load(cls,path):
        ck=torch.load(path,map_location='cpu',weights_only=False);m=cls(ck['dim'],ck['width']);m.load_state_dict(ck['state']);return m.eval()

def fit(data,updates=5000):
    started=time.time();records=data['branches'];train=[r for r in records if r['split']=='train'];validation=[r for r in records if r['split']=='validation']
    if not train or not validation:raise RuntimeError('Independent train/validation gain data required')
    x=np.asarray([r['features'] for r in train],np.float32)
    surplus=lambda r:r['gain']-max(.015,.03*abs(r['base_cost']))
    y=np.clip([surplus(r) for r in train],-200,200).astype(np.float32)
    positive=np.flatnonzero(y>0);negative=np.flatnonzero(y<-.03);neutral=np.flatnonzero((y<=0)&(y>=-.03))
    vx=torch.tensor(np.asarray([r['features'] for r in validation],np.float32));vy=torch.tensor(np.asarray([surplus(r) for r in validation],np.float32))
    torch.manual_seed(193);m=BenefitJudge().cuda();m.mean.copy_(torch.as_tensor(x.mean(0),device='cuda'));m.scale.copy_(torch.as_tensor(np.maximum(x.std(0),.05),device='cuda'))
    # Support envelope uses train data only; calibration route predictions are separate.
    m.lower_support.copy_(torch.as_tensor(np.quantile(x,.001,axis=0)-.15,device='cuda'))
    m.upper_support.copy_(torch.as_tensor(np.quantile(x,.999,axis=0)+.15,device='cuda'))
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(193)
    xt=torch.as_tensor(x,device='cuda');yt=torch.as_tensor(y,device='cuda')
    for it in range(updates):
        loss=0.
        for member in m.models:
            parts=[rng.choice(pool,128 if k==0 else 64,replace=True) if len(pool) else rng.integers(len(x),size=128 if k==0 else 64) for k,pool in enumerate([positive,negative,neutral])]
            ids=torch.tensor(np.concatenate(parts),device='cuda');pred=member(((xt[ids]-m.mean)/m.scale).clamp(-12,12)).squeeze(-1)
            loss+=torch.nn.functional.huber_loss(pred,yt[ids]/100.,delta=.1)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step()
    m=m.cpu().eval()
    with torch.no_grad():
        predictions=m(vx);lower=predictions.mean(0)-2*predictions.std(0,unbiased=False)
        # Upper residual quantile penalizes optimism on reserved geometries.
        error=lower-vy;calibration=max(0.,float(torch.quantile(error,.99)));m.calibration.fill_(calibration)
        accepted_bound=(lower-calibration)>0.
        supported=((vx>=m.lower_support)&(vx<=m.upper_support)).all(-1)
        accepted=accepted_bound&supported
        report=dict(seconds=time.time()-started,train_pairs=len(train),validation_pairs=len(validation),calibration=calibration,
            heldout_lower_bound_coverage=float((lower-calibration<=vy).float().mean()),
            heldout_acceptance=float(accepted.float().mean()),heldout_bound_only_acceptance=float(accepted_bound.float().mean()),heldout_support_fraction=float(supported.float().mean()),
            heldout_nonbenefit_rate_among_accepted=None if not accepted.any() else float((vy[accepted]<=0).float().mean()),
            heldout_harm_rate_among_accepted=None if not accepted.any() else float((torch.tensor([r['gain'] for r in validation])[accepted]<0).float().mean()),
            regression_target='Paired full-mission gain minus max(0.015,3percent absolute base cost)',
            balanced_training=dict(beneficial=int(len(positive)),harmful=int(len(negative)),neutral=int(len(neutral))),calibration_quantile=.99,
            interpretation='Marginal heldout calibration under this proposal distribution,not a deployment guarantee or conditional near-limit bound.')
    target=OUT/'judge.pt';m.save(target,report);save(OUT/'judge_calibration.json',report);return target
