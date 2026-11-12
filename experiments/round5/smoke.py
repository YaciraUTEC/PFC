"""Software-only end-to-end training checks; never eligible for model selection."""
import copy,json,time
import numpy as np
import torch
from .models import Agent,ARCHS
from .bc import setup,starting_config
from .data import SequenceData
from .env import ZERO
from .evaluate import OUT,evaluate,close_pool
from . import train as tr
from ..run import save

def main():
    cfg=dict(estimator='original',readiness=False,emergency='legacy',reserve=.005)
    tr.core_config=lambda:cfg.copy()
    def tiny_validation(m,config=None):
        model=copy.deepcopy(m).cpu().double().eval();rows=[]
        for condition in ['exact','estimated']:rows.extend(evaluate(model,condition=condition,seeds=range(525000,525002),levels=[0.,.295],config=cfg))
        return dict(score=float(np.mean([r['stable_landing'] for r in rows])),episodes=rows)
    tr.validation=tiny_validation
    rng=np.random.default_rng(193);n=192
    data=dict(s=rng.normal(size=(n,80)).astype(np.float32),a=np.tile(ZERO,(n,1)),observer=np.tile(ZERO,(n,1)),valid=np.ones(n,bool),observer_valid=np.ones(n,bool),executed=np.tile(ZERO,(n,1)),
        records=[dict(start=0,stop=n,split='train',configuration=dict(intervention='residual'))],done=np.r_[np.zeros(n-1,bool),True],expert=np.ones(n,bool),train=np.ones(n,bool),validation=np.zeros(n,bool))
    dataset=SequenceData(data);start=time.time()
    for arch in ARCHS:
        dest=OUT/'smoke'/arch;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():continue
        setup(97);m=Agent(starting_config(arch)).cuda()
        tr.finetune(m,dataset,data,97,dest,budget=18432)
        result=json.loads((dest/'complete.json').read_text());assert result['steps']==18432
        del m;torch.cuda.empty_cache();print('SMOKE_COMPLETE',arch,'elapsed',round(time.time()-start),flush=True)
    close_pool();save(OUT/'smoke_complete.json',dict(purpose='Software verification with synthetic supervision and real simulator; excluded from scientific comparison',architectures=ARCHS,steps_each=18432,seconds=time.time()-start))
if __name__=='__main__':main()
