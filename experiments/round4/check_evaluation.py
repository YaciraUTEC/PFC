"""Evaluation batching/pool-state parity on reserved validation cases."""
import numpy as np
import torch
from .models import Agent
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=Agent.load(OUT/'hpo'/'lstm_0'/'bc.pt')
    a=evaluate(m,seeds=[430000,430001,430002,430003],levels=[.25,.29],workers=4,condition='exact')
    b=evaluate(m,seeds=[430000,430001,430002,430003],levels=[.25,.29],workers=8,condition='exact')
    # Reusing one pool across condition switches must match a fresh pool.
    c=evaluate(m,seeds=[430000,430001,430002,430003],levels=[.25,.29],workers=4,condition='estimated')
    evaluate(m,seeds=[430000],levels=[0],workers=4,condition='exact')
    d=evaluate(m,seeds=[430000,430001,430002,430003],levels=[.25,.29],workers=4,condition='estimated')
    checks=[]
    for name,x,y in [('batch4_vs8',a,b),('pool_condition_switch',c,d)]:
        for one,two in zip(x,y):
            assert one['stable_landing']==two['stable_landing']
            assert one['outcome']==two['outcome']
            assert one['episode_steps']==two['episode_steps']
        err=max(abs(one['tracking_rmse']-two['tracking_rmse']) for one,two in zip(x,y))
        assert err<1e-5,(name,err)
        checks.append(dict(name=name,max_rmse_difference=err,cases=len(x)))
    save(OUT/'evaluation_parity.json',checks);close_pool();print(checks,flush=True)
if __name__=='__main__':main()
