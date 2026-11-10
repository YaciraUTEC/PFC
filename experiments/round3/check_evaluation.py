import time,json,numpy as np
from stable_baselines3 import PPO
from .common import evaluate as serial,OUT,save
from .evaluate import batched_evaluate,close_pool

def main():
    model=PPO.load(OUT/'seed_11/bc.zip',device='cuda')
    t=time.time();a=serial(model,seeds=range(100020,100022),levels=[0.,.2]);sa=time.time()-t
    t=time.time();b=batched_evaluate(model,seeds=range(100020,100022),levels=[0.,.2],workers=2);sb=time.time()-t
    a=sorted(a,key=lambda x:(x['level'],x['seed']))
    assert [r['outcome'] for r in a]==[r['outcome'] for r in b]
    for x,y in zip(a,b):
        assert x['episode_steps']==y['episode_steps']
        assert abs(x['tracking_rmse']-y['tracking_rmse'])<1e-5
    save(OUT/'evaluation_parity.json',dict(serial_seconds=sa,batched_seconds=sb,cases=4,outcome_match=True,step_match=True,max_rmse_difference=max(abs(x['tracking_rmse']-y['tracking_rmse']) for x,y in zip(a,b))))
    print('Evaluator parity passed',sa,sb,flush=True);close_pool()
if __name__=='__main__':main()