import time,numpy as np,torch
from stable_baselines3 import PPO
from .common import OUT,save
from .evaluate import batched_evaluate,close_pool

def main():
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    m=PPO.load(OUT/'seed_11/bc.zip',device='cuda');reference=None;result=[]
    for workers,compact in [(4,False),(4,True)]:
        start=time.time();rr=batched_evaluate(m,seeds=range(100020,100025),workers=workers,compact=compact);cold=time.time()-start
        if reference is None:reference=rr
        assert [r['outcome'] for r in reference]==[r['outcome'] for r in rr]
        assert [r['episode_steps'] for r in reference]==[r['episode_steps'] for r in rr]
        error=max(abs(a['tracking_rmse']-b['tracking_rmse']) for a,b in zip(reference,rr));assert error<1e-4,error
        start=time.time();rr2=batched_evaluate(m,seeds=range(100020,100025),workers=workers,compact=compact);warm=time.time()-start
        assert [r['outcome'] for r in rr]==[r['outcome'] for r in rr2]
        result.append(dict(workers=workers,compact=compact,cases=len(rr),cold_seconds=cold,warm_seconds=warm,sim_steps=sum(r['episode_steps'] for r in rr),max_rmse_difference=error))
        print(result[-1],flush=True)
    close_pool();save(OUT/'compact_evaluation_parity.json',dict(results=result,outcomes_and_steps_match=True))
if __name__=='__main__':main()