"""CUDA batch8/16 parity with four physical workers and padded independent rows."""
import json,time
import numpy as np
import torch
from .models import ARCHS,Agent,tree_map
from .evaluate import OUT,evaluate,close_pool
from ..run import save

class PaddedAgent:
    def __init__(self,model,batch):
        self.model=model;self.batch=batch;self.encoder=model.encoder
    def parameters(self):return self.model.parameters()
    def predict_actions(self,obs,state=None):
        n=obs.shape[0]
        x=torch.cat([obs,torch.zeros((self.batch-n,obs.shape[1]),device=obs.device)],0)
        if state is not None:
            state=tree_map(lambda s:torch.cat([s,torch.zeros((self.batch-n,*s.shape[1:]),device=s.device)],0),state)
        a,state=self.model.predict_actions(x,state)
        return a[:n],tree_map(lambda s:s[:n],state)

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for arch in ARCHS:
        choice=json.loads((OUT/'hpo'/f'{arch}_selection.json').read_text())
        path=OUT/'hpo'/f"{arch}_{choice['selected_index']}"/'best.pt'
        m=Agent.load(path)
        for condition in ['exact','estimated']:
            rows={}
            for batch in [8,16]:
                ts=time.time();rr=evaluate(PaddedAgent(m,batch),seeds=range(430000,430004),levels=[.25,.29],condition=condition,workers=4)
                rows[batch]=rr
            a,b=rows[8],rows[16]
            result=dict(architecture=arch,condition=condition,outcome_equal=all(x['outcome']==y['outcome'] for x,y in zip(a,b)),
                max_rmse_difference=max(abs(x['tracking_rmse']-y['tracking_rmse']) for x,y in zip(a,b)),
                max_step_difference=max(abs(x['episode_steps']-y['episode_steps']) for x,y in zip(a,b)),cases=len(a))
            results.append(result);save(OUT/'cuda_batch_parity_progress.json',results);print(result,flush=True)
            if not result['outcome_equal'] or result['max_rmse_difference']>1e-4 or result['max_step_difference']>1:
                raise RuntimeError('CUDA batch differences exceed acceptance: '+str(result))
        del m;torch.cuda.empty_cache()
    close_pool();save(OUT/'cuda_batch_parity.json',dict(passed=True,results=results,weights='HPO',final_test_used=False,
        note='batch8/16 actual actor execution,padded independent rows;4physical workers;accept same outcomes,<=1tick,RMSE<=1e-4m'))
if __name__=='__main__':main()
