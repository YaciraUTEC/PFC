"""Float64 inference parity,with the common float32 actuator boundary."""
import json
import numpy as np
import torch
from .models import ARCHS
from .inference import load_agent
from .check_cuda_batch import PaddedAgent
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for arch in ARCHS:
        choice=json.loads((OUT/'hpo'/f'{arch}_selection.json').read_text())
        path=OUT/'hpo'/f"{arch}_{choice['selected_index']}"/'best.pt'
        for condition in ['exact','estimated']:
            rows={}
            for device,batch in [('cpu',8),('cpu',16),('cuda',8)]:
                m=load_agent(path,device=device)
                rows[(device,batch)]=evaluate(PaddedAgent(m,batch),seeds=range(430000,430004),
                    levels=[.25,.29],condition=condition,workers=4)
                del m
            base=rows[('cpu',8)]
            for variant in [('cpu',16),('cuda',8)]:
                other=rows[variant]
                result=dict(architecture=arch,condition=condition,variant=str(variant),
                    outcome_equal=all(a['outcome']==b['outcome'] for a,b in zip(base,other)),
                    max_rmse_difference=max(abs(a['tracking_rmse']-b['tracking_rmse']) for a,b in zip(base,other)),
                    max_step_difference=max(abs(a['episode_steps']-b['episode_steps']) for a,b in zip(base,other)),cases=len(base))
                results.append(result);save(OUT/'float64_parity_progress.json',results);print(result,flush=True)
                if not result['outcome_equal'] or result['max_rmse_difference']>1e-6 or result['max_step_difference']!=0:
                    raise RuntimeError('Float64 reference not stable: '+str(result))
    close_pool();save(OUT/'float64_parity.json',dict(passed=True,results=results,
        precision='float64 inference,float32 action boundary,original float32 checkpoints',
        weights='HPO',final_test_used=False))
if __name__=='__main__':main()
