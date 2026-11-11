"""CPU/GPU physical-flight parity before selecting final evaluation backend.
Uses HPO checkpoints and validation routes only,never final test routes.
"""
import json,time
import numpy as np
import torch
from .models import ARCHS,Agent
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks=[]
    for arch in ARCHS:
        choice=json.loads((OUT/'hpo'/f'{arch}_selection.json').read_text())
        path=OUT/'hpo'/f"{arch}_{choice['selected_index']}"/'best.pt'
        for condition in ['exact','estimated']:
            rows={}
            for device in ['cpu','cuda']:
                m=Agent.load(path,device=device);ts=time.time()
                rr=evaluate(m,seeds=range(430000,430004),levels=[.25,.29],workers=4,condition=condition)
                rows[device]=rr
                checks.append(dict(architecture=arch,condition=condition,device=device,seconds=time.time()-ts,cases=len(rr)))
                del m
            cpu,gpu=rows['cpu'],rows['cuda']
            outcome_equal=all(a['outcome']==b['outcome'] for a,b in zip(cpu,gpu))
            max_rmse=max(abs(a['tracking_rmse']-b['tracking_rmse']) for a,b in zip(cpu,gpu))
            max_steps=max(abs(a['episode_steps']-b['episode_steps']) for a,b in zip(cpu,gpu))
            result=dict(architecture=arch,condition=condition,outcome_equal=outcome_equal,
                max_rmse_difference=max_rmse,max_step_difference=max_steps)
            print(result,flush=True)
            save(OUT/'backend_flight_parity_progress.json',dict(timings=checks,last=result))
            if not outcome_equal or max_rmse>1e-4 or max_steps>1:
                raise RuntimeError('Backend differences exceed acceptance: '+str(result))
        print('ARCHITECTURE_PARITY_PASSED',arch,flush=True)
    close_pool();save(OUT/'backend_flight_parity.json',dict(passed=True,timings=checks,
        acceptance='same outcomes,maxRMSEdifference1e-4m,max1controltick,on96unique validation cases',
        weights='selected HPO checkpoints',final_test_used=False))
if __name__=='__main__':main()
