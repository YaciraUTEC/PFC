"""Prospective CPU/GPU inference calibration using HPO weights/validation states."""
import time,json
import numpy as np
import torch
from .models import ARCHS,Agent
from .evaluate import OUT
from ..run import save

def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=np.load(OUT/'data_governor_v2'/'route_032.npz')['s']
    results=[]
    for arch in ARCHS:
        choice=json.loads((OUT/'hpo'/f'{arch}_selection.json').read_text());path=OUT/'hpo'/f"{arch}_{choice['selected_index']}"/'best.pt'
        for batch in [1,8]:
            copies={d:Agent.load(path,device=d) for d in ['cpu','cuda']}
            predictions={};timings={}
            for device,m in copies.items():
                state=None;durations=[];acts=[]
                with torch.no_grad():
                    for t in range(220):
                        frame=data[(np.arange(batch)+t*batch)%len(data)]
                        if t%100==0:state=None
                        if device=='cuda':torch.cuda.synchronize()
                        start=time.perf_counter()
                        x=torch.as_tensor(frame,device=device,dtype=torch.float32)
                        a,state=m.predict_actions(x,state);out=a.cpu().numpy()
                        if device=='cuda':torch.cuda.synchronize()
                        if t>=20:durations.append((time.perf_counter()-start)*1000)
                        acts.append(out)
                predictions[device]=np.asarray(acts);timings[device]=dict(p50_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)))
            err=float(np.max(np.abs(predictions['cpu']-predictions['cuda'])))
            if err>3e-5:raise RuntimeError((arch,batch,err))
            result=dict(architecture=arch,batch=batch,max_action_difference=err,timings=timings,scope='HPO weights and validation states,concurrent training active')
            results.append(result);print(result,flush=True)
            del copies;torch.cuda.empty_cache()
    save(OUT/'inference_backend_calibration.json',results)
if __name__=='__main__':main()
