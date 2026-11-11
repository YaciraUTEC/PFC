"""Representative one-drone inference time including command return to CPU.
This measures the network interface,not estimation/allocation or hardware control.
"""
import time
import numpy as np
import torch
from .models import ARCHS,Agent,tree_map
from .evaluate import OUT
from ..run import save

def run():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;results=[]
    path=OUT/'data_governor_v2'/'route_032.npz'
    data=np.load(path)['s'][:350]
    for arch in ARCHS:
        checkpoint=OUT/'models'/arch/'seed_71'/'best.pt'
        for device in ['cpu','cuda']:
            m=Agent.load(checkpoint,device=device).double();state=None;durations=[]
            with torch.no_grad():
                for t,frame in enumerate(data):
                    if t%100==0:state=None
                    if device=='cuda':torch.cuda.synchronize()
                    start=time.perf_counter()
                    x=torch.as_tensor(frame[None],device=device,dtype=torch.float64)
                    action,state=m.predict_actions(x,state)
                    command=action.cpu().numpy().astype(np.float32)
                    if device=='cuda':torch.cuda.synchronize()
                    duration=(time.perf_counter()-start)*1000
                    if t>=50:durations.append(duration)
            results.append(dict(architecture=arch,device=device,network_dtype='float64',parameters=sum(p.numel() for p in m.parameters()),
                observations=len(durations),p50_ms=float(np.median(durations)),p95_ms=float(np.quantile(durations,.95)),
                p99_ms=float(np.quantile(durations,.99)),control_period_ms=1000/48,
                scope='input transfer,one persistent policy step,command return;excludes estimator/allocator'))
            del m;torch.cuda.empty_cache()
    save(OUT/'inference_latency.json',results);return results
if __name__=='__main__':run()
