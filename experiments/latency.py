"""End-to-end batch-one decision latency on the user's desktop, not onboard timing."""
from pathlib import Path
import argparse,time,json
import numpy as np
import torch
from stable_baselines3 import PPO
from .run import ROOT,ARCHS,save
from .env import FlightEnv

def main():
    p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=100);args=p.parse_args()
    out=ROOT/'runs/dslpid_comparison_v2';dest=ROOT.parent/'resultados'
    env=FlightEnv(history=32,fixed_fault=.1,warm_start=True)
    try:obs,_=env.reset(seed=77)
    finally:env.close()
    results=[]
    for arch in ARCHS:
        row=dict(architecture=arch,batch=1,history=32,control_period_ms=1000/48)
        for device in ['cpu','cuda']:
            model=PPO.load(out/arch/'best.zip',device=device)
            row['policy_parameters']=sum(p.numel() for p in model.policy.parameters())
            for _ in range(10):model.predict(obs,deterministic=True)
            samples=[]
            for _ in range(args.repeats):
                if device=='cuda':torch.cuda.synchronize()
                t=time.perf_counter();action,_=model.predict(obs,deterministic=True)
                if device=='cuda':torch.cuda.synchronize()
                samples.append((time.perf_counter()-t)*1000)
                assert np.isfinite(action).all()
            row[device]=dict(n=len(samples),median_ms=float(np.median(samples)),
                p95_ms=float(np.percentile(samples,95)),mean_ms=float(np.mean(samples)),
                compiled_cell=bool(getattr(model.policy.features_extractor,'_ltc_compiled',False)))
            del model
            if device=='cuda':torch.cuda.empty_cache()
        results.append(row);print(row,flush=True)
        save(dest/'latencia_decision.json',dict(hardware='i9-12900K, RTX 3080 Ti 12GB',
            cpu_threads=torch.get_num_threads(),repeats=args.repeats,models=results,
            scope='Batch one deterministic policy.predict, includes CPU/GPU transfers; excludes sensor input, DSLPIDControl and actuator output. Fixed history 32; Measured after training stopped; desktop activity may affect timing. Not onboard or guaranteed real-time timing.'))
if __name__=='__main__':main()