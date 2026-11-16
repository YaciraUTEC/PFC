"""Development-only frozen K/V cache flight equivalence and batch cost."""
import time
import numpy as np
import torch
from .common import OUT,LEGACY_OUT,read,save,old_model,close_milestone
from .policy import TransformerPolicy
from .fast_gtrxl import install
from .evaluation import cases,evaluate,key


def main():
    torch.set_num_threads(1);start=time.time();cc=cases(range(719984,719992),levels=[.295,.296],conditions=['estimated'],noises=[1.,2.])
    specs=[('flash_native',dict(proposer=str(old_model(201)),mode='native')),('bc_counterfactual',dict(read(LEGACY_OUT/'h3_filter_ablation_conclusion.json')['selected']['spec']))]
    rows=[]
    for name,spec in specs:
        original=evaluate(TransformerPolicy.from_spec(spec),cc,workers=8)
        cached=evaluate(install(TransformerPolicy.from_spec(spec)),cc,workers=16)
        baseline={key(r):r for r in original['episodes']};largest=0.;equivalent=True;changed=[]
        for r in cached['episodes']:
            a=baseline[key(r)];error=float(np.abs(np.asarray(a['terminal_position'])-r['terminal_position']).max());largest=max(largest,error)
            same=a['outcome']==r['outcome'] and a['episode_steps']==r['episode_steps'] and error<1e-8
            equivalent &=same
            if not same:changed.append(dict(case=list(key(r)),old=a['outcome'],new=r['outcome'],position_error=error,old_steps=a['episode_steps'],new_steps=r['episode_steps']))
        result=dict(name=name,cases=len(cc),equivalent=bool(equivalent),maximum_position_error=largest,changed=changed,original_seconds=original['seconds'],cached_seconds=cached['seconds'],original_workers=8,cached_workers=16)
        rows.append(result);save(OUT/'kv_flight_pilot_progress.json',rows);print('KV_FLIGHT_PILOT',result,flush=True)
    close_milestone('kv_flight_pilot','Frozen K/V cache and16 parallel cases compared against original CPU implementation with8 cases; all are development-only routes.',results=rows,all_equivalent=all(r['equivalent'] for r in rows),seconds=time.time()-start)
if __name__=='__main__':main()
