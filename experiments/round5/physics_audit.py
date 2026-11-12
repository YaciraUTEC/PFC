"""Natural continuation versus restored paired branches; shared-sensor physics."""
import json,time,contextlib,os
import numpy as np
import pybullet as p
from .env import LandingEnv,ZERO
from .counterfactual import snapshot,restore,rollout_cost
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def main():
    cfg=json.loads((OUT/'core_selection.json').read_text())['config'];rows=[];start=time.time()
    for seed in range(527000,527012):
        for level in [0.,.25,.295]:
            with open(os.devnull,'w') as f,contextlib.redirect_stdout(f):e=LandingEnv(config=cfg)
            try:
                e.reset(seed=seed,options=dict(level=level));point=96 if seed%2==0 else 192
                alive=True
                for _ in range(point):
                    _,_,d,t,_=e.step(ZERO)
                    if d or t:alive=False;break
                if not alive:continue
                saved=snapshot(e);horizon=1440-e.t
                natural,outcome=rollout_cost(e,ZERO,'residual',horizon=horizon)
                restore(e,saved);first,one=rollout_cost(e,ZERO,'residual',horizon=horizon)
                restore(e,saved);second,two=rollout_cost(e,ZERO,'residual',horizon=horizon)
                p.removeState(saved[0],physicsClientId=e.core._env.CLIENT)
                row=dict(seed=seed,level=level,query_tick=point,natural_cost=natural,restored_cost=first,repeated_cost=second,
                    natural_outcome=outcome,restored_outcome=one,repeated_outcome=two,repeat_identical=bool(first==second and one==two),
                    natural_outcome_agreement=outcome==one,natural_cost_difference=abs(natural-first),
                    below_gate_margin=abs(natural-first)<max(.015,.03*abs(natural)))
                rows.append(row)
            finally:e.close()
    summary=dict(cases=len(rows),repeat_identical=sum(r['repeat_identical'] for r in rows),natural_outcome_agreement=sum(r['natural_outcome_agreement'] for r in rows),
        below_gate_margin=sum(r['below_gate_margin'] for r in rows),max_natural_cost_difference=max(r['natural_cost_difference'] for r in rows),seconds=time.time()-start,records=rows)
    save(OUT/'paired_physics_audit.json',summary);print('PAIRED_PHYSICS_AUDIT',{k:v for k,v in summary.items() if k!='records'},flush=True)
    primary=[]
    for contact_hz,iterations in [(960,200),(1920,200),(960,400)]:
        for condition in ['exact','estimated']:
            target=OUT/'physics_precision'/f'{contact_hz}_{iterations}_{condition}.json'
            rr=evaluate(condition=condition,seeds=range(527000,527012),levels=[0.,.25,.295],config=dict(cfg,contact_hz=contact_hz,solver_iterations=iterations))
            save(target,rr);primary.append(dict(contact_hz=contact_hz,iterations=iterations,condition=condition,success=sum(r['stable_landing'] for r in rr),cases=len(rr)))
    close_pool();save(OUT/'physics_precision_summary.json',primary);print('PHYSICS_PRECISION',json.dumps(primary),flush=True)
if __name__=='__main__':main()
