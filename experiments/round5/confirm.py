"""Independent production confirmation on1000new routes,selected before test."""
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from .inventory import OUT,core,cached_evaluation,TEST_LEVELS
from .evaluate import close_pool
from .campaign import freeze,verify
from ..run import save

MILD=[0.,.02,.05,.10,.15,.20,.25]
def build_plan():
    rng=np.random.default_rng(9142026);cases=[]
    for seed in range(531000,532000):
        cases.extend([dict(seed=seed,level=l,condition='estimated',role='primary') for l in [.28,.29,.295]])
        cases.append(dict(seed=seed,level=float(rng.choice(MILD)),condition='estimated',role='mild'))
        cases.append(dict(seed=seed,level=float(rng.choice(TEST_LEVELS)),condition='exact',role='exact'))
    return cases

def plan():
    path=OUT/'confirmation_case_plan.json'
    if path.exists():return json.loads(path.read_text())
    cases=build_plan();save(path,cases);return cases

def specifications():
    original=dict(estimator='original',readiness=False,emergency='legacy',reserve=.005)
    specs=[dict(id='baseline_original',path=None,config=original,threshold=0.,shadow=False,kind='baseline'),dict(id='baseline_selected',path=None,config=core(),threshold=0.,shadow=False,kind='baseline')]
    observer=json.loads((OUT/'observer_fallback_selection.json').read_text())
    if observer['production_eligible']:specs.insert(1,dict(id='baseline_observer_only',path=None,config=observer['config'],threshold=0.,shadow=False,kind='baseline'))
    challenger=json.loads((OUT/'structural_challenger.json').read_text())
    if challenger['production_eligible']:specs.append(dict(id='baseline_balanced',path=None,config=challenger['config'],threshold=0.,shadow=False,kind='baseline'))
    candidates=json.loads((OUT/'runtime_selection_complete.json').read_text())['production_candidates']
    for c in candidates:specs.append(dict(id=c['id'],path=c['path'],config=c['config'],threshold=c['strategy']['threshold'],shadow=c['strategy'].get('shadow',False),kind=c['kind'],seed=c['seed']))
    return specs

def evaluate_spec(spec,cases):
    torch.set_num_threads(1);start=time.time();allrows=[]
    try:
        for role in ['primary','mild','exact']:
            part=[r for r in cases if r['role']==role]
            for condition in sorted({r['condition'] for r in part}):
                for level in sorted({r['level'] for r in part if r['condition']==condition}):
                    seeds=[r['seed'] for r in part if r['condition']==condition and r['level']==level]
                    target=OUT/'confirmation'/'parts'/f"{spec['id']}_{role}_{condition}_{level}.json"
                    rows=cached_evaluation(spec['path'],target,condition,seeds,[level],spec['config'],threshold=spec['threshold'],proposal_only=spec['shadow'])
                    allrows.extend([dict(r,role=role,condition=condition) for r in rows])
    finally:close_pool()
    assert len(allrows)==len(cases);save(OUT/'confirmation'/f"{spec['id']}.json",allrows)
    return dict(id=spec['id'],seconds=time.time()-start,cases=len(allrows))

def main():
    frozen=freeze();verify(frozen);cases=plan();specs=specifications();save(OUT/'confirmation_manifest.json',dict(specifications=specs,cases=len(cases),multiplicity_endpoints=36,one_sided_family_error=.05,
        target='Concrete frozen checkpoints,not a claim about every training seed',risk='Conditional probability of harming a baseline-success flight in the predeclared severity mixture'))
    start=time.time();done=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(evaluate_spec,s,cases) for s in specs]
        for f in as_completed(fs):done.append(f.result());save(OUT/'confirmation_progress.json',dict(done=done,total=len(specs),seconds=time.time()-start));print('CONFIRMATION',done[-1],flush=True)
    verify(frozen);save(OUT/'confirmation_complete.json',dict(seconds=time.time()-start,models=done,frozen_verified=True))
if __name__=='__main__':main()
