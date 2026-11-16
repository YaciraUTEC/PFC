"""Conservative complete-policy OFF replay saves repeated identical physics.
Every baseline history is scanned through the recurrent proposal and authorizer;
only sufficient all-OFF cases are reused. Every other case is freshly simulated.
This is an action-identity optimization, not a physical-safety certificate.
"""
import time
from pathlib import Path
import numpy as np
import torch
from .composed_replay import certify
from .common import save,read
from .policy import TransformerPolicy
from .evaluation import evaluate,key
from .execution import settings


def baseline(case_list,folder,workers=None):
    workers=settings()['workers'] if workers is None else workers
    folder=Path(folder);path=folder/'baseline.json'
    if path.exists():
        result=read(path)
        if {key(r) for r in result['episodes']}!={key(r) for r in case_list}:raise RuntimeError('Baseline case allocation changed')
        if not all(Path(r['trace_path']).exists() for r in result['episodes']):raise RuntimeError('Missing baseline histories')
        return result
    return evaluate(None,case_list,workers=workers,path=path,capture_directory=folder/'baseline_histories')


def run(spec,base,folder,workers=None):
    start=time.time();folder=Path(folder);options=settings();workers=options['workers'] if workers is None else workers;batch=options['certification_batch'];model=TransformerPolicy.from_spec(spec).to(options['certification_device']).double();rows=[];pending=[];proof=[]
    torch.set_num_threads(1)
    for first in range(0,len(base['episodes']),batch):
        group=base['episodes'][first:first+batch];observations=[]
        for row in group:
            with np.load(row['trace_path']) as z:observations.append(z['observation'].copy())
        if spec.get('mode')=='off':flags=np.ones(len(group),bool);signals=np.zeros(len(group))
        else:flags,signals=certify(model,observations)
        for row,off,signal in zip(group,flags,signals):
            if off:
                rows.append(dict(row,baseline_reused_verified_off=True,fresh_simulation=False))
                proof.append(dict(case=list(key(row)),maximum_conservative_activity_indicator=float(signal),basis='complete composed policy cannot actuate throughout baseline history, with permissive numerical margins'))
            else:pending.append({k:row[k] for k in ['seed','level','condition','noise','clock']})
    del model
    if options['certification_device'].startswith('cuda'):torch.cuda.empty_cache()
    if pending:
        deployment=TransformerPolicy.from_spec(spec)
        if options['fast_gtrxl']:
            from .fast_gtrxl import install
            deployment=install(deployment)
        result=evaluate(deployment,pending,workers=workers,inference_device=options['inference_device'],capture_directory=folder/'changed_outcome_traces',capture_baseline=base)
        rows.extend(dict(r,baseline_reused_verified_off=False,fresh_simulation=True) for r in result['episodes'])
    assert len(rows)==len(base['episodes'])
    result=dict(episodes=rows,seconds=time.time()-start,logical_cases=len(rows),fresh_simulations=len(pending),verified_off_reuse=len(proof),
        execution_settings=options,proof_scope='Actuation identity by induction under complete-policy abstention; ambiguous/borderline cases receive fresh closed-loop simulation. No safety certificate for an intervention.',spec=spec)
    save(folder/'off_replay_proof.json',proof);save(folder/'results.json',result);return result
