"""Sensitivity for every family and every concrete production candidate."""
import json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from .inventory import groups,core,cached_evaluation
from .confirm import specifications
from .campaign import freeze,verify
from .evaluate import OUT,close_pool
from ..run import save

def specs():
    result=[dict(s,id='production_'+s['id']) for s in specifications()]
    for group in groups():
        entry=group['entries'][0];result.append(dict(id='family_'+group['id'],path=entry['path'],config=dict(core(),intervention=group['intervention']),threshold=0.,shadow=False,kind=group['kind'],seed=entry['seed']))
    return result

def job(spec):
    result=[];start=time.time()
    try:
        for mode,extra in [('noise2',dict(noise_multiplier=2.)),('clock4',dict(fault_clock=4.))]:
            target=OUT/'sensitivity'/f"{spec['id']}_{mode}.json"
            rows=cached_evaluation(spec['path'],target,'estimated',range(530000,530024),[0.,.25,.295,.296],dict(spec['config'],**extra),threshold=spec['threshold'],proposal_only=spec['shadow'])
            result.append(dict(mode=mode,cases=len(rows),success=sum(r['stable_landing'] for r in rows)))
    finally:close_pool()
    return dict(id=spec['id'],seconds=time.time()-start,result=result)

def main():
    frozen=freeze();verify(frozen);start=time.time();done=[];jobs=specs();save(OUT/'sensitivity_manifest.json',jobs)
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(job,j) for j in jobs]
        for f in as_completed(fs):done.append(f.result());save(OUT/'sensitivity_progress.json',dict(done=done,total=len(jobs),seconds=time.time()-start));print('SENSITIVITY',done[-1]['id'],flush=True)
    verify(frozen);save(OUT/'sensitivity_complete.json',dict(seconds=time.time()-start,jobs=done,passed_integrity=True))
if __name__=='__main__':main()
