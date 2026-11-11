"""Two bounded PC training workers;each job owns one architecture/seed directory."""
import argparse,contextlib,os,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from .models import ARCHS
from .train import SEEDS,train
from .data import load,SequenceData
from .evaluate import OUT
from ..run import save
_DATA=None
def initialize():
    global _DATA
    os.environ['NICOL_EVAL_WORKERS']='4'
    _DATA=SequenceData(load())
def job(pair):
    arch,seed=pair
    d=OUT/'models'/arch/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True)
    with (d/'execution.log').open('a',encoding='utf-8',buffering=1) as f:
        with contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            train(_DATA,archs=[arch],seeds=[seed],keep_pool=True)
    return dict(architecture=arch,seed=seed)
def main(workers=2):
    start=time.time();jobs=[(a,s) for s in SEEDS for a in ARCHS]
    done=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as pool:
        fs=[pool.submit(job,j) for j in jobs]
        for f in as_completed(fs):
            result=f.result();done.append(result)
            save(OUT/'training_campaign_progress.json',dict(completed=len(done),total=len(jobs),jobs=done,
                workers=workers,seconds=time.time()-start))
            print('MODEL_COMPLETE',result,'total',len(done),'/',len(jobs),flush=True)
    save(OUT/'training_campaign_complete.json',dict(seconds=time.time()-start,workers=workers,models=len(jobs)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();main(a.workers)
