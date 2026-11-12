"""Engineering pilot; final-test route IDs are never opened here."""
import json,time,argparse
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def main():
    start=time.time();results=[]
    variants=[dict(label='original',estimator='original',readiness=False,emergency='legacy'),
        dict(label='ready',estimator='original',readiness=True,emergency='legacy'),
        dict(label='impulse',estimator='impulse',readiness=True,emergency='legacy'),
        dict(label='adaptive',estimator='adaptive',readiness=True,emergency='legacy'),
        dict(label='adaptive_no_ready',estimator='adaptive',readiness=False,emergency='legacy'),
        dict(label='impulse_no_ready',estimator='impulse',readiness=False,emergency='legacy'),
        dict(label='damped',estimator='adaptive',readiness=True,emergency='damped'),
        dict(label='balanced',estimator='adaptive',readiness=True,emergency='balanced')]
    try:
        for variant in variants:
            label=variant['label'];cfg={k:v for k,v in variant.items() if k!='label'}
            for condition in ['exact','estimated']:
                target=OUT/'engineering_v3'/f'{label}_{condition}.json'
                if target.exists():rows=json.loads(target.read_text())
                else:rows=evaluate(condition=condition,seeds=range(525000,525012),levels=[0.,.02,.25,.28,.29,.295],config=cfg);save(target,rows)
                summary={str(l):sum(r['stable_landing'] for r in rows if r['level']==l) for l in sorted({r['level'] for r in rows})}
                results.append(dict(label=label,condition=condition,config=cfg,successes=summary))
                print('ENGINEERING',label,condition,summary,'elapsed',round(time.time()-start),flush=True)
    finally:close_pool()
    save(OUT/'engineering_v3_summary.json',dict(seconds=time.time()-start,results=results))
if __name__=='__main__':main()
