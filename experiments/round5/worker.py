"""Two bounded training workers. Each stage exits on failure and can restart.
Incomplete policies restart from their unchanged BC and initial reward artifact;
completed jobs are skipped. No midway simulator-state resumption is claimed.
"""
import argparse,json,subprocess,sys,time,datetime
from pathlib import Path
from .evaluate import OUT
from ..run import save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1],required=True);args=parser.parse_args();worker=args.worker
    while not (OUT/'preparation_complete.json').exists():time.sleep(5)
    start=time.time();receipts=[]
    stages=[('residual_policies',['-m','experiments.round5.train','--phase','train','--worker',str(worker)]),
        ('learned_rewards',['-m','experiments.round5.train','--phase','rewards','--worker',str(worker)]),
        ('learned_observers',['-m','experiments.round5.bc','--phase','observers','--worker',str(worker)])]
    for label,command in stages:
        before=time.time();log=OUT/f'worker_{worker}_{label}.log'
        save(OUT/f'worker_{worker}_status.json',dict(stage=label,status='running',started_epoch=before))
        with log.open('a',encoding='utf-8') as f:
            f.write('\nATTEMPT_START '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');f.flush()
            result=subprocess.run([sys.executable,*command],cwd=Path.cwd(),stdout=f,stderr=subprocess.STDOUT)
        receipt=dict(stage=label,started_epoch=before,finished_epoch=time.time(),seconds=time.time()-before,exit_code=result.returncode)
        receipts.append(receipt);save(OUT/f'worker_{worker}_receipts.json',receipts)
        if result.returncode:
            save(OUT/f'worker_{worker}_status.json',dict(stage=label,status='failed',log=str(log),exit_code=result.returncode));raise RuntimeError(f'Worker{worker}:{label}failed;see{log}')
        print('WORKER_STAGE_COMPLETE',worker,label,'seconds',round(time.time()-before),flush=True)
    save(OUT/f'worker_{worker}_complete.json',dict(started_epoch=start,finished_epoch=time.time(),seconds=time.time()-start,stages=receipts))
    save(OUT/f'worker_{worker}_status.json',dict(stage='all_training_complete',status='complete'))
if __name__=='__main__':main()
