"""Durable single-stage receipts, no implicit transition into final test."""
import sys,time,subprocess
from .common import ROOT,OUT,save,read

def stage(label,args,complete=None):
    if complete is not None and (OUT/complete).exists():return
    started=time.time();path=OUT/(label+'.log')
    save(OUT/'status.json',dict(stage=label,status='running',started_epoch=started))
    with path.open('a',encoding='utf-8') as log:
        log.write('\nATTEMPT '+str(started)+'\n');log.flush()
        process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        save(OUT/(label+'_process.json'),dict(pid=process.pid,started_epoch=started,args=args))
        while process.poll() is None:time.sleep(5)
    receipt=dict(started_epoch=started,finished_epoch=time.time(),seconds=time.time()-started,exit_code=process.returncode,log=str(path),args=args)
    save(OUT/(label+'_receipt.json'),receipt)
    if process.returncode:
        save(OUT/'status.json',dict(stage=label,status='failed',receipt=receipt));raise RuntimeError(label+' failed; '+str(path))
    print('STAGE_COMPLETE',label,round(receipt['seconds'],1),flush=True)



def parallel_stages(jobs):
    """Independent train/calibration acquisitions; one writer for shared status."""
    active=[]
    save(OUT/'status.json',dict(stage='parallel_acquisition',status='running',jobs=[j[0] for j in jobs]))
    try:
        for label,args,complete in jobs:
            if complete is not None and (OUT/complete).exists():continue
            started=time.time();path=OUT/(label+'.log');handle=path.open('a',encoding='utf-8');handle.write('\nATTEMPT '+str(started)+'\n');handle.flush()
            process=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save(OUT/(label+'_process.json'),dict(pid=process.pid,started_epoch=started,args=args));active.append(dict(label=label,args=args,started=started,path=path,handle=handle,process=process))
        while active:
            for item in active[:]:
                process=item['process']
                if process.poll() is None:continue
                item['handle'].close();receipt=dict(started_epoch=item['started'],finished_epoch=time.time(),seconds=time.time()-item['started'],exit_code=process.returncode,log=str(item['path']),args=item['args'],parallel_independent_acquisition=True)
                save(OUT/(item['label']+'_receipt.json'),receipt);active.remove(item)
                if process.returncode:raise RuntimeError(item['label']+' failed; '+str(item['path']))
                print('STAGE_COMPLETE',item['label'],round(receipt['seconds'],1),flush=True)
            if active:time.sleep(5)
    finally:
        for item in active:
            if item['process'].poll() is None:item['process'].terminate();item['process'].wait()
            item['handle'].close()
