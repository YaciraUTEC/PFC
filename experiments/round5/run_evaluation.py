"""Finish every predeclared evaluation and audit before reporting completion."""
import sys,time,json,subprocess,datetime,ctypes
from pathlib import Path
from .evaluate import OUT
from ..run import save

def stage(label,module):
    start=time.time();save(OUT/'evaluation_pipeline_status.json',dict(stage=label,status='running',started_epoch=start))
    with (OUT/f'{label}.log').open('a',encoding='utf-8') as f:
        f.write('\nSTAGE_START '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');f.flush()
        result=subprocess.run([sys.executable,'-m',module],cwd=Path.cwd(),stdout=f,stderr=subprocess.STDOUT)
    receipt=dict(started_epoch=start,finished_epoch=time.time(),seconds=time.time()-start,exit_code=result.returncode)
    save(OUT/f'{label}_receipt.json',receipt)
    if result.returncode:
        save(OUT/'evaluation_pipeline_status.json',dict(stage=label,status='failed',exit_code=result.returncode));raise RuntimeError(f'{label}failed;see{OUT/label}.log')
    print('EVALUATION_STAGE_COMPLETE',label,'seconds',round(time.time()-start),flush=True)

def main():
    # Temporary system-awake request for this long,explicitly requested campaign.
    # The display may sleep; this is automatically released when the process ends.
    awake=False
    try:
        if sys.platform=='win32':awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        save(OUT/'evaluation_pipeline_status.json',dict(stage='waiting_for_all_training',status='waiting',temporary_awake_request=awake))
        while True:
            if all((OUT/f'worker_{w}_complete.json').exists() for w in [0,1]):break
            for w in [0,1]:
                path=OUT/f'worker_{w}_status.json'
                if path.exists() and json.loads(path.read_text()).get('status')=='failed':raise RuntimeError(f'Training worker{w}failed;do not evaluate incomplete models')
            time.sleep(10)
        for label,module in [('runtime_selection','experiments.round5.select_runtime'),('inference_audit','experiments.round5.inference_audit'),
            ('final_campaign','experiments.round5.campaign'),('production_confirmation','experiments.round5.confirm'),
            ('sensitivity','experiments.round5.sensitivity'),('latency','experiments.round5.latency'),
            ('analysis','experiments.round5.analysis'),('recommended_replay','experiments.round5.deployment'),('report','experiments.round5.report'),('final_integrity_audit','experiments.round5.final_audit')]:stage(label,module)
        save(OUT/'ALL_COMPLETE.json',dict(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report=str(OUT/'INFORME_FINAL_RONDA5.txt')))
        save(OUT/'evaluation_pipeline_status.json',dict(stage='all_complete',status='complete'))
    finally:
        if awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
