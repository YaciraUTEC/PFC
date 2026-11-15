"""Sequential phase gates and durable subprocess receipts. Safe to restart."""
import json,sys,time,subprocess,datetime
from pathlib import Path
from .evaluate import OUT,save
ROOT=Path(__file__).resolve().parents[2]
def stage(label,commands):
    start=time.time();jobs=[]
    save(OUT/'status.json',dict(stage=label,status='running',started_epoch=start))
    for j,cmd in enumerate(commands):
        logfile=OUT/f'{label}_{j}.log';f=logfile.open('a',encoding='utf-8');f.write('\nATTEMPT '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');f.flush()
        p=subprocess.Popen([sys.executable,*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));jobs.append((p,f,logfile))
    while any(p.poll() is None for p,f,l in jobs):time.sleep(5)
    receipts=[]
    for p,f,l in jobs:f.close();receipts.append(dict(log=str(l),exit_code=p.returncode))
    save(OUT/f'{label}_receipt.json',dict(started_epoch=start,finished_epoch=time.time(),seconds=time.time()-start,jobs=receipts))
    if any(r['exit_code'] for r in receipts):
        save(OUT/'status.json',dict(stage=label,status='failed',jobs=receipts));raise RuntimeError(label+' failed;see logs')
    print('CAMPAIGN_STAGE_COMPLETE',label,round(time.time()-start),flush=True)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'campaign_started.json').exists():save(OUT/'campaign_started.json',dict(started_epoch=time.time(),python=sys.executable))
    while not (OUT/'phase1_closed.json').exists():time.sleep(5)
    if not (OUT/'data_complete.json').exists():
        stage('data',[['-m','experiments.round6.prepare_data']]);save(OUT/'data_complete.json',dict(finished_epoch=time.time()))
    if not (OUT/'expert_data_complete.json').exists():
        stage('mpc_expert_data',[['-m','experiments.round6.mpc_teacher_data']])
    if not (OUT/'rescue_expert_data_complete.json').exists():
        stage('rescue_expert_data',[['-m','experiments.round6.rescue_teacher_data']])
    if not (OUT/'training_smoke_complete.json').exists():
        stage('training_smoke',[['-m','experiments.round6.train','--smoke','--budget','4096','--workers','1']]);save(OUT/'training_smoke_complete.json',dict(finished_epoch=time.time()))
    if not (OUT/'phase2_conclusion.json').exists():
        stage('phase2_training',[['-m','experiments.round6.train','--worker',str(j)] for j in [0,1]])
        stage('phase2_selection',[['-m','experiments.round6.select','--phase','2']])
    if not (OUT/'phase3_conclusion.json').exists():
        stage('phase3_training',[['-m','experiments.round6.train_offpolicy','--worker',str(j)] for j in [0,1]])
        stage('phase3_selection',[['-m','experiments.round6.select','--phase','3']])
    stage('tests_before_test',[['-m','pytest','experiments/tests','-q','--disable-warnings']])
    stage('final_evaluation',[['-m','experiments.round6.final_evaluation']])
    stage('final_report',[['-m','experiments.round6.report']])
    save(OUT/'status.json',dict(stage='awaiting_visual_review',status='evaluation_complete'))
if __name__=='__main__':main()
