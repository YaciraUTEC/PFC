"""Sequential milestone closure, parallel independent train/calibration data."""
import time,sys,subprocess
from .common import ROOT,OUT,save,read,close
from ..round7.power import awake

def stage(name,args,complete=None):
 if complete and (OUT/complete).exists():return
 start=time.time();save(OUT/'status.json',dict(stage=name,status='running'));path=OUT/(name+'.log')
 with path.open('a',encoding='utf-8') as f:
  p=subprocess.Popen([sys.executable,'-u','-B',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));save(OUT/(name+'_process.json'),dict(pid=p.pid,args=args,started_epoch=start));code=p.wait()
 save(OUT/(name+'_receipt.json'),dict(seconds=time.time()-start,exit_code=code,args=args))
 if code:raise RuntimeError(name+' failed: '+str(path))
 print('STAGE_COMPLETE',name,round(time.time()-start),flush=True)

def main():
 while not (OUT/'h1_flight_pilot_conclusion.json').exists():time.sleep(5)
 with awake():
  stage('h2_natural_option_semantics',['-m','experiments.round8.natural_pilot'],'h2_natural_option_semantics_conclusion.json')
  # Pilot two complementary strata before committing to the complete acquisition.
  from .data import collect_one
  for index in [0,5]:collect_one(index,'train')
  close('h2_acquisition_pilot','Exact/estimated pilot strata preserve their natural base caller and all query prefixes; strict outcomes and actual projected actions verified.',groups=[0,5])
  jobs=[];save(OUT/'status.json',dict(stage='h2_base_data',status='running',groups_per_split=120))
  for split in ['train','calibration']:
   name='h2_base_data_'+split;marker=OUT/('h2_base_options_v1_'+split+'_conclusion.json')
   if marker.exists():continue
   path=OUT/(name+'.log');handle=path.open('a',encoding='utf-8');before=time.time();args=['-m','experiments.round8.data','--split',split,'--groups','120','--workers','8'];p=subprocess.Popen([sys.executable,'-u','-B',*args],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));jobs.append((name,p,handle,before,args));save(OUT/(name+'_process.json'),dict(pid=p.pid,args=args,started_epoch=before))
  try:
   for name,p,handle,before,args in jobs:
    code=p.wait();handle.close();save(OUT/(name+'_receipt.json'),dict(seconds=time.time()-before,exit_code=code,args=args))
    if code:raise RuntimeError(name+' failed')
  finally:
   for name,p,handle,before,args in jobs:
    if p.poll() is None:p.terminate();p.wait()
    if not handle.closed:handle.close()
  close('h2_base_data','Both naturally reconstructed train/calibration datasets complete before fitting; no changes to baseline or landing contract.',groups_per_split=120)
  for seed in [501,502,503]:stage('h2_factor_fit_'+str(seed),['-m','experiments.round8.fit','--seed',str(seed),'--folder',str(OUT/'data/base_options_v1'),'--tag','base_options_v1'])
  close('h2_factor_fits','Three seeds fitted to natural base-state native/raw options with explicit post-saturation inputs.')
  save(OUT/'status.json',dict(stage='h2_ready_for_review',status='running'))
if __name__=='__main__':main()
