"""Close the expanded engineering study before any student is trained."""
import json,time,hashlib
import numpy as np
from .phase1 import VARIANTS as ORIGINAL,run_job,summary,objective
from .bounded import BOUNDED
from .structured import VARIANTS as STRUCTURED
from .engineering import VARIANTS as ENGINEERING
from .fast import VARIANTS as FAST
from .evaluate import OUT,save

def files_ready():
 paths=[OUT/'phase1_conclusion.json',OUT/'phase1/bounded_configs.json',OUT/'phase1/structured_configs.json',OUT/'phase1/fast_configs.json',OUT/'phase1/engineering/nmpc_matched_indi_estimated_noise2.json']
 return all(p.exists() for p in paths)
def main():
 while not files_ready():time.sleep(5)
 start=time.time();allconfigs={**ORIGINAL,**BOUNDED,**STRUCTURED,**ENGINEERING,**FAST};dev={};rows={}
 # The engineering screen decides between INDI gains on development routes.
 matched=[]
 for name in ['nmpc_matched','nmpc_matched_indi']:
  rr=[]
  for c,n in [('exact',1),('estimated',2)]:
   data=json.loads((OUT/f'phase1/engineering/{name}_{c}_noise{n}.json').read_text())
   rr.extend(dict(r,condition=c,noise=float(n)) for r in data['episodes'])
  score=np.mean([r['stable_landing'] for r in rr if r['level']<=.295]);matched.append((float(score),name))
 best_matched=max(matched,key=lambda x:(x[0],x[1]=='nmpc_matched'))[1]
 names=list(dict.fromkeys([*ORIGINAL,*BOUNDED,*STRUCTURED,'reserve_only','coordinated_margin',*FAST,best_matched]))
 for name in names:
  rows[name]=run_job(name,allconfigs[name],'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.50]);dev[name]=summary(rows[name])
 # Quality and timing are separate findings. Full nonlinear slow NMPC remains
 # a scientific comparator but cannot be the 48Hz PC controller.
 eligible=[n for n in names if dev[n]['solve_p95_mean']<=1/48]
 finalists=sorted(eligible,key=lambda n:dev[n]['score'],reverse=True)[:3]
 finalists=list(dict.fromkeys(['dslpid_reference',*finalists]))
 val={};vr={}
 for name in finalists:
  vr[name]=run_job(name,allconfigs[name],'validation',range(611000,611024),[0.,.02,.25,.28,.29,.295,.296,.30,.35,.40,.50]);val[name]=summary(vr[name])
 winner=max(finalists,key=lambda n:val[n]['score'])
 keys=lambda rr:{(r['condition'],r['noise'],r['level'],r['seed']):r for r in rr}
 br=keys(vr['dslpid_reference']);wr=keys(vr[winner]);gains=sum(wr[k]['stable_landing'] and not br[k]['stable_landing'] for k in br);losses=sum(br[k]['stable_landing'] and not wr[k]['stable_landing'] for k in br)
 mild_losses=sum(br[k]['stable_landing'] and not wr[k]['stable_landing'] and br[k]['level']<=.25 for k in br)
 if mild_losses:
  winner='dslpid_reference';gains=losses=0
 conclusion=dict(selected=winner,config=allconfigs[winner],development=dev,validation=val,gains=gains,losses=losses,mild_losses_of_initial_winner=mild_losses,
  eligible_at_48hz=eligible,best_slow_nmpc=best_matched,best_nmpc_quality=max([n for n in dev if n.startswith('nmpc')],key=lambda n:dev[n]['score']),all_configs=allconfigs,
  notes=['No model/test data used for control selection.','INDI gain0 is labeled NMPC without INDI.','Structured observer assumes one faulty rotor.','Kinematic governor is not a nonlinear feasibility proof.','Latency under concurrent CPU load is reported; dedicated timing follows.'],finished_epoch=time.time(),seconds=time.time()-start)
 save(OUT/'phase1_conclusion.json',conclusion);save(OUT/'phase1_closed.json',dict(selected=winner,finished_epoch=time.time()))
 lines=['FASE 1 CERRADA','Control seleccionado para48Hz: '+winner,'Ganancias pareadas: '+str(gains),'Perdidas pareadas: '+str(losses),
  'El criterio de aterrizaje estricto no cambia.','NMPC lento se conserva como comparador, no como control viable a48Hz.','Prueba final permanece reservada.',json.dumps(conclusion,indent=2)]
 (OUT/'CONCLUSION_FASE1.txt').write_text('\n'.join(lines),encoding='utf-8')
 print('FINAL_PHASE1_CONCLUSION',winner,gains,losses,flush=True)
if __name__=='__main__':main()
