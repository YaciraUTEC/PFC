"""Final phase-aware braking study, independently validated before training."""
import json,time
from .phase1 import BASE,run_job,summary
from .evaluate import OUT,save
from .structured import VARIANTS as STRUCT
NEW={}
for physical in [False,True]:
 name='landing_physical' if physical else 'landing_coordinated'
 NEW[name]=dict(BASE,control_kind='coordinated',estimator='structured',manager='coordinated',filter_tau=0.,yaw_priority=1.,emergency='legacy',reserve=.001,
    yaw_acceleration_limit=3.,coordinate_thrust=True,landing_only=True,supervisor_options=dict(capacity_envelope=physical,planned_yaw_acceleration=3.))
def main():
 dev={};validation={};rows={}
 for name,cfg in NEW.items():dev[name]=summary(run_job(name,cfg,'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.5]))
 # New routes prevent tuning to the already inspected validation sample.
 candidates={'dslpid_reference':BASE,'structured_governor':STRUCT['structured_governor'],**NEW}
 for name,cfg in candidates.items():
  rows[name]=run_job(name,cfg,'physical_validation',range(611032,611056),[0.,.02,.25,.28,.29,.295,.296,.30,.35,.40,.50]);validation[name]=summary(rows[name])
 while not (OUT/'phase1_closed.json').exists() or 'FINAL_PHASE1_CONCLUSION' not in (OUT/'decide_control.log').read_text(errors='replace'):time.sleep(5)
 old=json.loads((OUT/'phase1_conclusion.json').read_text());winner=max(candidates,key=lambda n:validation[n]['score'])
 keyed=lambda rr:{(r['condition'],r['noise'],r['level'],r['seed']):r for r in rr}
 reference=keyed(rows['dslpid_reference']);chosen=keyed(rows[winner]);mild_losses=sum(reference[k]['stable_landing'] and not chosen[k]['stable_landing'] and reference[k]['level']<=.25 for k in reference)
 if mild_losses:winner='dslpid_reference';chosen=reference
 gains=sum(chosen[k]['stable_landing'] and not reference[k]['stable_landing'] for k in reference);losses=sum(reference[k]['stable_landing'] and not chosen[k]['stable_landing'] for k in reference)
 old['preceding_control_decision']=dict(selected=old['selected'],validation=old['validation'])
 old.update(selected=winner,config=candidates[winner],validation=validation,gains=gains,losses=losses,physical_validation_routes=list(range(611032,611056)),finished_epoch=time.time())
 old['all_configs'].update(NEW);old['development'].update(dev)
 save(OUT/'phase1_conclusion.json',old)
 text='FASE1 CERRADA TRAS VALIDACION FISICA INDEPENDIENTE\nControl elegido: '+winner+'\nGanancias/perdidas: '+str(gains)+'/'+str(losses)+'\n'+json.dumps(validation,indent=2)
 (OUT/'CONCLUSION_FASE1.txt').write_text(text,encoding='utf-8')
 save(OUT/'phase1_physical_closed.json',dict(selected=winner,gains=gains,losses=losses,finished_epoch=time.time()))
 print('PHYSICAL_PHASE1_CONCLUSION',winner,gains,losses,flush=True)
if __name__=='__main__':main()
