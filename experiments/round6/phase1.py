"""Phase1 closes a validation decision before either neural curriculum starts."""
import argparse,json,time
import numpy as np
from collections import Counter
from .evaluate import evaluate,save,OUT,rates

BASE=dict(control_kind='dslpid',manager='legacy',readiness=False,estimator='original',emergency='legacy',reserve=.005,gate_mode='off')
VARIANTS={
 'dslpid_reference':dict(BASE),
 'previous_full':dict(BASE,readiness=True,estimator='adaptive',emergency='damped',reserve=.003),
 'governor_only':dict(BASE,manager='coordinated'),
 'coordinated_raw':dict(BASE,control_kind='coordinated',manager='coordinated',rate_braking=True,filter_tau=0.,yaw_priority=1.,emergency='damped'),
 'coordinated_filtered':dict(BASE,control_kind='coordinated',manager='coordinated',rate_braking=True,filter_tau=.04,yaw_priority=1.,emergency='damped'),
 'coordinated_priority':dict(BASE,control_kind='coordinated',manager='coordinated',rate_braking=True,filter_tau=.04,yaw_priority=2.,emergency='damped'),
 'coordinated_adaptive':dict(BASE,control_kind='coordinated',manager='coordinated',rate_braking=True,filter_tau=.04,yaw_priority=1.,emergency='damped',estimator='adaptive'),
 'nmpc_original':dict(BASE,control_kind='nmpc',nmpc_options={},fault_braking=False,filter_tau=0.),
 'nmpc_coordinated':dict(BASE,control_kind='nmpc',manager='coordinated',fault_braking=True,mpc_stride=2,filter_tau=0.,nmpc_options=dict(horizon=12,solver_maxiter=25,solver_time_limit=.03,angular_rate_weights=[.5,.5,.3],fault_yaw_rate_fraction=1.)),
}
CONDITIONS=[('exact',1.),('estimated',1.),('exact',2.),('estimated',2.)]
def run_job(name,config,stage,seeds,levels,conditions=CONDITIONS):
    records=[]
    for condition,noise in conditions:
        path=OUT/'phase1'/stage/f'{name}_{condition}_noise{noise:g}.json'
        if not path.exists():
            cfg=dict(config,noise_multiplier=noise);start=time.time()
            print('PHASE1_START',stage,name,condition,noise,flush=True)
            result=evaluate(cfg,condition,seeds,levels,workers=6)
            save(path,dict(name=name,condition=condition,noise=noise,config=cfg,**result))
            print('PHASE1_DONE',stage,name,condition,noise,'seconds',round(time.time()-start),rates(result['episodes']),flush=True)
        data=json.loads(path.read_text());records.extend(dict(r,condition=condition,noise=noise) for r in data['episodes'])
    return records

def objective(rows):
    # Protection is scored explicitly; a nominal-only severe gain cannot win.
    severe=[r for r in rows if .28<=r['level']<=.296 and r['condition']=='estimated']
    mild=[r for r in rows if r['level']<=.25]
    exact=[r for r in rows if r['condition']=='exact' and r['level']<=.296]
    cells=[np.mean([r['stable_landing'] for r in severe if r['noise']==n]) for n in [1.,2.]]
    return float(min(cells)+.25*np.mean(cells)-.75*(1-np.mean([r['stable_landing'] for r in mild]))-.5*(1-np.mean([r['stable_landing'] for r in exact])))

def summary(rows):
    return dict(score=objective(rows),cells={f'{c}_noise{n:g}':rates([r for r in rows if r['condition']==c and r['noise']==n]) for c,n in CONDITIONS},
        outcomes=dict(Counter(r['outcome'] for r in rows)),solve_p95_mean=float(np.mean([r['solve_seconds_p95'] for r in rows])),
        mean_pause_seconds=float(np.mean([r['descent_paused_seconds'] for r in rows])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--group',choices=['pid','nmpc','all'],default='all');a=p.parse_args()
    if a.smoke:
        for name in ['dslpid_reference','coordinated_filtered','nmpc_coordinated']:
            run_job(name,VARIANTS[name],'smoke_v2',range(610000,610002),[0.,.295],conditions=[('estimated',2.)])
        return
    names=[n for n in VARIANTS if a.group=='all' or n.startswith('nmpc')==(a.group=='nmpc')]
    dev={}
    for name in names:dev[name]=summary(run_job(name,VARIANTS[name],'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.50]))
    save(OUT/'phase1'/f'development_{a.group}.json',dev)
    if a.group!='all':return
    selected=sorted(dev,key=lambda n:dev[n]['score'],reverse=True)[:3]
    selected=list(dict.fromkeys(['dslpid_reference',*selected]))
    # Both NMPC versions are measured over the high-fault range, even if rejected.
    for name in [n for n in VARIANTS if n.startswith('nmpc')]:
        run_job(name,VARIANTS[name],'high_fault_diagnostic',range(610008,610012),[.30,.35,.40,.50,1.],conditions=[('exact',1.),('estimated',2.)])
    val={};rows={}
    for name in selected:
        rows[name]=run_job(name,VARIANTS[name],'validation',range(611000,611024),[0.,.02,.25,.28,.29,.295,.296,.30,.35,.40,.50]);val[name]=summary(rows[name])
    winner=max(val,key=lambda n:val[n]['score'])
    reference=rows['dslpid_reference']; keyed=lambda rr:{(r['condition'],r['noise'],r['seed'],r['level']):r for r in rr}
    br=keyed(reference);wr=keyed(rows[winner]);gains=sum(wr[k]['stable_landing'] and not br[k]['stable_landing'] for k in br);losses=sum(br[k]['stable_landing'] and not wr[k]['stable_landing'] for k in br)
    conclusion=dict(selected=winner,config=VARIANTS[winner],development=dev,validation=val,gains=gains,losses=losses,
        interpretation='Validation decision only; test remains unopened. Strict touchdown criteria unchanged. Static governor is not an MPC feasibility proof.',
        next_phase='Compare noise curricula with Mamba-3 and GTrXL on selected control. No test-driven tuning.',finished_epoch=time.time())
    save(OUT/'phase1_conclusion.json',conclusion)
    lines=['FASE 1 CERRADA','Control seleccionado: '+winner,'Ganancias pareadas: '+str(gains),'Perdidas pareadas: '+str(losses),
        'Se conserva aterrizaje estricto. Vuelo giratorio no equivale a aterrizaje.','Seleccion por validacion; prueba final todavia reservada.',json.dumps(val,indent=2)]
    (OUT/'CONCLUSION_FASE1.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('PHASE1_CONCLUSION',winner,gains,losses,flush=True)
if __name__=='__main__':main()
