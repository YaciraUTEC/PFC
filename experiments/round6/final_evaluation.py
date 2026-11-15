"""Frozen final tests and a predeclared paired confirmation; never tune on test."""
import json,time,hashlib,datetime,subprocess,sys
from pathlib import Path
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from .evaluate import OUT,LEVELS,SEEDS,save,evaluate
from .train import selected_config
from .models import Agent,ARCHS
from .offpolicy import OffPolicy
from .phase1 import BASE
from .select import model_entries
from .replay import cache_baseline,evaluate_cached
ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_model(entry):return (Agent.load(entry['path'],'cpu') if entry['kind']=='ppo' else OffPolicy.load(entry['path'],'cpu')).double().eval()
def entries():return model_entries(2)+model_entries(3)
def label(e):return f"{e['kind']}_{e['arch']}_{e['schedule']}_{e['seed']}"
MODEL_WORKERS=2
CERTIFICATION_BATCH=64

def _worker_init():torch.set_num_threads(1)

def _model_job(job):
    path=Path(job['path'])
    if path.exists() and not job.get('force',False):return str(path)
    e=job['entry'];print(job.get('tag','MODEL_EVALUATION'),label(e),job['condition'],job['noise'],flush=True)
    m=load_model(e)
    result=evaluate_cached(m,job['rows'],Path(job['folder']),job['config'],job['condition'],job['noise'],
        ungated=job.get('ungated',False),certification_batch=job.get('certification_batch',CERTIFICATION_BATCH))
    save(path,dict(entry=e,**job.get('extra',{}),**result));return str(path)

def _run_jobs(jobs):
    pending=[j for j in jobs if j.get('force',False) or not Path(j['path']).exists()]
    if not pending:return
    with ProcessPoolExecutor(max_workers=MODEL_WORKERS,mp_context=mp.get_context('spawn'),initializer=_worker_init) as pool:
        for path in pool.map(_model_job,pending,chunksize=1):print('MODEL_EVALUATION_DONE',Path(path).name,flush=True)

def _job(e,path,rows,folder,config,condition,noise,**options):
    return dict(entry=e,path=str(path),rows=rows,folder=str(folder),config=config,condition=condition,noise=noise,**options)

def _compare_value(a,b):
    import math
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        return max((_compare_value(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,list):
        assert isinstance(b,list) and len(a)==len(b)
        return max((_compare_value(x,y) for x,y in zip(a,b)),default=0.)
    if isinstance(a,bool) or isinstance(b,bool):assert type(a)==type(b) and a==b;return 0.
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
        return abs(a-b)
    assert a==b,(a,b)
    return 0.

def verify_execution():
    marker=OUT/'evaluation_execution_verified.json'
    if marker.exists():assert json.loads(marker.read_text())['passed'];return
    cfg=selected_config();start=time.time();chosen=selected_learning_entries()
    wanted=[('ppo','mamba3'),('ppo','gtrxl'),('flashsac','mamba3'),('fql','gtrxl')]
    models=[next(e for e in chosen if (e['kind'],e['arch'],e['seed'])==(kind,arch,201)) for kind,arch in wanted]
    serial_jobs=[];parallel_jobs=[]
    for noise in [1.,2.]:
        rows,folder=cache_baseline('selected',cfg,'estimated',noise,range(611088,611092),[0.,.295,.296],role='execution_verification')
        for ungated in [False,True]:
            for e in models:
                name=f'{label(e)}_ungated{int(ungated)}_noise{noise:g}.json'
                shared=dict(ungated=ungated,force=True,tag='EXECUTION_VERIFICATION')
                serial_jobs.append(_job(e,OUT/'execution_verification'/'serial'/name,rows,folder,cfg,'estimated',noise,certification_batch=8,**shared))
                parallel_jobs.append(_job(e,OUT/'execution_verification'/'parallel'/name,rows,folder,cfg,'estimated',noise,certification_batch=CERTIFICATION_BATCH,**shared))
    t=time.time()
    for j in serial_jobs:_model_job(j)
    serial_seconds=time.time()-t;t=time.time();_run_jobs(parallel_jobs);parallel_seconds=time.time()-t
    largest=0.;cases=0
    for sj,pj in zip(serial_jobs,parallel_jobs):
        a=json.loads(Path(sj['path']).read_text());b=json.loads(Path(pj['path']).read_text())
        # Native solver times are operational wall measurements, not trajectories.
        clean=lambda rr:[{k:v for k,v in r.items() if k not in ['solve_seconds_mean','solve_seconds_p95']} for r in rr]
        largest=max(largest,_compare_value(clean(a['episodes']),clean(b['episodes'])));cases+=len(a['episodes'])
    save(marker,dict(passed=True,seconds=time.time()-start,serial_seconds=serial_seconds,parallel_seconds=parallel_seconds,
        logical_cases_each_execution=cases,model_workers=MODEL_WORKERS,physical_envs_per_model=8,certification_batch=CERTIFICATION_BATCH,
        routes=list(range(611088,611092)),models=[label(e) for e in models],noise=[1.,2.],levels=[0.,.295,.296],gated_and_ungated=True,
        max_numeric_difference=largest,relative_and_absolute_tolerance=1e-10,all_discrete_outcomes_identical=True,
        scope='Validation-only execution equivalence;no model selection or tuning;all final test routes remain closed.'))
    print('EVALUATION_EXECUTION_VERIFIED',cases,serial_seconds,parallel_seconds,largest,flush=True)

def freeze():
    path=OUT/'frozen_before_test.json'
    if path.exists():return json.loads(path.read_text())
    assert (OUT/'phase1_closed.json').exists() and (OUT/'phase3_conclusion.json').exists()
    receipt=json.loads((OUT/'tests_before_test_receipt.json').read_text());assert all(r['exit_code']==0 for r in receipt['jobs'])
    assert json.loads((OUT/'evaluation_execution_verified.json').read_text())['passed']
    checkpoints=[]
    for e in entries():
        p=Path(e['path']);complete=json.loads(p.with_name('complete.json').read_text());assert complete['steps']==524288
        assert complete['dataset_sha256']==json.loads((OUT/'dataset_fingerprint.json').read_text())['sha256']
        for name in ['best.pt','final.pt']:
            q=p.with_name(name);checkpoints.append(dict(path=str(q),sha256=sha(q)))
    previous=json.loads((ROOT/'runs/intervention_round5/frozen_before_test.json').read_text())
    for r in previous['checkpoints']:assert sha(r['path'])==r['sha256'],r['path']
    for p in sorted((OUT/'hpo').rglob('best.pt')):checkpoints.append(dict(path=str(p),sha256=sha(p),role='hyperparameter_pilot'))
    source={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'experiments').rglob('*.py'))}
    source.update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'gym_pybullet_drones').rglob('*.py'))})
    assets={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'gym_pybullet_drones/assets').rglob('*') if p.suffix.lower() in ['.urdf','.obj','.stl','.dae']}
    vendors={}
    for name in ['FlashSAC','fql']:
        cwd=ROOT.parent/'vendor'/name
        vendors[name]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()
    report_snapshot=OUT/'report_before_test.py';report_snapshot.write_bytes((ROOT/'experiments/round6/report.py').read_bytes())
    record=dict(report_original=dict(path=str(report_snapshot),sha256=sha(report_snapshot)),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoints=checkpoints,source_sha256=source,assets_sha256=assets,
        data_metadata_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'data').glob('route_*.json'))},
        vendor_commits=vendors,previous_checkpoints_verified=len(previous['checkpoints']),previous_checkpoints=previous['checkpoints'],
        configuration_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'phase1_conclusion.json',OUT/'phase2_conclusion.json',OUT/'phase3_conclusion.json',OUT/'PLAN.txt',OUT/'AMENDMENT_ENGINEERING.txt',OUT/'AMENDMENT_FINAL_EVALUATION.txt',OUT/'dataset_fingerprint.json',OUT/'data_audit.json',OUT/'hpo_selection.json',OUT/'hpo_protocol.json',OUT/'evaluation_execution_verified.json']},test_routes=list(range(612000,612060)),confirmation_routes=list(range(613000,613200)),
        model_evaluation_workers=MODEL_WORKERS,certification_episode_batch=CERTIFICATION_BATCH,secondary_nmpc_routes=list(range(612000,612012)),levels=LEVELS,models=len(entries()),
        protocol_notes=['NMPC secondary12routes/point due solver cost and failed48Hz requirement; primary learned/control tests60routes/point.',
            'Gate ablations16routes,5trainingseeds,29.5/29.6%,4condition/noise regimes.',
            'Shifted fault-clock sensitivity16routes,2clocks,estimated/noise2,29.5/29.6%.',
            'Confirmation is selected on validation; test cannot alter hyperparameters.'])
    save(path,record);return record

def primary_test():
    cfg=selected_config();models=entries()
    for condition in ['exact','estimated']:
        for noise in [1.,2.]:
            # Scientific physical reference and common selected-control cache.
            refpath=OUT/'test'/f'dslpid_reference_{condition}_noise{noise:g}.json'
            if not refpath.exists():save(refpath,evaluate(dict(BASE,noise_multiplier=noise),condition,range(612000,612060),LEVELS,workers=8))
            rows,folder=cache_baseline('selected',cfg,condition,noise,range(612000,612060),LEVELS)
            save(OUT/'test'/f'controller_selected_{condition}_noise{noise:g}.json',dict(episodes=rows))
            _run_jobs([_job(e,OUT/'test'/f'{label(e)}_{condition}_noise{noise:g}.json',rows,folder,cfg,condition,noise,tag='FINAL_TEST') for e in models])

def nmpc_secondary():
    control=json.loads((OUT/'phase1_conclusion.json').read_text());configs=control['all_configs'];dev=control['development']
    fast=max([n for n in dev if n.startswith('nmpc_fast')],key=lambda n:dev[n]['score'])
    names=list(dict.fromkeys([fast,control['best_slow_nmpc']]))
    save(OUT/'nmpc_test_plan.json',dict(names=names,routes=12,selected_from_development=True))
    for name in names:
        for condition in ['exact','estimated']:
            for noise in [1.,2.]:
                path=OUT/'test'/f'{name}_{condition}_noise{noise:g}.json'
                if path.exists():continue
                print('NMPC_SECONDARY_TEST',name,condition,noise,flush=True)
                save(path,dict(name=name,**evaluate(dict(configs[name],noise_multiplier=noise),condition,range(612000,612012),LEVELS,workers=8)))

def selected_learning_entries():
    p2=json.loads((OUT/'phase2_conclusion.json').read_text())
    return [e for e in entries() if e['kind']!='ppo' or e['schedule']==p2['schedule_by_arch'][e['arch']]]
def ablations():
    cfg=selected_config()
    for condition in ['exact','estimated']:
        for noise in [1.,2.]:
            rows,folder=cache_baseline('selected',cfg,condition,noise,range(612000,612060),LEVELS)
            subset=[r for r in rows if r['seed']<612016 and r['level'] in [.295,.296]]
            _run_jobs([_job(e,OUT/'ablations'/f'{label(e)}_ungated_{condition}_noise{noise:g}.json',subset,folder,cfg,condition,noise,ungated=True,extra=dict(ungated=True),tag='GATE_ABLATION') for e in selected_learning_entries()])

def shifted_clock():
    cfg=selected_config()
    for clock in [4.,8.]:
        cc=dict(cfg,fault_clock=clock)
        rows,folder=cache_baseline(f'selected_clock{clock:g}',cc,'estimated',2.,range(612000,612016),[.295,.296],role='sensitivity')
        save(OUT/'sensitivity'/f'baseline_clock{clock:g}.json',dict(episodes=rows))
        _run_jobs([_job(e,OUT/'sensitivity'/f'{label(e)}_clock{clock:g}.json',rows,folder,cc,'estimated',2.,extra=dict(clock=clock),tag='CLOCK_SENSITIVITY') for e in selected_learning_entries()])

def confirmation():
    cfg=selected_config();decision=json.loads((OUT/'phase3_conclusion.json').read_text());representative=decision['representative']
    candidates=[e for e in selected_learning_entries() if e['kind']==representative['kind'] and e['arch']==representative['arch']]
    for condition in ['exact','estimated']:
        for noise in [1.,2.]:
            rows,folder=cache_baseline('selected',cfg,condition,noise,range(613000,613200),[0.,.25,.295,.296],role='confirmation')
            save(OUT/'confirmation'/f'baseline_{condition}_noise{noise:g}.json',dict(episodes=rows))
            _run_jobs([_job(e,OUT/'confirmation'/f'{label(e)}_{condition}_noise{noise:g}.json',rows,folder,cfg,condition,noise,tag='CONFIRM') for e in candidates])

def integrity():
    frozen=json.loads((OUT/'frozen_before_test.json').read_text());changed=[]
    for p,h in frozen['source_sha256'].items():
        if p.replace('\\','/')=='experiments/round6/report.py':continue
        if sha(ROOT/p)!=h:changed.append(p)
    assert not changed,changed
    if frozen.get('report_original'):assert sha(frozen['report_original']['path'])==frozen['report_original']['sha256']
    for r in frozen['checkpoints']:assert sha(r['path'])==r['sha256'],r['path']
    for p,h in frozen['assets_sha256'].items():assert sha(ROOT/p)==h,p
    for p,h in frozen['configuration_sha256'].items():assert sha(ROOT/p)==h,p
    data_fingerprint=json.loads((OUT/'dataset_fingerprint.json').read_text())
    for p,h in data_fingerprint['files'].items():assert sha(OUT/p)==h,p
    for p,h in frozen['data_metadata_sha256'].items():assert sha(OUT/p)==h,p
    for r in frozen['previous_checkpoints']:assert sha(r['path'])==r['sha256'],r['path']
    return dict(source_files=len(frozen['source_sha256']),checkpoints=len(frozen['checkpoints']),physical_assets=len(frozen['assets_sha256']),passed=True)
def main():
    torch.set_num_threads(1);start=time.time()
    if not (OUT/'latency.json').exists():
        from .latency import main as latency
        latency()
    verify_execution()
    freeze()
    for name,function in [('primary',primary_test),('gate_ablations',ablations),('shifted_clock',shifted_clock),('confirmation',confirmation),('nmpc_secondary',nmpc_secondary)]:
        marker=OUT/f'final_{name}_complete.json'
        if marker.exists():continue
        t=time.time();function();save(marker,dict(seconds=time.time()-t,finished_epoch=time.time()));print('FINAL_STAGE_COMPLETE',name,flush=True)
    save(OUT/'final_integrity.json',integrity());save(OUT/'final_evaluation_complete.json',dict(seconds=time.time()-start,finished_epoch=time.time()))
if __name__=='__main__':main()
