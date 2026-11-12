"""Frozen, independent final evaluation with actor-level deduplication."""
import argparse,json,hashlib,time,datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .inventory import groups,core,sha,actor_hash,always_off,config_for,cached_evaluation,TEST_LEVELS,STRESS
from .models import load_agent,ARCHS
from .evaluate import OUT,close_pool
from ..run import save

ROOT=Path(__file__).resolve().parents[2]
def freeze():
    target=OUT/'frozen_before_test.json'
    if target.exists():return json.loads(target.read_text())
    for worker in [0,1]:assert (OUT/f'worker_{worker}_complete.json').exists()
    assert (OUT/'runtime_selection_complete.json').exists()
    assert (OUT/'oracle_proposal_audit_complete.json').exists()
    assert json.loads((OUT/'inference_audit_complete.json').read_text())['passed']
    assert json.loads((OUT/'tests_final_receipt.json').read_text())['exit_code']==0
    from .confirm import specifications
    from .analysis import comparison_pairs
    planned_endpoints=3*len(comparison_pairs([r['id'] for r in specifications()]))
    assert planned_endpoints<=36,'Update the prospective endpoint allocation before opening any final test'
    checkpoints=[]
    for group in groups():
        for entry in group['entries']:
            directory=Path(entry['directory']);needed=['bc.pt'] if group['kind']=='observer' else ['bc.pt','best.pt','final.pt']
            if group['kind']=='reward' and group['name'] in ['meirl','rime']:needed+=['reward.pt','reward_initial.pt']
            for name in needed:
                path=directory/name;assert path.exists(),str(path);checkpoints.append(dict(path=str(path),sha256=sha(path)))
            if group['kind']!='observer':
                receipt=json.loads((directory/'complete.json').read_text());assert receipt['steps']==524288
            else:assert json.loads((directory/'bc_timing.json').read_text())['updates']==1200
    checkpoints.append(dict(path=str(OUT/'judge.pt'),sha256=sha(OUT/'judge.pt')))
    for mode in ['residual','effectiveness']:
        for arch in ARCHS:
            pilot=OUT/'pilots'/mode/arch/'bc.pt';assert pilot.exists();checkpoints.append(dict(path=str(pilot),sha256=sha(pilot),role='data_collection_pilot'))
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(set((ROOT/'experiments').rglob('*.py'))|set((ROOT/'gym_pybullet_drones').rglob('*.py')))}
    configs={str(p):sha(p) for p in [OUT/'core_selection.json',OUT/'runtime_selection_complete.json',OUT/'protocol_initial.json',OUT/'objective_and_sensor_amendment.json',OUT/'final_evaluation_protocol.json',OUT/'confirmation_case_plan.json',OUT/'gate_training_amendment.json',OUT/'structural_challenger.json',OUT/'value_scale_amendment.json',OUT/'pre_freeze_correctness_amendment.json',OUT/'runtime_environment.json',OUT/'training_horizon_amendment.json',OUT/'statistical_precision_amendment.json',OUT/'exact_replay_amendment.json',OUT/'observer_fallback_selection.json',OUT/'observer_confirmation_amendment.json',OUT/'oracle_diagnostic_amendment.json']+[OUT/'hpo'/f'{a}_selection.json' for a in ARCHS]+list((ROOT/'experiments/round5/configs').rglob('*.json'))}
    old=ROOT/'runs/controlled_landing_round4';oldmanifest=json.loads((old/'frozen_before_test.json').read_text()) if (old/'frozen_before_test.json').exists() else dict(checkpoints=[])
    for entry in oldmanifest['checkpoints']:assert sha(old/entry['path'])==entry['sha256']
    assets={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'gym_pybullet_drones').rglob('*') if p.suffix.lower() in ['.urdf','.obj','.stl','.dae']}
    record=dict(planned_confirmation_endpoints=planned_endpoints,physical_asset_hashes=assets,presentation_only_paths=['experiments/round5/report.py','experiments/round5/plots.py'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoints=checkpoints,source_hashes=sources,configuration_hashes=configs,
        previous_frozen_models_verified=len(oldmanifest['checkpoints']),test_routes=list(range(529000,529060)),confirmation_routes=list(range(531000,532000)))
    save(target,record);return record

def verify(frozen):
    for path,digest in frozen.get('physical_asset_hashes',{}).items():assert sha(ROOT/path)==digest,path
    for r in frozen['checkpoints']:assert sha(r['path'])==r['sha256'],r['path']
    for path,digest in frozen['configuration_hashes'].items():assert sha(path)==digest,path
    for path,digest in frozen['source_hashes'].items():
        if path.replace('\\','/') not in frozen.get('presentation_only_paths',[]):assert sha(ROOT/path)==digest,path

def requests():
    result=[]
    def add(label,path,cfg,levels,conditions=('exact','estimated'),threshold=0.,shadow=False,**meta):
        result.append(dict(label=label,path=None if path is None else str(path),config=cfg,levels=levels,conditions=list(conditions),threshold=threshold,shadow=shadow,**meta))
    base=dict(estimator='original',readiness=False,emergency='legacy',reserve=.005);selected=core()
    variants=dict(original=base,selected=selected,balanced_governor=dict(selected,emergency='balanced'),ready_only=dict(base,readiness=True),observer_only=dict(base,estimator=selected['estimator']),
        yaw_only=dict(base,emergency='damped'),reserve_only=dict(base,reserve=selected['reserve']),without_observer=dict(selected,estimator='original'),
        without_ready=dict(selected,readiness=False),without_yaw=dict(selected,emergency='legacy'),without_reserve=dict(selected,reserve=.005))
    for label,cfg in variants.items():add('baseline_'+label,None,cfg,TEST_LEVELS+STRESS,kind='baseline',name=label,seed=None)
    choices=json.loads((OUT/'runtime_selection_complete.json').read_text())['groups'];choice={r['group']['id']:r['chosen_strategy'] for r in choices}
    for group in groups():
        strategy=choice[group['id']]
        for entry in group['entries']:
            label=f"{group['id']}_{entry['seed']}"
            raw=core();raw.update(intervention=group['intervention'])
            add(label+'_raw',entry['path'],raw,TEST_LEVELS+STRESS,kind=group['kind'],name=group['name'],seed=entry['seed'],mode='raw')
            add(label+'_selected',entry['path'],config_for(group,strategy),TEST_LEVELS+STRESS,threshold=strategy['threshold'],shadow=strategy.get('shadow',False),kind=group['kind'],name=group['name'],seed=entry['seed'],mode='selected',strategy=strategy)
            if group['kind']=='residual':add(label+'_bc',Path(entry['directory'])/'bc.pt',raw,TEST_LEVELS,kind=group['kind'],name=group['name'],seed=entry['seed'],mode='bc')
            if group['kind']!='observer':add(label+'_final',Path(entry['directory'])/'final.pt',raw,[.28,.29,.295],conditions=('estimated',),kind=group['kind'],name=group['name'],seed=entry['seed'],mode='final')
    return result

def deduplicate(reqs):
    unique={};logical=[]
    for r in reqs:
        m=None if r['path'] is None else load_agent(r['path']);certificate=None
        if m is not None:m.gate_logit_threshold=r['threshold'];certified,upper=always_off(m);certificate=dict(certified=certified,upper=upper,threshold=r['threshold'])
        zero=m is None or r['config'].get('gate_mode')=='off' or (certificate['certified'] and not r['shadow'])
        cfg=dict(r['config'])
        if zero:cfg.update(intervention='residual',gate_mode='learned',judge_path=None)
        actor='ZERO' if zero else ('PROPOSAL:' if r['shadow'] else '')+actor_hash(m)
        key=hashlib.sha256(json.dumps(dict(actor=actor,config=cfg),sort_keys=True).encode()).hexdigest()
        if key not in unique:unique[key]=dict(key=key,path=None if zero else r['path'],config=cfg,threshold=r['threshold'],shadow=bool(r['shadow'] and not zero),condition_levels={})
        for condition in r['conditions']:unique[key]['condition_levels'].setdefault(condition,set()).update(r['levels'])
        logical.append(dict(r,unique_key=key,zero_policy=zero,certificate=certificate))
        del m
    jobs=[]
    for job in unique.values():job['condition_levels']={k:sorted(v) for k,v in job['condition_levels'].items()};jobs.append(job)
    return jobs,logical

def job(spec):
    torch.set_num_threads(1);start=time.time();outputs=[]
    try:
        for condition,levels in spec['condition_levels'].items():
            path=OUT/'test'/'unique'/f"{spec['key']}_{condition}.json"
            rows=cached_evaluation(spec['path'],path,condition,range(529000,529060),levels,spec['config'],threshold=spec['threshold'],proposal_only=spec['shadow'])
            outputs.append(dict(condition=condition,path=str(path),cases=len(rows)))
    finally:close_pool()
    return dict(key=spec['key'],outputs=outputs,seconds=time.time()-start)

def main():
    frozen=freeze();verify(frozen);start=time.time();jobs,logical=deduplicate(requests());save(OUT/'final_test_manifest.json',dict(unique_jobs=jobs,logical=logical))
    done=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(job,spec) for spec in jobs]
        for f in as_completed(futures):
            result=f.result();done.append(result);save(OUT/'evaluation_progress.json',dict(completed=len(done),total=len(jobs),jobs=done,seconds=time.time()-start))
            print('FINAL_EVALUATION',len(done),'/',len(jobs),'cases',sum(r['cases'] for r in result['outputs']),flush=True)
    files=0;cases=0
    for item in logical:
        for condition in item['conditions']:
            source=OUT/'test'/'unique'/f"{item['unique_key']}_{condition}.json";rows=[r for r in json.loads(source.read_text()) if r['level'] in item['levels']]
            expected={(l,s) for l in item['levels'] for s in range(529000,529060)}
            assert {(r['level'],r['seed']) for r in rows}==expected and len(rows)==len(expected)
            target=OUT/'test'/f"{item['label']}_{condition}.json";save(target,rows);save(target.with_suffix('.meta.json'),dict(source=str(source),logical_request=item))
            files+=1;cases+=len(rows)
    verify(frozen);save(OUT/'final_test_complete.json',dict(seconds=time.time()-start,unique_policies=len(jobs),logical_files=files,logical_cases=cases,frozen_verified=True))
if __name__=='__main__':main()
