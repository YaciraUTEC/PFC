"""Required completion audit:coverage,finite weights,freeze,replay and tests."""
import json,math,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
from .evaluate import OUT
from .campaign import freeze,verify
from .inventory import groups,sha,TEST_LEVELS,STRESS
from .models import load_agent
from .confirm import specifications
from .analysis import decide
from ..run import save

def read(path):return json.loads(Path(path).read_text())
def audit_outcomes(rows,expected=None):
    assert rows
    if expected is not None:assert len(rows)==expected,(len(rows),expected)
    for row in rows:
        assert bool(row['stable_landing'])==(row['outcome']=='stable_landing')
        assert 1<=row['episode_steps']<=1440
        for key in ['tracking_rmse','mean_gate','eta_mae','q_delta_energy']:assert math.isfinite(row[key]),key
        assert 0<=row['mean_gate']<=1+1e-6
        if row['stable_landing']:
            if row.get('level',0)>0:assert row['exposed'],'Successful flight never encountered scheduled fault'
            assert row['stable_seconds']>=1-1e-8 and row['first_contact'] is not None
            for key,limit in [('vertical_speed',.25),('horizontal_speed',.25),('tilt',math.radians(12)),('angular_speed',1.),('radius',.2)]:assert row['first_contact'][key]<=limit+1e-9,key

def main():
    frozen=freeze();verify(frozen);torch.set_num_threads(1);checks={};weights=[]
    for group in groups():
        for entry in group['entries']:
            m=load_agent(entry['path']);assert all(torch.isfinite(p).all() for p in m.state_dict().values())
            weights.append(dict(group=group['id'],seed=entry['seed'],sha256=sha(entry['path'])));del m
            directory=Path(entry['directory'])
            if group['kind']=='observer':assert read(directory/'bc_timing.json')['updates']==1200
            else:assert read(directory/'complete.json')['steps']==524288
    assert len(weights)==75;checks['all_75_models_trained_and_finite']=True
    data=read(OUT/'data_audit.json');assert data['train_routes']==48 and data['validation_routes']==12 and data['transition_chain_verified'];checks['data_disjoint_and_transitions_valid']=True
    mismatches=[];datacount=0
    for file in (OUT/'data').glob('*/route_*.json'):
        record=read(file);datacount+=1
        for path,digest in record['source_sha256'].items():
            if sha(path)!=digest:mismatches.append((str(file),path))
    assert datacount==72 and not mismatches,mismatches;checks['all_data_physics_sources_unchanged']=True
    hpo=[read(OUT/'hpo'/f'{a}_selection.json') for a in ['lstm','mamba','mamba2','mamba3','gtrxl','ltc']]
    assert all(len(r['candidates'])==4 for r in hpo)
    hposeeds={ep['seed'] for r in hpo for c in r['candidates'] for ep in c['validation']['episodes']};assert hposeeds==set(range(527100,527116)),hposeeds
    checks['hpo_routes_disjoint_from_model_selection']=True
    choices={r['config']['arch']:r['config'] for r in hpo}
    for group in groups():
        if group['kind']=='observer':continue
        for entry in group['entries']:
            model=load_agent(entry['path']);assert model.cfg==choices[model.cfg['arch']];del model
    checks['all_finetuned_models_match_reserved_hpo_config']=True

    def state_equal(first,second):
        a=load_agent(first);b=load_agent(second)
        assert set(a.state_dict())==set(b.state_dict())
        assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    for arch in ['lstm','mamba','mamba2','mamba3','gtrxl','ltc']:
        for i in [1,2,3]:state_equal(OUT/'hpo'/f'{arch}_0/bc.pt',OUT/'hpo'/f'{arch}_{i}/bc.pt')
    for seed in [101,102,103,104,105]:
        for method in ['csil','meirl','rime']:state_equal(OUT/'models/lstm'/f'seed_{seed}/bc.pt',OUT/'reward_models'/method/f'seed_{seed}/bc.pt')
    checks['hpo_and_reward_comparisons_share_identical_initial_states']=True
    pilot_reproduction=OUT/'pilot_reproduction_audit.json'
    if pilot_reproduction.exists():assert read(pilot_reproduction)['bit_identical']
    checks['available_historical_pilot_reproduction_passed']=True

    manifest=read(OUT/'final_test_manifest.json');logicalcases=0
    for item in manifest['logical']:
        for condition in item['conditions']:
            rows=read(OUT/'test'/f"{item['label']}_{condition}.json");audit_outcomes(rows,60*len(item['levels']));logicalcases+=len(rows)
            assert {(r['level'],r['seed']) for r in rows}=={(l,s) for l in item['levels'] for s in range(529000,529060)}
    checks['all_curve_cases_and_ablations_covered']=True
    case_plan=read(OUT/'confirmation_case_plan.json');expected={(r['role'],r['seed'],r['level'],r['condition']) for r in case_plan};assert len(expected)==5000
    specs=specifications()
    for spec in specs:
        rows=read(OUT/'confirmation'/f"{spec['id']}.json");audit_outcomes(rows,5000)
        assert {(r['role'],r['seed'],r['level'],r['condition']) for r in rows}==expected
    checks['independent_confirmation_complete']=True
    for spec in read(OUT/'sensitivity_manifest.json'):
        for mode in ['noise2','clock4']:audit_outcomes(read(OUT/'sensitivity'/f"{spec['id']}_{mode}.json"),96)
    checks['all_family_sensitivity_cases_complete']=True
    physics=read(OUT/'paired_physics_audit.json');assert physics['cases']==physics['repeat_identical']==physics['natural_outcome_agreement']==physics['below_gate_margin']==36
    precision=read(OUT/'physics_precision_summary.json');assert len(precision)==6 and all(r['cases']==r['success']==36 for r in precision)
    inf=read(OUT/'inference_audit_complete.json');assert inf['passed'] and len(inf['actors'])==75 and len(inf['physical_groups'])==15
    checks['numerical_replay_and_all_actor_parity_passed']=True
    oracle=read(OUT/'oracle_proposal_audit_complete.json');assert oracle['models']==75 and oracle['episodes']==300 and len(oracle['groups'])==15
    checks['actual_frozen_proposals_diagnosed_in_every_family']=True
    test_receipt=read(OUT/'tests_final_receipt.json');assert test_receipt['exit_code']==0
    xml=ET.parse(OUT/'tests_final.xml').getroot();suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    assert suites and all(int(s.attrib.get('failures',0))==0 and int(s.attrib.get('errors',0))==0 for s in suites)
    checks['unit_and_regression_tests_passed']=True
    before=read(OUT/'decision.json');after=decide();assert before==after and len(after['comparisons'])*3<=36;checks['decision_reproduced_without_retraining']=True
    bundle=read(OUT/'recommended_controller.json');replay=read(OUT/'recommended_replay.json');assert replay['bundle']==bundle;audit_outcomes(replay['rows'],8)
    name=bundle['id']
    if name=='baseline_selected':reference=read(OUT/'runtime_selection/base_estimated.json')
    elif name=='baseline_original':reference=read(OUT/'structural_challenger/original_estimated.json')
    elif name=='baseline_observer_only':
        label=read(OUT/'observer_fallback_selection.json')['label'];reference=read(OUT/'core_validation_shared_sensors'/f'{label}_estimated.json')
    elif name=='baseline_balanced':reference=read(OUT/'structural_challenger/balanced_estimated.json')
    else:
        candidate=next(c for c in read(OUT/'runtime_selection_complete.json')['production_candidates'] if c['id']==name)
        reference=read(OUT/'runtime_selection/cases'/f"{name}_{candidate['seed']}_{candidate['strategy']['id']}_estimated.json")
    ref={(r['level'],r['seed']):r for r in reference}
    for row in replay['rows']:
        old=ref[(row['level'],row['seed'])];assert row['outcome']==old['outcome'] and row['episode_steps']==old['episode_steps'] and abs(row['tracking_rmse']-old['tracking_rmse'])<1e-8
    checks['recommended_bundle_reproduces_validation']=True
    report=read(OUT/'report_complete.json');assert sha(report['report'])==report['report_sha256']
    for image in report['figures']:assert sha(image['path'])==image['sha256'] and Path(image['path']).stat().st_size>1000
    checks['report_and_all_figures_present']=True
    from .replay_cache import source_digest
    replay=[];current_replay_source=source_digest()
    for path in OUT.rglob('*.meta.json'):
        record=read(path).get('zero_action_replay')
        if record is None:continue
        assert record['source_sha256']==current_replay_source
        assert record['trajectory_certified_zero_cases']+record['fully_simulated_policy_cases']==record['cases']
        replay.append(dict(path=str(path.relative_to(OUT)),**record))
    checks['zero_action_replay_provenance_and_case_coverage']=True
    verify(frozen);checks['frozen_code_weights_and_previous_models_unchanged']=True
    artifacts=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.suffix in ['.json','.csv','.txt','.xml','.png','.npz','.pt'] and path.name not in ['final_audit.json','evaluation_pipeline_status.json','final_integrity_audit_receipt.json','ALL_COMPLETE.json']:
            artifacts.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size))
    save(OUT/'final_audit.json',dict(passed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,policy_weights=weights,
        logical_final_cases=logicalcases,unique_curve_policies=len(manifest['unique_jobs']),confirmation_cases=5000*len(specs),data_route_files=datacount,
        previous_round_frozen_models=frozen['previous_frozen_models_verified'],unit_tests=sum(int(s.attrib.get('tests',0)) for s in suites),zero_replay_certificates=replay,artifacts=artifacts,recommended=name))
    print('FINAL_AUDIT_PASSED',len(checks),'checks',len(weights),'models',logicalcases,'logical_curve_cases','recommended',name,flush=True)
if __name__=='__main__':main()
