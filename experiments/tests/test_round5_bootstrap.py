"""Cold-start and registered-input reproducibility checks."""
import json,hashlib
from pathlib import Path
import pytest
from experiments.round5.bootstrap import initialize,CONFIG
from experiments.round5.bc import starting_config
from experiments.round5.confirm import build_plan


def test_initial_architecture_settings_do_not_require_previous_run_outputs():
    expected=json.loads((CONFIG/'reference_inputs.json').read_text())['starting_architectures']
    for arch,cfg in expected.items():
        actual=starting_config(arch);assert actual==cfg
        actual['lr']=99.;assert starting_config(arch)['lr']==cfg['lr']


def test_registered_case_generator_matches_preregistered_fingerprint():
    expected=json.loads((CONFIG/'reference_inputs.json').read_text())['case_plan_sha256'];cases=build_plan()
    digest=hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest==expected and len(cases)==5000
    assert len({r['seed'] for r in cases})==1000


def test_bootstrap_populates_missing_inputs_without_overwriting_existing_run(tmp_path):
    (tmp_path/'runtime_environment.json').write_text('{}')
    result=initialize(tmp_path);assert result['registered_cases']==5000
    casefile=tmp_path/'confirmation_case_plan.json';before=casefile.read_bytes()
    marker=tmp_path/'protocol_initial.json';marker.write_text('{"preserve_existing":true}')
    again=initialize(tmp_path)
    assert not again['copied_protocols'] and casefile.read_bytes()==before and json.loads(marker.read_text())==dict(preserve_existing=True)


def test_fresh_prepare_runs_engineering_before_data_instead_of_waiting_forever(tmp_path,monkeypatch):
    from experiments.round5 import prepare,bootstrap
    monkeypatch.setattr(prepare,'OUT',tmp_path);monkeypatch.setattr(bootstrap,'initialize',lambda out:None)
    calls=[]
    class StopBeforeTraining(Exception):pass
    def stage(label,args):
        calls.append(label)
        outputs=dict(core_engineering=['core_selection.json'],governor_validation=['structural_challenger.json'],physics_precision=['paired_physics_audit.json','physics_precision_summary.json'],observer_only_validation=['observer_fallback_selection.json'])
        if label=='collect_initial':raise StopBeforeTraining
        for name in outputs[label]:(tmp_path/name).write_text(json.dumps(dict(engineering_version=4)))
    monkeypatch.setattr(prepare,'stage',stage)
    with pytest.raises(StopBeforeTraining):prepare.main()
    assert calls==['core_engineering','governor_validation','physics_precision','observer_only_validation','collect_initial']
