"""Paired reports must expose harm even when aggregate success is unchanged."""
import json

def test_shifted_fault_counts_do_not_cancel_harm(tmp_path,monkeypatch):
    from experiments.round6 import report
    monkeypatch.setattr(report,'OUT',tmp_path)
    folder=tmp_path/'sensitivity';folder.mkdir()
    base=[dict(seed=seed,level=level,stable_landing=level==.295) for seed in [1,2] for level in [.295,.296]]
    for clock in [4.,8.]:
        (folder/f'baseline_clock{clock:g}.json').write_text(json.dumps(dict(episodes=base)))
        for seed in [201,202]:
            rr=[dict(r) for r in base]
            if clock==4. and seed==201:
                rr[0]['stable_landing']=False;rr[1]['stable_landing']=True
            entry=dict(kind='ppo',arch='mamba3',schedule='linear',seed=seed)
            (folder/f'model{seed}_clock{clock:g}.json').write_text(json.dumps(dict(entry=entry,clock=clock,episodes=rr)))
    rows=report.sensitivity_statistics();early=next(r for r in rows if r['fault_clock_seconds']==4.)
    assert early['paired_difference_pp']==0.
    assert early['gains']==early['losses']==1
    assert early['cases']==8 and early['training_seeds']==2 and early['routes']==2
    assert next(r for r in rows if r['fault_clock_seconds']==8.)['losses']==0

def test_gate_summary_keeps_benefits_and_prevented_failures(tmp_path,monkeypatch):
    from experiments.round6 import report
    monkeypatch.setattr(report,'OUT',tmp_path)
    rows=[dict(family='fql_mamba3_linear',routes=16,gated_success=12,ungated_success=10,prevented_failures=3,missed_benefits=1),
          dict(family='fql_mamba3_linear',routes=16,gated_success=14,ungated_success=15,prevented_failures=0,missed_benefits=1)]
    r=report.summarize_gate(rows)[0]
    assert r['cases']==32 and r['prevented_failures']==3 and r['missed_benefits']==2
    assert r['gated_success_percent']==81.25 and r['ungated_success_percent']==78.125


def test_noise_decision_after_final_step_does_not_count_as_exposure():
    from experiments.round6.report import requested_noise_limits
    changes=[dict(steps=50,upper=2.),dict(steps=100,upper=5.)]
    d=requested_noise_limits(changes,100,1.)
    assert d['maximum_during_steps']==2. and d['mean_requested_upper']==1.5
    assert d['upper_after_last_decision']==5.
    early=requested_noise_limits(changes,50,1.)
    assert early['maximum_during_steps']==1. and early['mean_requested_upper']==1.
    bc=requested_noise_limits(changes,0,1.)
    assert bc['maximum_during_steps'] is None and bc['mean_requested_upper'] is None


def test_integrity_allows_only_presentation_source_with_both_path_styles(tmp_path,monkeypatch):
    import pytest
    from experiments.round6 import final_evaluation as ev
    root=tmp_path/'repo';out=root/'runs'/'round6';out.mkdir(parents=True)
    presentation=root/'experiments'/'round6'/'report.py';presentation.parent.mkdir(parents=True)
    presentation.write_text('updated presentation')
    model_source=root/'experiments'/'checked.py'
    monkeypatch.setattr(ev,'ROOT',root);monkeypatch.setattr(ev,'OUT',out)
    (out/'dataset_fingerprint.json').write_text(json.dumps(dict(files={})))
    for key in ['experiments/round6/report.py','experiments\\round6\\report.py']:
        model_source.write_text('fixed scientific source')
        frozen=dict(source_sha256={key:'old presentation hash',str(model_source.relative_to(root)):ev.sha(model_source)},
            checkpoints=[],assets_sha256={},configuration_sha256={},data_metadata_sha256={},previous_checkpoints=[])
        (out/'frozen_before_test.json').write_text(json.dumps(frozen))
        assert ev.integrity()['passed']
        model_source.write_text('unexpected scientific source change')
        with pytest.raises(AssertionError):ev.integrity()


def test_adoption_rejects_harm_in_severe_estimated_cases():
    from experiments.round6.report import acceptance_checks
    primary=dict(lower=.1,total_losses=1,mild_losses=0,exact_losses=0)
    confirmation=dict(lower=.1,total_losses=0,mild_losses=0,exact_losses=0)
    clocks=[dict(losses=0),dict(losses=0)]
    checks=acceptance_checks(primary,confirmation,clocks,5.)
    assert not checks['protected_primary'] and not all(checks.values())
    primary['total_losses']=0
    assert all(acceptance_checks(primary,confirmation,clocks,5.).values())


def test_execution_verification_requires_identical_discrete_results():
    import pytest
    from experiments.round6.final_evaluation import _compare_value
    a=dict(success=True,values=[1.,.02],count=12)
    assert _compare_value(a,dict(success=True,values=[1.+1e-12,.02],count=12))<1e-10
    with pytest.raises(AssertionError):_compare_value(a,dict(success=False,values=[1.,.02],count=12))
    with pytest.raises(AssertionError):_compare_value(a,dict(success=True,values=[1.1,.02],count=12))
