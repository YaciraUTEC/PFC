"""Regression checks for the completed experimental protocol and production replay."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.round5.train import validation_routes
from experiments.round5.statistics import compare_confirmation
from experiments.round5.models import Agent,configurations,ARCHS
from experiments.round5.inventory import always_off,sha
from experiments.round5.final_audit import audit_outcomes

def confirmation(improved=False,unsafe=False):
    rows=[]
    for seed in range(1000):
        for level in [.28,.29,.295]:rows.append(dict(role='primary',seed=seed,level=level,condition='estimated',stable_landing=seed<(900 if improved else 700)))
        rows.append(dict(role='mild',seed=seed,level=.02,condition='estimated',stable_landing=not(unsafe and seed<20)))
        rows.append(dict(role='exact',seed=seed,level=.295,condition='exact',stable_landing=True))
    return rows

def test_adoption_rejects_equal_policy_and_accepts_clear_safe_gain():
    baseline=confirmation();assert not compare_confirmation(baseline,baseline)['statistical_pass']
    result=compare_confirmation(confirmation(improved=True),baseline)
    assert result['statistical_pass'] and result['primary']['lower']>0

def test_good_primary_result_cannot_override_protected_harm():
    result=compare_confirmation(confirmation(improved=True,unsafe=True),confirmation())
    assert result['primary']['lower']>0 and not result['statistical_pass'] and result['risk']['mild']['upper']>.01

def test_confirmation_demands_all_paired_case_ids():
    with pytest.raises(AssertionError):compare_confirmation(confirmation()[:-1],confirmation())

def test_hpo_never_uses_model_selection_or_final_routes():
    hpo=set(validation_routes(Path('runs/hpo/lstm_0')));selection=set(validation_routes(Path('runs/models/lstm/seed_101')))
    assert hpo==set(range(527100,527116)) and selection==set(range(528000,528008)) and not hpo&selection

@pytest.mark.parametrize('threshold,certified',[(0.,True),(-4.,False)])
def test_off_certificate_is_a_bound_not_an_observation(threshold,certified):
    m=Agent(configurations('lstm')[0]);m.gate_logit_threshold=threshold
    off,upper=always_off(m);assert off==certified
    if off:
        with torch.no_grad():a,_=m.predict_actions(torch.randn(64,80)*100)
        assert torch.all(a[:,5]==-1)
    with torch.no_grad():m.actor[-1].weight[5,0]=20.
    assert not always_off(m)[0]

def test_bundle_rejects_changed_checkpoint_before_loading(tmp_path):
    from experiments.round5.deployment import read_bundle
    file=tmp_path/'weights.pt';file.write_bytes(b'original')
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps(dict(checkpoint=str(file),checkpoint_sha256=sha(file),judge_path=None)))
    assert read_bundle(bundle)['checkpoint']==str(file)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):read_bundle(bundle)

def test_audit_rejects_contact_without_controlled_impact():
    row=dict(stable_landing=True,outcome='stable_landing',episode_steps=100,tracking_rmse=.1,mean_gate=0,eta_mae=0,q_delta_energy=0,stable_seconds=1,
        first_contact=dict(vertical_speed=.1,horizontal_speed=.1,tilt=.1,angular_speed=.5,radius=.1))
    audit_outcomes([row],1);bad=copy.deepcopy(row);bad['first_contact']['angular_speed']=1.01
    with pytest.raises(AssertionError):audit_outcomes([bad],1)
    missed=copy.deepcopy(row);missed.update(level=.295,exposed=False)
    with pytest.raises(AssertionError):audit_outcomes([missed],1)

def test_full_latency_path_with_all_six_real_recurrent_policies(tmp_path,monkeypatch):
    from experiments.round5 import latency
    torch.set_num_threads(1);jobs=[]
    for arch in ARCHS:
        p=tmp_path/(arch+'.pt');m=Agent(configurations(arch)[0]);m.save(p)
        jobs.append(dict(id=arch,path=str(p),config=dict(estimator='adaptive',readiness=True,emergency='damped',reserve=.003),threshold=0.,shadow=False))
    monkeypatch.setattr(latency,'OUT',tmp_path);monkeypatch.setattr(latency,'specifications',lambda:jobs);monkeypatch.setattr(latency,'groups',lambda:[])
    latency.main();result=json.loads((tmp_path/'latency_complete.json').read_text())
    assert len(result['results'])==6
    assert all(r['samples']==256 and r['p95_seconds']>0 and np.isfinite(r['p95_seconds']) for r in result['results'])


def test_structured_updates_require_incremental_confirmed_benefit():
    from experiments.round5.analysis import choose_structured
    ids=['baseline_original','baseline_observer_only','baseline_selected']
    c={'baseline_observer_only/baseline_original':dict(adoption_pass=True),'baseline_selected/baseline_original':dict(adoption_pass=True),'baseline_selected/baseline_observer_only':dict(adoption_pass=False)}
    assert choose_structured(ids,c)=='baseline_observer_only'
    c['baseline_selected/baseline_observer_only']['adoption_pass']=True
    assert choose_structured(ids,c)=='baseline_selected'
    c['baseline_observer_only/baseline_original']['adoption_pass']=False
    assert choose_structured(ids,c)=='baseline_selected'

def test_hierarchical_and_learning_contrasts_fit_declared_endpoint_budget():
    from experiments.round5.analysis import comparison_pairs
    ids=['baseline_original','baseline_observer_only','baseline_selected','residual_lstm','observer_mamba','reward_csil']
    pairs=comparison_pairs(ids);assert len(pairs)==12 and len(pairs)*3==36
    assert ('baseline_selected','baseline_observer_only') in pairs and not any(a=='baseline_original' for a,b in pairs)


def test_actual_proposal_oracle_restores_state_and_rng():
    import json
    from experiments.round5.oracle_proposals import compare_proposal
    from experiments.round5.env import LandingEnv
    env=LandingEnv(condition='estimated')
    try:
        env.reset(seed=525000);physical=env.core.obs_raw.copy();q=env.c._previous_q.copy();state=env._state().copy();rng=json.dumps(env.sensor_rng.bit_generator.state)
        result=compare_proposal(env,np.array([.2,-.1,.1,0.,-.8,1.],np.float32),'residual',horizon=8)
        assert env.t==0 and np.isfinite(result['gain']) and result['gain']==result['base_cost']-result['candidate_cost']
        np.testing.assert_array_equal(env.core.obs_raw,physical);np.testing.assert_array_equal(env.c._previous_q,q);np.testing.assert_array_equal(env._state(),state)
        assert json.dumps(env.sensor_rng.bit_generator.state)==rng
    finally:env.close()
