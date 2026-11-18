"""Development qualification cannot hide losses in the added severity sweep."""
import pytest
from experiments.round8 import select_checkpoint as selection
from experiments.round7.evaluation import compare

def row(seed,success):
 return dict(seed=seed,level=.30,condition='estimated',noise=2.,clock=None,stable_landing=success,fresh_simulation=True,baseline_reused_verified_off=False)

def setup_pair(monkeypatch):
 old=dict(episodes=[row(1,True)])
 previous=dict(episodes=[row(1,True)],fingerprint={'gate':'fixed'})
 sweep=dict(episodes=[row(2,True)])
 extra=dict(episodes=[row(2,False)],fingerprint={'gate':'fixed'},fresh_simulations=1,verified_off_reuse=0,seconds=1.)
 monkeypatch.setattr(selection,'run',lambda *args,**kwargs:extra)
 candidate=dict(seed=502,family='guarded_bootstrap_native',spec={'version':4})
 return candidate,previous,sweep,old,extra

def test_new_severity_loss_survives_combination_and_old_reference_is_not_new_flight(monkeypatch,tmp_path):
 c,p,s,b,e=setup_pair(monkeypatch)
 result=selection.qualify(c,p,s,b,tmp_path)
 assert compare(result,dict(episodes=b['episodes']+s['episodes']))['losses']==1
 assert result['logical_cases']==2 and result['identical_policy_reuse']==1
 assert result['fresh_simulations']==1
 prior=next(x for x in result['episodes'] if x['seed']==1)
 assert prior['identical_policy_result_reused'] and not prior['fresh_simulation']

def test_qualification_refuses_to_merge_different_policy_fingerprints(monkeypatch,tmp_path):
 c,p,s,b,e=setup_pair(monkeypatch);e['fingerprint']={'gate':'changed'}
 with pytest.raises(AssertionError):selection.qualify(c,p,s,b,tmp_path)
