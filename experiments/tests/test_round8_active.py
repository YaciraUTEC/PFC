import numpy as np
from experiments.round8.active_scan import choose_ticks,partition

def test_queries_cover_first_decision_variation_and_native_onset_with_bounded_cost():
 n=np.zeros(300,bool);n[20:]=True;t=choose_ticks(n,[60,120,61,125,None]);assert {20,60,61,120,125}.issubset(t);assert len(t)<=12

def test_no_native_proposal_means_no_counterfactual_query():
 assert choose_ticks(np.zeros(100,bool),[30,50,None])==[]

def test_nearby_actual_first_decisions_are_retained_as_distinct_correlated_states():
 n=np.ones(100,bool);t=choose_ticks(n,[20,21,22]);assert {20,21,22}.issubset(t);assert len(t)==len(set(t))

def test_route_partition_is_repeatable_and_does_not_depend_on_regime_or_outcome():
 a={seed:partition(seed) for seed in range(810000,810256)};assert a=={seed:partition(seed) for seed in a};assert set(a.values())=={'train','calibration'};assert 40<=sum(x=='calibration' for x in a.values())<=90


def test_shared_proposal_context_matches_full_original_proposal_and_canonical_history():
 import torch
 from experiments.round8.active_data import shared_proposal_context
 from experiments.round7.common import old_model
 from experiments.round7.episode import load_policy
 from experiments.round7.policy import propose,canonical_risk,risk_features
 from experiments.round7.fast_gtrxl import install
 if not old_model(201).exists():
  import pytest
  pytest.skip('Frozen201 checkpoint required for equivalence check')
 torch.set_num_threads(1);torch.manual_seed(775)
 a=install(load_policy(old_model(201)));b=install(load_policy(old_model(201)));risk=install(canonical_risk().double().eval());sa=sb=sr=None
 with torch.no_grad():
  for _ in range(140):
   x=torch.randn(2,88,dtype=torch.float64);ra,oa,sa,context=shared_proposal_context(a,x,sa);rb,ob,sb=propose(b,x,sb);rc,sr=risk_features(risk,x,sr)
   torch.testing.assert_close(ra,rb,rtol=0,atol=0);assert torch.equal(oa,ob);torch.testing.assert_close(context,rc,rtol=0,atol=0);torch.testing.assert_close(sa[1],sb[1],rtol=0,atol=0)
