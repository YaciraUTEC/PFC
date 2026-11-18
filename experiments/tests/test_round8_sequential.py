import itertools,math
import numpy as np
import pytest
from experiments.round8.sequential import evidence,spending,assignment

def test_seven_independent_gains_support_benefit_and_preservation():
 b=np.ones(1600,bool);c=b.copy();b[:7]=False
 e=evidence(b,c);assert e['positive_gain_evidence'];assert e['e_value']==128;assert e['conditional_harm_upper']<.005;assert e['both_statistical_conditions']

def test_any_lost_baseline_success_prevents_acceptance():
 b=np.ones(1600,bool);c=b.copy();b[:20]=False;c[30]=False
 e=evidence(b,c);assert e['losses']==1;assert e['e_value']==0;assert not e['both_statistical_conditions']

def test_zero_effect_is_not_an_improvement():
 e=evidence(np.ones(2048,bool),np.ones(2048,bool));assert e['preservation_evidence'];assert not e['positive_gain_evidence']

def test_revision_alpha_spending_is_bounded():
 assert sum(spending(i)['benefit_alpha'] for i in range(1,100))==pytest.approx(.025);assert sum(spending(i)['preservation_alpha'] for i in range(1,100))==pytest.approx(.025)

def test_optional_stopping_gain_wealth_has_correct_null_crossing_probability():
 # Fair independent discordant signs. Crossing can occur before all14 signs.
 crossed=0
 for signs in itertools.product([False,True],repeat=14):
  g=0;wealth=1;hit=False
  for gain in signs:
   wealth=wealth*2 if gain else 0
   if wealth>=80:hit=True
  crossed+=hit
 assert crossed/(2**14)==pytest.approx(1/128);assert crossed/(2**14)<=spending(1)['benefit_alpha']

def test_assignment_has_one_independent_model_regime_per_scenario():
 a=assignment(range(1000,1200),73119);assert a==assignment(range(1000,1200),73119);assert len({r['seed'] for r in a})==200;assert set(r['model_seed'] for r in a)==set(range(501,506));assert all(r['condition']=='estimated' for r in a)

def test_protection_bound_threshold_is_numerically_correct():
 alpha=spending(1)['preservation_alpha'];n=math.ceil(math.log(alpha)/math.log(.995));assert evidence(np.ones(n,bool),np.ones(n,bool))['preservation_evidence'];assert not evidence(np.ones(n-1,bool),np.ones(n-1,bool))['preservation_evidence']
