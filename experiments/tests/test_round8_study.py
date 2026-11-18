"""Inference and accounting checks needed for a checkpoint-specific prospective study."""
import numpy as np
from experiments.round8.sequential import assignment
from experiments.round8.study import point_statistics,subset

def row(seed,success,noise=1.):return dict(seed=seed,level=.296,condition='estimated',noise=noise,clock=None,stable_landing=success)
def test_selected_operational_checkpoint_is_assigned_without_hidden_random_model_mixture():
 a=assignment(range(852000,853024),98319,[502]);assert {x['model_seed'] for x in a}=={502};assert len({x['seed'] for x in a})==1024;assert {x['level'] for x in a}=={.295,.296};assert {x['noise'] for x in a}=={1.,2.}
def test_operational_bootstrap_clusters_repeated_settings_by_route():
 b=dict(episodes=[row(1,False),row(1,False,2.),row(2,True)]);r=dict(episodes=[row(1,True),row(1,True,2.),row(2,True)]);m=point_statistics(r,b,replicates=1000);assert m['route_clusters']==2;assert m['mean']==.5;assert m['lower95']==0.;assert m['upper95']==1.
def test_route_bootstrap_does_not_claim_benefit_when_nothing_changes():
 b=dict(episodes=[row(i,True) for i in range(10)]);m=point_statistics(b,b,replicates=1000);assert m['mean']==m['lower95']==m['upper95']==0.
def test_result_subset_uses_complete_case_not_only_route():
 b=dict(episodes=[row(1,True,2.)]);r=dict(episodes=[row(1,True),row(1,False,2.),row(2,True)]);ss=subset(r,b);assert len(ss['episodes'])==1;assert ss['episodes'][0]['noise']==2.;assert not ss['episodes'][0]['stable_landing']
