"""Option compaction retains decisions and the complete feedback pulse."""
import pytest
import torch
from experiments.round8.common import OUT
from experiments.round7.common import old_model
from experiments.round7.fast_gtrxl import install
from experiments.round4.models import tree_map
from experiments.round8.loading_compact import Policy


def models(seed,force=False):
 spec=dict(version=4,proposer=str(old_model(201)),gate=str(OUT/f'factor_gates/active_v1/seed_{seed}.pt'),minimum_gain=-2. if force else .01,maximum_harm=2. if force else .02,penalty=1.,option_mode='both',risk_aggregation='max_member',allowed_options=[5],fast_execution=True,observed_loss_min=.02,sigma_multiplier=0.)
 a=install(Policy.from_spec(spec));b=install(Policy.from_spec(dict(spec,compact_options=True)))
 if force:a.proposer.gate=False;b.proposer.gate=False
 return a,b

@pytest.mark.parametrize('seed',[501,505])
def test_compaction_preserves_all_admissible_features_and_forecasts(seed):
 torch.set_num_threads(1);torch.manual_seed(381);a,b=models(seed);obs=torch.randn(7,88,dtype=torch.float64)*.1;obs[:,64:68]=.5;obs[:,72]=.8;obs[:,31]=1.1;obs[:,57]=0;context=torch.randn(7,184,dtype=torch.float64);raw=torch.randn(7,4,dtype=torch.float64)*.2;on=torch.ones(7,dtype=torch.bool)
 with torch.no_grad():
  aa,ab,ax=a.build_options(obs,context,raw,on);ba,bb,bx=b.build_options(obs,context,raw,on);assert a.option_ids==list(range(8)) and b.option_ids==[5];assert torch.equal(aa[:,5],ba[:,0]) and torch.equal(ab[:,5],bb[:,0]) and torch.equal(ax[:,5],bx[:,0]);al,am=a.assess(ax,aa);bl,bm=b.assess(bx,ba);assert torch.equal(al[:,5],bl[:,0])
  for k in am:torch.testing.assert_close(am[k][:,5],bm[k][:,0],rtol=1e-12,atol=1e-12)


def test_compact_feedback_and_resets_match_dense_actions_with_real_gtrxl():
 torch.set_num_threads(1);torch.manual_seed(889);a,b=models(505,True);sa=sb=None;starts=0
 with torch.no_grad():
  for t in range(180):
   obs=torch.randn(3,88,dtype=torch.float64)*.1;obs[:,64:68]=.5;obs[:,72]=.8;obs[:,57]=0;obs[:,31]=1.1 if t>=10 else 0.;obs[:,32:35]=0
   aa,sa=a.predict_actions(obs,sa);bb,sb=b.predict_actions(obs,sb);assert torch.equal(aa.float(),bb.float());assert torch.equal(sa[2],sb[2]) and torch.equal(sa[3],sb[3]);assert torch.equal(a.last_diagnostics['option_started'],b.last_diagnostics['option_started']);starts+=int(a.last_diagnostics['option_started'].sum())
   for k in ['option_initial_base_success','option_initial_preservation','option_initial_joint_rescue','option_initial_joint_harm','option_initial_conditional_harm','option_initial_lower_gain']:torch.testing.assert_close(a.last_diagnostics[k],b.last_diagnostics[k],rtol=1e-12,atol=1e-12)
   active=sa[3]>0
   assert torch.equal(a.last_diagnostics['selected_option'][active],b.last_diagnostics['selected_option'][active])
   if t in [75,125]:
    keep=torch.tensor([False,True,True]);sa=tree_map(lambda x:x*keep.reshape(3,*([1]*(x.ndim-1))),sa);sb=tree_map(lambda x:x*keep.reshape(3,*([1]*(x.ndim-1))),sb)
 assert starts>=5
