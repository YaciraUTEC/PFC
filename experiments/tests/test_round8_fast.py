import numpy as np
import pytest
import torch
from experiments.round8.common import OUT,read
from experiments.round8.fast_policy import FastFactory
from experiments.round8.loading import Policy
from experiments.round7.fast_gtrxl import install
from experiments.round4.models import tree_map
from experiments.tests.test_round8 import fixed_gate

@pytest.mark.parametrize('version,option',[(2,1),(3,5),(3,1)])
def test_shared_encoder_and_completed_option_skip_preserve_actions_through_partial_resets(version,option):
 torch.set_num_threads(1)
 gate=OUT/'factor_gates/active_v1/seed_501.pt'
 if not gate.exists():pytest.skip('Frozen actual proposal required')
 template=read(OUT/'h5_conservative_validation_conclusion.json')['entries'][0]['spec'];spec=dict(template,version=version,allowed_options=[option],option_mode='both' if version==3 else 'native');slow=install(Policy.from_spec(spec));fast=install(FastFactory.from_spec(spec))
 # Force a committed option, then force the old veto OFF mid-option. This checks
 # both native and raw semantics while actual GTrXL memory is reset per episode.
 slow.gate=fixed_gate(.2,.9999,.8,201);fast.gate=fixed_gate(.2,.9999,.8,201);slow.proposer.threshold=fast.proposer.threshold=-1e20;ss=fs=None;rng=np.random.default_rng(891);skipped=0
 with torch.no_grad():
  for tick in range(160):
   if tick in [40,81,122]:
    keep=torch.tensor([False,True,False]);ss=tree_map(lambda x:x*keep.reshape(3,*([1]*(x.ndim-1))),ss);fs=tree_map(lambda x:x*keep.reshape(3,*([1]*(x.ndim-1))),fs);slow.proposer.threshold=fast.proposer.threshold=-1e20
   if tick in [7,47,88,129]:slow.proposer.threshold=fast.proposer.threshold=1e20
   obs=torch.as_tensor(rng.normal(0,.2,(3,88)),dtype=torch.float64);obs[:,57]=0.;obs[:,72]=1.;obs[:,64:68]=.4
   a,ss=slow.predict_actions(obs,ss);b,fs=fast.predict_actions(obs,fs);torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(ss[2],fs[2],rtol=0,atol=0);torch.testing.assert_close(ss[3],fs[3],rtol=0,atol=0);torch.testing.assert_close(ss[5],fs[5],rtol=0,atol=0);skipped+=not bool(fast.last_diagnostics['neural_evaluated'].any())
 assert skipped>30
