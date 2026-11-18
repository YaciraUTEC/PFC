"""Early qualification of the declared severity sweep, only two CPU simulators."""
import argparse
import torch
from .common import OUT,read,close
from .cache import baseline
from .cache_guarded import run
from ..round7.evaluation import cases,compare
from ..round7.power import awake

def main(name="bootstrap_native",seed=502):
 torch.set_num_threads(1);root=OUT/'development/checkpoint_selection';spec=read(OUT/f'development/observed_support_v1/{name}/seed_{seed}/results.json')['spec']
 with awake():
  b=baseline(cases(range(814000,814064),levels=[.02,.10,.28,.29,.30]),root/'severity_sweep',workers=2);r=run(spec,b,root/f'with_severity_sweep/guarded_{name}/sweep_seed_{seed}',workers=2);close('h9_priority_severity' if name=='bootstrap_native' and seed==502 else f'h9_priority_severity_{name}_{seed}','Prospectively declared development severity sweep evaluated before any frozen final trial.',metrics=compare(r,b),workers=2)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',default='bootstrap_native');p.add_argument('--seed',type=int,default=502);a=p.parse_args();main(a.name,a.seed)
