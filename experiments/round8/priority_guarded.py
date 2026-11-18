"""Prioritized raw candidate on a small independent CPU pool, same declared cases."""
import argparse
import torch
from .common import OUT,read,save,close
from .cache_guarded import run
from ..round7.evaluation import compare
from ..round7.power import awake

def main(workers=2,seed=501,family="active",menu="raw",known=False):
 torch.set_num_threads(1);base=read(OUT/('development/short_margin_v1/baseline.json' if known else 'development/active_v1/new_baseline/baseline.json'));template=read(OUT/'h5_conservative_validation_conclusion.json')['entries'][0]['spec'];tag={'active':'active_v1','bootstrap':'base_options_calibrated','original':'full_support_v1'}[family];name=menu if family=='active' else family+'_'+menu;spec=dict(template,gate=str(OUT/f'factor_gates/{tag}/seed_{seed}.pt'),version=4,fast_execution=True,allowed_options=[5] if menu=='raw' else [1],option_mode='both' if menu=='raw' else 'native',observed_loss_min=.02,sigma_multiplier=0.);save(OUT/f'priority_guarded_{name}_{seed}{"_known" if known else ""}_allocation.json',dict(spec=spec,cases=len(base['episodes']),workers=workers,scope='Same declared development cohort and policy as upcoming guarded_flow.raw501; exact cached result reused there. No second16-worker pool.'))
 with awake():r=run(spec,base,(OUT/f'development/checkpoint_selection/guarded_{name}/known_seed_{seed}' if known else OUT/f'development/observed_support_v1/{name}/seed_{seed}'),workers=workers);close(f'h9_priority_{name}_{seed}_known' if known else 'h9_priority_raw' if family=='active' and menu=='raw' and seed==501 else f'h9_priority_{name}_{seed}','Observed-support candidate evaluated while other controls finish; exact cache can be reused, no final case used.',metrics=compare(r,base),workers=workers)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=501);p.add_argument('--family',choices=['active','bootstrap','original'],default='active');p.add_argument('--menu',choices=['raw','native'],default='raw');p.add_argument('--known',action='store_true');a=p.parse_args();main(a.workers,a.seed,a.family,a.menu,a.known)
