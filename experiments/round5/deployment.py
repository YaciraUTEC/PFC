"""Reconstruct the exact recommended simulation controller and recurrent policy."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .evaluate import OUT,evaluate,close_pool
from .inventory import sha
from .models import load_agent
from .policies import ProposalPolicy
from .env import LandingEnv
from ..run import save

def read_bundle(path=None):
    path=Path(path or OUT/'recommended_controller.json');bundle=json.loads(path.read_text())
    for item,digest in [('checkpoint','checkpoint_sha256'),('judge_path','judge_sha256')]:
        if bundle.get(item):
            if sha(bundle[item])!=bundle[digest]:raise ValueError('Recommended artifact changed:'+bundle[item])
    return bundle

def load_policy(bundle):
    if bundle['checkpoint'] is None:return None
    policy=load_agent(bundle['checkpoint']);policy.gate_logit_threshold=bundle['gate_logit_threshold']
    return ProposalPolicy(policy) if bundle['proposal_only'] else policy

def make_environment(path=None,condition='estimated'):
    """Policy sees the measured observation; true effectiveness stays in simulator."""
    bundle=read_bundle(path)
    return LandingEnv(config=bundle['configuration'],condition=condition),load_policy(bundle)

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--level',type=float,default=.295);p.add_argument('--cases',type=int,default=8);p.add_argument('--condition',choices=['exact','estimated'],default='estimated');args=p.parse_args()
    if not 1<=args.cases<=32:raise ValueError('Choose1..32 reserved validation routes')
    bundle=read_bundle(args.bundle);policy=load_policy(bundle)
    try:rows=evaluate(policy,condition=args.condition,seeds=range(528000,528000+args.cases),levels=[args.level],config=bundle['configuration'])
    finally:close_pool()
    target=OUT/'recommended_replay.json';save(target,dict(bundle=bundle,rows=rows,successes=sum(r['stable_landing'] for r in rows),cases=len(rows),scope='Reproducibility check on validation routes;not additional independent evidence'))
    print('RECOMMENDED_REPLAY',bundle['id'],sum(r['stable_landing'] for r in rows),'/',len(rows),str(target),flush=True)
if __name__=='__main__':main()
