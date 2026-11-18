"""Load the frozen operational GTrXL controller and run one reproducible demo."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,TASK,read,save,sha
from .loading_compact import Policy as PreservePolicy
from ..round7.fast_gtrxl import install
from ..round7.episode import fly

def load(config_path):
 config_path=Path(config_path).resolve();cfg=read(config_path);root=Path(cfg.get('bundle_root',str(TASK)))
 if not root.is_absolute():root=(config_path.parent/root).resolve()
 spec=dict(cfg['spec'])
 for field in ['gate','proposer']:
  p=Path(spec[field]);p=p if p.is_absolute() else root/p;assert p.is_file(),str(p);assert sha(p)==cfg['model_sha256'][field],field+' checkpoint checksum mismatch';spec[field]=str(p)
 # Canonical risk uses exactly the frozen201 prior at the relative project tree.
 expected=TASK/'PFC/runs/intervention_round6/phase3/flashsac/gtrxl/seed_201/best.pt';assert sha(expected)==cfg['model_sha256']['proposer']
 from ..round7.policy import controller_descriptor
 np.testing.assert_array_equal(controller_descriptor(spec['proposer']),controller_descriptor(expected),err_msg='Proposer folder descriptor changed; preserve the delivered relative model tree')
 torch.set_num_threads(1);return install(PreservePolicy.from_spec(spec)),cfg

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--seed',type=int,default=810085);p.add_argument('--loss',type=float,default=29.6,help='RPM loss percent of one motor');p.add_argument('--condition',choices=['exact','estimated'],default='estimated');p.add_argument('--noise',type=float,default=1.);p.add_argument('--clock',type=float);p.add_argument('--output',required=True);a=p.parse_args();target=Path(a.output).resolve();assert target.is_relative_to(TASK.resolve()),'All demo output must stay inside the thesis/bundle folder'
 assert 0<=a.loss<=100 and a.noise>=0;model,cfg=load(a.config);base=fly(a.seed,a.loss/100,a.condition,a.noise,a.clock);candidate=fly(a.seed,a.loss/100,a.condition,a.noise,a.clock,model=model);save(target,dict(source_root=str(ROOT.resolve()),task_root=str(TASK.resolve()),baseline=base,candidate=candidate,operational_gate=cfg['operational_gate'],scope='One reproducible demonstration, not an independent performance estimate. No real drone actuation.'));print('DSLPID',base['outcome'],'GTrXL',candidate['outcome'],flush=True)
if __name__=='__main__':main()
