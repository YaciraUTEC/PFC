"""Close reproduction before the first three conditional-model fits."""
import time
import numpy as np
import torch
from .common import ROOT,OUT,R7,read,save,verify_round7,close
from ..round7.power import awake
from ..round7.evaluation import evaluate,key
from .fit import main as fit

def main():
 torch.set_num_threads(1);start=time.time();integrity=verify_round7();save(OUT/'status.json',dict(stage='h0_reproduction',status='running'))
 base=read(R7/'final/primary/baseline.json');lookup={key(r):r for r in base['episodes']}
 cc=[dict(seed=s,level=l,condition=c,noise=n,clock=None) for s,l,c,n in [(740029,.296,'estimated',1.),(740035,.296,'estimated',1.),(740038,.296,'estimated',1.),(740000,0.,'exact',1.)]]
 result=evaluate(None,cc,workers=4,path=OUT/'h0_baseline_reproduction.json');largest=0.
 for row in result['episodes']:
  old=lookup[key(row)];assert row['outcome']==old['outcome'] and row['episode_steps']==old['episode_steps'];error=float(np.max(np.abs(np.asarray(row['terminal_position'])-old['terminal_position'])));largest=max(largest,error);assert error<1e-8
 close('h0_reproduction','The isolated copy preserves natural baseline outcomes, step counts and terminal positions; previous sources and selected models remain unchanged.',integrity=integrity,cases=len(cc),largest_position_difference=largest,structural_tests_passed=9,seconds=time.time()-start)
 (OUT/'PLAN.txt').write_text((ROOT/'experiments/round8/PLAN.txt').read_text(encoding='utf-8'),encoding='utf-8')
 with awake():
  save(OUT/'status.json',dict(stage='h1_factor_fits',status='running'))
  for seed in [501,502,503]:
   if not (OUT/'factor_gates/initial'/f'seed_{seed}_complete.json').exists():fit(seed=seed,updates=3200)
 close('h1_factor_fits','Three route-bootstrap seeds fitted with separate baseline/preservation/rescue likelihoods and disjoint calibration roles.',seeds=[501,502,503])
 save(OUT/'status.json',dict(stage='h1_ready_for_flight_pilot',status='running'))
if __name__=='__main__':main()
