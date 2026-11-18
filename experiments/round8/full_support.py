"""Isolate training support from bootstrap diversity; same data/model/options."""
import argparse,time
import torch
from .common import OUT,read,save,close
from .fit import main as fit
from .cache import run
from ..round7.evaluation import compare
from ..round7.statistics import paired
from ..round7.power import awake

def train():
 torch.set_num_threads(1)
 for seed in range(501,506):
  marker=OUT/f'factor_gates/full_support_v1/seed_{seed}_complete.json'
  if marker.exists():assert read(marker)['route_bootstrap']==False;continue
  fit(seed=seed,folder=OUT/'data/base_options_v1',tag='full_support_v1',route_bootstrap=False)
 close('h5_full_support_fits','Isolated complete-data deep ensembles fitted: all120 training routes per member, same conditional loss, optimizer, tuning/calibration routes and model architecture. No extra demonstrations fabricated.',seeds=list(range(501,506)),reference='https://arxiv.org/abs/1612.01474',scope='Independent initialization/minibatches induce diversity; disagreement still is not a formal calibrated safety bound.')

def evaluate():
 torch.set_num_threads(1);start=time.time();old=read(OUT/'h5_conservative_validation_conclusion.json');folder=OUT/'development/full_support_v1';base=read(OUT/'development/short_margin_v1/baseline.json');save(folder/'baseline.json',base);specs=[dict(row['spec'],gate=str(OUT/f"factor_gates/full_support_v1/seed_{row['seed']}.pt")) for row in old['entries']];save(folder/'prospective_specs.json',dict(entries=specs,change='All training routes in each ensemble member instead of route bootstrap; all inference thresholds/options fixed relative to short_margin_v1.',no_new_final_routes=True))
 with awake():
  results=[];entries=[]
  for seed,spec in zip(range(501,506),specs):
   save(OUT/'status.json',dict(stage='h5_full_support_'+str(seed),status='running'));r=run(spec,base,folder/f'seed_{seed}');m=compare(r,base);results.append(r);entries.append(dict(seed=seed,spec=spec,metrics=m));print('FULL_SUPPORT_GATE',seed,m,flush=True)
  close('h5_full_support_validation','Complete-data ensemble compared with route-bootstrap ensemble using unchanged inference and complete development union.',entries=entries,statistics=paired(results,base),gains=sum(r['metrics']['gains'] for r in entries),losses=sum(r['metrics']['losses'] for r in entries),cases_per_model=len(base['episodes']),development_folder=str(folder),seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='h5_full_support_ready_for_review',status='running'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','evaluate'],required=True);a=p.parse_args();globals()[a.phase]()
