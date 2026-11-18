"""Development-only simplification: bounded half-second native option and stronger benefit margin."""
import time
import torch
from .common import OUT,read,save,close
from .cache import run
from ..round7.evaluation import compare
from ..round7.statistics import paired
from ..round7.power import awake

def main():
 torch.set_num_threads(1);start=time.time();old=read(OUT/'h5_fixed_expert_validation_conclusion.json');folder=OUT/'development/short_margin_v1';base=read(OUT/'development/fixed_expert201/baseline.json');save(folder/'baseline.json',base)
 specs=[dict(row['spec'],minimum_gain=.01,allowed_options=[1]) for row in old['entries']]
 save(folder/'prospective_specs.json',dict(entries=specs,reason='All but one development rescue use native24tick half-amplitude option. The sole fixed201 loss begins late with lower gain .006596. Isolate bounded option plus stronger benefit margin; whole policy must be rerun because changing first authorization is not nested.',source_harm_case=[810150,.296,'estimated',2.,None],scope='Development only, no final route opened; hypothesis, not safety proof.'))
 with awake():
  results=[];entries=[]
  for seed,spec in zip(range(501,506),specs):
   save(OUT/'status.json',dict(stage='h5_conservative_'+str(seed),status='running'));r=run(spec,base,folder/f'seed_{seed}');m=compare(r,base);results.append(r);entries.append(dict(seed=seed,spec=spec,metrics=m));print('CONSERVATIVE_GATE',seed,m,flush=True)
  close('h5_conservative_validation','Whole-policy bounded native option and stronger benefit margin evaluated on all development cases.',entries=entries,statistics=paired(results,base),gains=sum(r['metrics']['gains'] for r in entries),losses=sum(r['metrics']['losses'] for r in entries),cases_per_model=len(base['episodes']),seconds=time.time()-start)
  save(OUT/'status.json',dict(stage='h5_conservative_ready_for_review',status='running'))
if __name__=='__main__':main()
