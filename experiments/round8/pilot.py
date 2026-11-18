"""First causal controls on explicitly reused development routes from round7."""
import time,argparse
from pathlib import Path
import numpy as np
import torch
from .common import OUT,R7,read,save,close
from .cache import run,certify
from .policy import PreservePolicy
from ..round7.evaluation import cases,key,compare,evaluate
from ..round7.power import awake

def baseline_subset(cc):
 full=read(R7/'final/primary/baseline.json');lookup={key(r):r for r in full['episodes']};rows=[lookup[key(r)] for r in cc]
 return dict(episodes=rows,source=str(R7/'final/primary/baseline.json'),scope='Previously inspected round7 test now explicitly reused as development; not a new held-out test.')

def main():
 while not (OUT/'h1_factor_fits_conclusion.json').exists():time.sleep(5)
 torch.set_num_threads(1);start=time.time();save(OUT/'status.json',dict(stage='h1_flight_pilot',status='running'))
 routes=sorted(set([740000+4*i for i in range(16)]+[740029,740035,740038]));cc=cases(routes,levels=[0.,.25,.295,.296]);base=baseline_subset(cc);save(OUT/'pilot/baseline.json',base)
 prior=read(R7/'selected_candidate.json')['entries'][0];common=dict(proposer=prior['proposer'],native=True,max_options=1,minimum_gain=.005,penalty=1.)
 factor=dict(common,gate=str(OUT/'factor_gates/initial/seed_501.pt'),gate_kind='factor',risk_mode='conditional')
 flat=dict(common,gate=prior['gate'],gate_kind='flat')
 specs=[('flat_once_joint',dict(flat,risk_mode='joint',minimum_gain=.02,maximum_harm=.005)),('flat_once_conditional',dict(flat,risk_mode='conditional',maximum_harm=.005)),('factor_strict',dict(factor,maximum_harm=.005)),('factor_moderate',dict(factor,maximum_harm=.02)),('factor_relaxed',dict(factor,maximum_harm=.05)),('factor_strict_penalty2',dict(factor,maximum_harm=.005,penalty=2.))]
 save(OUT/'pilot/prospective_specs.json',dict(specs=specs,routes=routes,cases=len(cc),selection='Pilot only: inspect gains and losses; no final evidence or tuning on new final tests.'))
 selected=[r for r in base['episodes'] if r['seed'] in routes[:3] and r['level'] in [.295,.296]];hist=[]
 for row in selected:
  with np.load(row['trace_path']) as z:hist.append(z['observation'].copy())
 flags=[]
 for device in ['cpu','cuda']:
  flags.append(certify(PreservePolicy.from_spec(specs[4][1]).to(device).double(),hist))
 assert np.array_equal(*flags);save(OUT/'pilot/cpu_cuda_preflight.json',dict(cases=len(hist),flags=flags[0],equal=True,scope='Conservative OFF checks agree before optimization use; full-flight fresh equivalence is checked separately.'))
 rows=[]
 with awake():
  for name,spec in specs:
   result=run(spec,base,OUT/'pilot'/name);metric=compare(result,base);rows.append(dict(name=name,spec=spec,metrics=metric));print('PILOT_RESULT',name,metric,flush=True)
 # Re-run a mixed set fresh, including potentially active cases.
 spec=specs[4][1];mixed=selected[:12];fresh=evaluate(PreservePolicy.from_spec(spec),[{k:r[k] for k in ['seed','level','condition','noise','clock']} for r in mixed],workers=8)
 cached=run(spec,dict(episodes=mixed),OUT/'pilot/preflight_subset');lookup={key(r):r for r in fresh['episodes']};largest=0.
 for row in cached['episodes']:
  other=lookup[key(row)];assert row['outcome']==other['outcome'] and row['episode_steps']==other['episode_steps'];err=float(np.abs(np.asarray(row['terminal_position'])-other['terminal_position']).max());largest=max(largest,err);assert err<1e-8
 save(OUT/'pilot/full_flight_preflight.json',dict(cases=len(mixed),largest_position_difference=largest,equal=True))
 close('h1_flight_pilot','Causal controls compare once-only continuation, conditional risk and factorized learning. These inspected routes are development, not held-out evidence.',results=rows,routes=len(routes),cases=len(cc),seconds=time.time()-start)
 save(OUT/'status.json',dict(stage='h1_ready_for_review',status='running'))
if __name__=='__main__':main()
