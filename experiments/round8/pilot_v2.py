"""Prospective natural-base option pilot; no final route is consumed."""
import time
import numpy as np
import torch
from .common import OUT,R7,read,save,close
from .cache import run,certify
from .policy import PreservePolicy
from .pilot import baseline_subset
from ..round7.evaluation import cases,compare,key,evaluate
from ..round7.power import awake

def main(tag='base_options_v1',label='h2_flight_pilot',proposer_id=201,compact=False):
 assert (OUT/'h2_factor_fits_conclusion.json').exists()
 torch.set_num_threads(1);start=time.time();routes=sorted(set([740000+4*i for i in range(16)]+[740029,740035,740038]));cc=cases(routes,levels=[0.,.25,.295,.296]);base=baseline_subset(cc);folder=OUT/'pilots'/tag/('proposer_'+str(proposer_id));save(folder/'baseline.json',base)
 prior=read(R7/'selected_candidate.json')['entries'][proposer_id-201];specs=[]
 for mode in ['native','raw','both']:
  for risk in [.005,.02]:
   spec=dict(version=2,proposer=prior['proposer'],gate=str(OUT/'factor_gates'/tag/'seed_501.pt'),minimum_gain=.005,maximum_harm=risk,penalty=1.,option_mode=mode)
   specs.append((mode+'_risk_'+str(risk),spec))
 for gain in [.02]:
  for risk in [.005,.02]:specs.append(('both_gain_'+str(gain)+'_risk_'+str(risk),dict(specs[-1][1],minimum_gain=gain,maximum_harm=risk)))
 # Training-label audit suggests isolated one-tick options have much lower harm
 # than long unvetoed feedback. Control that factor before selecting a policy.
 for option in [0,4,5]:
  for risk in [.005,.02]:specs.append(('option_'+str(option)+'_risk_'+str(risk),dict(specs[5][1],option_mode='both',allowed_options=[option],minimum_gain=.005,maximum_harm=risk)))
 for allowed in [None,[4]]:specs.append(('all_members_'+('both' if allowed is None else 'raw_one_tick'),dict(specs[5][1],risk_aggregation='max_member',allowed_options=allowed,maximum_harm=.02)))
 if compact:specs=[(name,spec) for name,spec in specs if name in ['both_risk_0.005','both_risk_0.02','option_4_risk_0.005','option_4_risk_0.02']]
 save(folder/'prospective_specs.json',dict(specs=specs,routes=routes,cases=len(cc),role='Development pilot only; thresholds not tuned on new held-out tests.'))
 subset=[r for r in base['episodes'] if r['seed'] in routes[:3] and r['level'] in [.295,.296]];hist=[]
 for row in subset:
  with np.load(row['trace_path']) as z:hist.append(z['observation'].copy())
 flags=[]
 for device in ['cpu','cuda']:flags.append(certify(PreservePolicy.from_spec(specs[-1][1]).to(device).double(),hist))
 assert np.array_equal(*flags);save(folder/'cpu_cuda_preflight.json',dict(cases=len(hist),equal=True,flags=flags[0]))
 result=[]
 with awake():
  for name,spec in specs:
   r=run(spec,base,folder/name);metric=compare(r,base);result.append(dict(name=name,spec=spec,metrics=metric));print('NATURAL_BASE_PILOT',name,metric,flush=True)
 # Fresh source/reference inference includes cases not declared OFF.
 pilot=next((row for row in result if row['name']=='both_risk_0.02'),result[0]);full=read(folder/pilot['name']/'results.json');active=[r for r in full['episodes'] if r['fresh_simulation']];checks=(active[:8]+[r for r in base['episodes'] if r['level'] in [.295,.296]][:8])[:16]
 checks=list({key(r):r for r in checks}.values())
 if checks:
  fresh=evaluate(PreservePolicy.from_spec(pilot['spec']),[{k:r[k] for k in ['seed','level','condition','noise','clock']} for r in checks],workers=8);lookup={key(r):r for r in full['episodes']};largest=0.
  for r in fresh['episodes']:
   old=lookup[key(r)];assert r['outcome']==old['outcome'] and r['episode_steps']==old['episode_steps'];error=float(np.abs(np.asarray(r['terminal_position'])-old['terminal_position']).max());largest=max(largest,error);assert error<1e-8
  save(folder/'full_flight_preflight.json',dict(cases=len(checks),equal=True,largest_position_error=largest))
 close(label,'Base-distribution native/raw/both option policies compared with strict conditional preservation, identical controller and unchanged landing contract.',results=result,routes=len(routes),cases=len(cc),seconds=time.time()-start)
 save(OUT/'status.json',dict(stage=label+'_ready_for_review',status='running'))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--tag',default='base_options_v1');p.add_argument('--label',default='h2_flight_pilot');p.add_argument('--proposer-id',type=int,default=201);p.add_argument('--compact',action='store_true');a=p.parse_args();main(a.tag,a.label,a.proposer_id,a.compact)
