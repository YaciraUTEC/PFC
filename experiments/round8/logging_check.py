"""A reporting-only amendment must preserve the previously evaluated flights."""
import numpy as np
import torch
from .common import OUT,read,save,close,sha
from .evaluation import evaluate
from .policy import PreservePolicy
from ..round7.evaluation import key

def main():
 torch.set_num_threads(1);folder=OUT/'development/natural_native_v1';old=read(folder/'seed_501/results.json');base=read(folder/'baseline.json');lookup={key(r):r for r in old['episodes']};routes={(740029,1.),(740029,2.),(740031,2.),(740035,1.)};rows=[r for r in base['episodes'] if (r['seed'],r['noise']) in routes and r['level']==.296 and r['condition']=='estimated'];subset=dict(episodes=rows)
 snapshot=OUT/'source_history/h3_before_logging'
 for name in ['policy.py','policy_v2.py','factor_gate.py','options.py','cache.py']:assert sha(snapshot/name)==old['fingerprint'][name]
 new=evaluate(PreservePolicy.from_spec(old['spec']),[{k:r[k] for k in ['seed','level','condition','noise','clock']} for r in rows],workers=4,capture_directory=OUT/'logging_verification/traces',capture_baseline=subset,path=OUT/'logging_verification/results.json');largest=0.
 for r in new['episodes']:
  prior=lookup[key(r)];assert r['outcome']==prior['outcome'] and r['episode_steps']==prior['episode_steps'];error=float(np.abs(np.asarray(r['terminal_position'])-prior['terminal_position']).max());largest=max(largest,error);assert error==0.
  summary=r['policy_summary']
  if summary['options_started']:assert summary['initial_forecasts']['conditional_harm']<old['spec']['maximum_harm']
 close('h4_logging_equivalence','Per-flight summaries and actual decision-risk logging preserve all prior outcomes, steps and terminal positions exactly; inference source history retained.',cases=len(rows),maximum_position_difference=largest)
if __name__=='__main__':main()
