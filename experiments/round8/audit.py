"""Descriptive changed-flight audit with actual motor actions and initial forecasts."""
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from .common import read,save
from ..round7.evaluation import key
from .options import OPTIONS

def summarize(result,baseline):
 lookup={key(r):r for r in baseline['episodes']};changed=[]
 for r in result['episodes']:
  b=lookup[key(r)]
  if r['stable_landing']==b['stable_landing']:continue
  row=dict(kind='rescue' if r['stable_landing'] else 'loss',case={k:r[k] for k in ['seed','level','condition','noise','clock']},baseline_outcome=b['outcome'],candidate_outcome=r['outcome'],baseline_steps=b['episode_steps'],candidate_steps=r['episode_steps'],baseline_contact=b.get('first_contact'),candidate_contact=r.get('first_contact'),trace_path=r.get('trace_path'))
  if r.get('trace_path'):
   with np.load(r['trace_path']) as z:
    starts=np.flatnonzero(z['policy_option_started']);assert len(starts)==1;tick=int(starts[0]);option=int(z['policy_selected_option'][tick]);duration=min(OPTIONS[option][0],1440-tick);activity=np.abs(z['projected_motor_delta']).max(-1)>1e-7;actual=np.flatnonzero(activity)
    with np.load(b['trace_path']) as original:prefix=float(np.abs(z['observation'][:tick]-original['observation'][:tick]).max()) if tick else 0.
    assert prefix==0.,'History diverged before the first authorized option'
    row.update(option_start_tick=tick,option_start_seconds=tick/48,option_id=option,option_description=OPTIONS[option],duration=duration,actual_motor_intervention_steps=int(activity.sum()),first_actual_motor_intervention_seconds=float(actual[0]/48) if len(actual) else None,prefix_observation_difference=prefix,initial_estimated_eta=z['estimated_eta_before'][tick].tolist(),initial_true_eta_for_audit_only=z['true_eta_before'][tick].tolist(),initial_physical_delta=z['projected_motor_delta'][tick].tolist())
    forecasts={}
    for name in ['base_success','preservation','joint_rescue','joint_harm','conditional_harm','lower_gain']:
     field='policy_option_initial_'+name
     if field in z:forecasts[name]=float(z[field][tick])
    row['initial_forecasts']=forecasts;row['forecast_scope']=('Conditional_harm is the actual decision risk under '+result['spec'].get('risk_aggregation','weighted_std')) if 'policy_summary' in r else ('Legacy development log stores weighted mean+std conditional_harm; actual decision used '+result['spec'].get('risk_aggregation','weighted_std'))
  changed.append(row)
 return dict(changed=changed,gains=sum(r['kind']=='rescue' for r in changed),losses=sum(r['kind']=='loss' for r in changed),unique_changed_routes=len({r['case']['seed'] for r in changed}),outcomes=dict(Counter(r['candidate_outcome'] for r in changed)),scope='Post-hoc descriptive audit, no policy selection on reserved final tests; prefix equality identifies the first divergence, not the isolated cause of the terminal outcome.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=summarize(read(a.result),read(a.baseline));save(a.output,r);print(r)
