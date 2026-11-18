"""Counterfactual factual arm must reproduce previously executed whole options."""
import argparse,time
import numpy as np
from .common import OUT,read,save,close

def key(r):return tuple(r[k] for k in ['seed','level','condition','noise','clock'])
def main(wait=False):
 if wait:
  while not (OUT/'h6_active_data_conclusion.json').exists():time.sleep(5)
 allocation=read(OUT/'data/active_v1/allocation.json');plan=read(OUT/'data/active_v1/collection_plan.json');entries={key(x['case']):x for x in plan['entries']};scanned={key(x['case']):x for x in allocation['entries']};checked=missing=0;maximum=0.;failures=[]
 for mi,spec in enumerate(allocation['models']):
  family='short_margin_v1' if mi<5 else 'full_support_v1';seed=501+mi%5;old=read(OUT/f'development/{family}/seed_{seed}/results.json');assert old['spec']==spec
  for actual in old['episodes']:
   start=actual.get('policy_summary',{}).get('option_start_seconds')
   if start is None:continue
   tick=round(start*48);entry=entries[key(actual)];p=OUT/'data/active_v1'/entry['split']/(entry['tag']+'.json')
   if not p.exists():missing+=1;continue
   data=read(p);rows=[r for r in data['records'] if r['tick']==tick and r['future']==0 and r['option']==1];assert len(rows)==1,(mi,key(actual),tick);candidate=rows[0]['candidate'];checked+=1
   same=all(candidate[k]==actual[k] for k in ['outcome','episode_steps','stable_landing','stable_seconds']);a=candidate['first_contact'];b=actual['first_contact'];same &= (a is None)==(b is None)
   if a is not None and b is not None:
    diff=max(abs(float(a[k])-float(b[k])) for k in a);maximum=max(maximum,diff);same &=diff<1e-8
   if not same:failures.append(dict(model=mi,case=key(actual),tick=tick,counterfactual=candidate,actual={k:actual[k] for k in ['outcome','episode_steps','stable_landing','stable_seconds','first_contact']}))
 result=dict(checked_whole_option_pairs=checked,pending=missing,maximum_contact_measure_difference=maximum,failures=failures,passed=not failures and (not wait or missing==0),scope='Factual native24half option from identical natural state versus the actually executed old compound policy at its exact first decision; matching outcomes, steps, stable duration and contact measures. Multiple gates sharing a tick are not independent flights.')
 save(OUT/'h6_factual_option_equivalence.json',result);print('FACTUAL_OPTION_EQUIVALENCE',checked,'pending',missing,'failures',len(failures),'max_contact_error',maximum,flush=True);assert not failures
 if wait:assert missing==0;close('h6_factual_option_equivalence','All factual counterfactual native arms reproduce the previous actual full-option flights.',**result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();main(a.wait)
