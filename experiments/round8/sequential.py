"""Predeclared independent-episode evidence for iterative candidate evaluation.
For a frozen policy and an independently assigned critical episode per scenario,
D is +1 rescue, -1 lost base success, 0 tie. Under the iid mixture null
P(rescue)<=P(loss), the multiplier 2/0/1 has expectation<=1. Thus
E_n=2**G_n*1[L_n==0] is a nonnegative supermartingale. Ville's inequality permits
stopping/continuing with a fixed candidate. Revision-wise alpha spending handles
adaptive new candidates tested on fresh independent scenarios. This statistical
statement requires iid scenarios and independent actor/regime assignment; it
is not a plant-safety or out-of-distribution guarantee.
"""
import math
import numpy as np

def spending(attempt):
 if attempt<1:raise ValueError('Attempts start at one')
 return dict(benefit_alpha=.025/(2**attempt),preservation_alpha=.025/(2**attempt),total_across_infinite_revisions=.05)

def evidence(baseline_success,candidate_success,attempt=1,maximum_conditional_harm=.005):
 b=np.asarray(baseline_success,bool);c=np.asarray(candidate_success,bool);assert b.shape==c.shape and b.ndim==1
 gain=int((~b&c).sum());loss=int((b&~c).sum());protected=int(b.sum());alpha=spending(attempt);log_e=gain*math.log(2) if loss==0 else None
 # On the zero-loss branch this is time-uniform: crossing U_n<p implies
 # no loss in the first ceil(log(alpha)/log(1-p)) protected iid episodes.
 upper=-math.expm1(math.log(alpha['preservation_alpha'])/protected) if protected and loss==0 else (1. if protected==0 else None)
 positive=loss==0 and log_e>=-math.log(alpha['benefit_alpha']);preserve=loss==0 and upper is not None and upper<=maximum_conditional_harm
 return dict(scenarios=len(b),gains=gain,losses=loss,protected_baseline_successes=protected,log_e_value=log_e,e_value=(2**gain if loss==0 else 0),benefit_alpha=alpha['benefit_alpha'],preservation_alpha=alpha['preservation_alpha'],positive_gain_evidence=bool(positive),conditional_harm_upper=upper,maximum_conditional_harm=maximum_conditional_harm,preservation_evidence=bool(preserve),both_statistical_conditions=bool(positive and preserve),scope='One independently assigned estimated-effectiveness critical episode and model per independent scenario. Zero observed losses is still required in every additional stress/curve test. Bound covers the fixed simulated mixture, not universal safety or the risk conditional on actual neural actuation.')

def assignment(seeds,random_seed,model_seeds=(501,502,503,504,505)):
 rng=np.random.default_rng(random_seed);rows=[]
 for seed in seeds:rows.append(dict(seed=int(seed),level=float(rng.choice([.295,.296])),condition='estimated',noise=float(rng.choice([1.,2.])),clock=None,model_seed=int(rng.choice(model_seeds))))
 assert len({r['seed'] for r in rows})==len(rows)
 return rows
