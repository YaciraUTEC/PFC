"""Reuse only natural controller flights independent of the rejected branch labels.
A matching frozen specification and weights are required, with a fresh paired
check spanning nominal/critical, exact/estimated and both route ends. Any failed
check sends the entire controller to fresh simulation. No old learned gate is reused.
"""
import time
import numpy as np
from .common import OUT,LEGACY_OUT,read,save,sha,config
from .evaluation import cases,evaluate,key
from .execution import settings
from .policy import TransformerPolicy


def baseline_for(case_list,folder):
    path=OUT/folder/'baseline.json';old=(OUT/'gate_flights'/'baseline.json') if folder.startswith('gate_flights_verified_') else LEGACY_OUT/folder/'baseline.json'
    if path.exists() or not old.exists():return None
    data=read(old)
    if {key(r) for r in data['episodes']}!={key(r) for r in case_list}:return None
    if read(LEGACY_OUT/'h0_reproduction_conclusion.json')['baseline_config']!=config():return None
    if not all(Path(r['trace_path']).exists() for r in data['episodes']):return None
    data=dict(data,natural_baseline_reused_from=str(old),natural_baseline_source_sha=sha(old));save(path,data);return data


def reuse(name,spec,case_list,folder):
    if spec.get('mode') not in ['native','forced'] or spec.get('gate'):return None
    path=(OUT/'gate_flights'/(name+'.json')) if folder.startswith('gate_flights_verified_') else LEGACY_OUT/folder/(name+'.json')
    if not path.exists():return None
    old=read(path);fingerprint=old.get('fingerprint',{})
    if fingerprint.get('spec')!=spec or fingerprint.get('proposer_sha')!=sha(spec['proposer']):return None
    lookup={key(r):r for r in old['episodes']}
    if set(lookup)!={key(r) for r in case_list}:return None
    low=min(r['seed'] for r in case_list);high=max(r['seed'] for r in case_list)
    desired=[(low,0.,'exact',1.),(low,.295,'estimated',2.),(high,.296,'exact',2.),(high,.296,'estimated',2.)]
    sample=[]
    for seed,level,condition,noise in desired:
        sample.append(next(r for r in case_list if r['seed']==seed and r['level']==level and r['condition']==condition and r['noise']==noise))
    model=TransformerPolicy.from_spec(spec);opts=settings()
    if opts['fast_gtrxl']:
        from .fast_gtrxl import install
        model=install(model)
    checked=evaluate(model,sample,workers=min(opts['workers'],4),inference_device=opts['inference_device']);largest=0.;good=True
    for row in checked['episodes']:
        prior=lookup[key(row)];error=float(np.abs(np.asarray(prior['terminal_position'])-row['terminal_position']).max());largest=max(largest,error)
        good &=row['outcome']==prior['outcome'] and row['episode_steps']==prior['episode_steps'] and error<1e-8
    receipt=dict(accepted=bool(good),legacy_path=str(path),legacy_sha=sha(path),checks=len(sample),maximum_position_error=largest,seconds=checked['seconds'],execution=opts,scope='Natural flights without a learned outer gate; independent of the rejected counterfactual labels.')
    save(OUT/'unchanged_control_checks'/folder/(name+'.json'),receipt)
    if not good:return None
    fresh={key(r):r for r in checked['episodes']};rows=[dict(fresh[key(r)],fresh_simulation=True) if key(r) in fresh else dict(r,fresh_simulation=False,unchanged_control_result_reused=True) for r in old['episodes']]
    return dict(episodes=rows,seconds=checked['seconds'],logical_cases=len(rows),fresh_simulations=len(sample),verified_off_reuse=0,unchanged_control_reuse=len(rows)-len(sample),spec=spec,execution_settings=opts,verified_natural_control_reuse=receipt)

from pathlib import Path
