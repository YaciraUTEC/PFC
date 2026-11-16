"""Terminal paired options on student states with independently sampled futures."""
import argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
import torch
from ..round6.env import LandingEnv
from .common import OUT,config,read,save,silent,old_model
from .episode import load_policy,action_from_delta
from .policy import canonical_risk,risk_features,propose,OPTIONS,controller_descriptor
from .replay_fork import ReplayFork
from .branches import continuation,feedback_continuation,outcome_class


def near_boundary(row):
    c=row.get('first_contact')
    if c is not None:
        for name,threshold,tol in [('vertical_speed',.25,1e-4),('horizontal_speed',.25,1e-4),('radius',.20,1e-4),('tilt',np.deg2rad(12),1e-4),('angular_speed',1.,1e-3)]:
            if abs(float(c[name])-threshold)<tol:return True
    if row['stable_landing'] and row['episode_steps']>=1438:return True
    if row['outcome']=='timeout' and row['stable_seconds']>=47/48:return True
    return False

def compact(row):
    return {k:row.get(k) for k in ['outcome','stable_landing','episode_steps','first_contact','stable_seconds']}

def selected_proposers():
    selected=read(OUT/'selected_proposal.json')['paths_by_seed']
    return [str(old_model(s)) for s in range(201,206)]+[selected[str(s)] for s in [301,302,303,304,305]]


def case_settings(index,count):
    block=index//count
    return dict(controller=index%count,condition=['exact','estimated'][block%2],noise=[1.,2.][(block//2)%2],clock=[None,4.,6.,8.][(block//4)%4])

def collect_group(index,split,proposers,behavior_spec=None):
    torch.set_num_threads(1);folder=OUT/'gate_data'/split;dest=folder/f'group_{index:03d}.npz';meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        cached=read(meta)
        if cached.get('fork_method')!='natural_prefix_replay':raise RuntimeError('Reject non-natural cached branch labels')
        return cached
    start=time.time();seed=(720000 if split=='train' else 730000)+index
    cell=case_settings(index,len(proposers));condition=cell['condition'];noise=cell['noise'];clock=cell['clock']
    policy_path=proposers[index%len(proposers)];descriptor=controller_descriptor(policy_path);model=load_policy(policy_path);risk=canonical_risk().double().eval()
    if behavior_spec is not None:
        from .policy import TransformerPolicy
        behavior_model=TransformerPolicy.from_spec(behavior_spec)
    else:behavior_model=None
    with silent():e=LandingEnv(condition=condition,config=config(noise_multiplier=noise,fault_clock=clock))
    features=[];labels=[];metadata=[];ambiguous=0;repeat_checks=0;prefix_checks=0;maximum_prefix_error=0.;trace=[];trace_indices=[];trace_starts=[];episode_boundaries=[]
    levels=[.295,.296]+([0.,.25] if index%16==0 else [])
    try:
        for level in levels:
            with silent():obs,_=e.reset(seed=seed,options=dict(level=level))
            pm=rm=bm=None;history=[];queries=0;phase_counts=Counter();episode_start=len(trace);episode_boundaries.append(episode_start);was_native=False;last_query_tick=-1000
            while True:
                trace.append(obs[-1].copy());trace_index=len(trace)-1
                x=torch.as_tensor(obs[-1:,:],dtype=torch.float64)
                with torch.no_grad():
                    delta,native,pm=propose(model,x,pm);f,rm=risk_features(risk,x,rm)
                raw=delta[0].numpy() if bool(native[0]) else np.zeros(4);behavior=action_from_delta(raw)
                if behavior_model is not None:
                    with torch.no_grad():executed,bm=behavior_model.predict_actions(x,bm)
                    behavior=executed[0].numpy()
                phase='terminal' if e.route_reached else 'route'
                due=((e.t>=48 and e.t%(48 if e.route_reached else 96)==0) or (e.t>=12 and bool(native[0]) and not was_native)) and e.t-last_query_tick>=24
                observable=bool(e.estimated_eta.min()<.98 or e.route_reached)
                if due and observable and not e.touchdown_seen and queries<6 and phase_counts[phase]<3:
                    tick=e.t;last_query_tick=tick;queries+=1;phase_counts[phase]+=1
                    with ReplayFork(e,history) as fork:
                        for future in range(2):
                            future_seed=9000000+seed*31+tick*7+future
                            branch=fork.restore();branch.sensor_rng=np.random.default_rng(future_seed);base=continuation(branch)
                            for option,(budget,alpha) in enumerate(OPTIONS):
                                duration=min(budget,1440-tick)
                                branch=fork.restore();branch.sensor_rng=np.random.default_rng(future_seed)
                                cand=feedback_continuation(branch,model,pm,raw,alpha,duration)
                                ambiguous+=int(near_boundary(base) or near_boundary(cand))
                                if near_boundary(base) or near_boundary(cand):continue
                                label=outcome_class(base['stable_landing'],cand['stable_landing'])
                                feat=np.r_[f[0].numpy(),raw*alpha,descriptor,duration/96].astype(np.float32)
                                features.append(feat);labels.append(label);trace_indices.append(trace_index);trace_starts.append(episode_start)
                                metadata.append(dict(seed=seed,level=level,condition=condition,noise=noise,clock=clock,tick=tick,phase=phase,future=future,option=option,
                                    duration=duration,alpha=alpha,base=compact(base),candidate=compact(cand),label=label,proposer=policy_path))
                            if future==0 and queries==1:
                                branch=fork.restore();branch.sensor_rng=np.random.default_rng(future_seed);repeat=continuation(branch)
                                assert repeat['outcome']==base['outcome'] and repeat['episode_steps']==base['episode_steps']
                                np.testing.assert_allclose(repeat['terminal_position'],base['terminal_position'],atol=0,rtol=0);repeat_checks+=1
                        prefix_checks+=fork.checks;maximum_prefix_error=max(maximum_prefix_error,fork.maximum_error)
                history.append(np.asarray(behavior,np.float32).copy())
                was_native=bool(native[0]);obs,_,d,tr,info=e.step(behavior)
                if d or tr:break
    finally:e.close()
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,x=np.asarray(features,np.float32).reshape(-1,196),y=np.asarray(labels,np.int64),seed=np.full(len(labels),seed,np.int64),trace=np.asarray(trace,np.float32),trace_index=np.asarray(trace_indices,np.int64),trace_start=np.asarray(trace_starts,np.int64),episode_boundaries=np.asarray(episode_boundaries+[len(trace)],np.int64),future=np.asarray([r['future'] for r in metadata],np.int64))
    result=dict(seed=seed,index=index,split=split,seconds=time.time()-start,labels=dict(Counter(labels)),samples=len(labels),ambiguous_excluded=ambiguous,repeat_checks=repeat_checks,records=metadata,
        semantics=['both_success','rescue','harm','both_fail'],deployable_input_dimension=196,terminal_comparison=True,futures_per_state=2,feedback_options=True,native_veto_preserved=True,fork_method='natural_prefix_replay',prefix_replay_checks=prefix_checks,maximum_prefix_error=maximum_prefix_error,behavior_spec=behavior_spec,behavior_distribution='selected composed predecessor' if behavior_spec is not None else 'native proposer')
    save(meta,result);return result


def _collect(split,count=160,workers=8,proposers=None,behavior_specs=None):
    start=time.time();proposers=selected_proposers() if proposers is None else proposers;rows=[]
    reused=sum((OUT/'gate_data'/split/f'group_{i:03d}.npz').exists() and (OUT/'gate_data'/split/f'group_{i:03d}.json').exists() for i in range(count))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(collect_group,i,split,proposers,None if behavior_specs is None else behavior_specs[i%len(proposers)]) for i in range(count)]):
            r=f.result();rows.append(r);print('GATE_PAIRED_DATA',split,r['index'],'samples',r['samples'],'labels',r['labels'],'seconds',round(r['seconds']),flush=True)
    labels=Counter()
    for r in rows:labels.update({int(k):v for k,v in r['labels'].items()})
    summary=dict(split=split,groups=count,samples=sum(r['samples'] for r in rows),labels=dict(labels),seconds=time.time()-start,
        ambiguous_excluded=sum(r['ambiguous_excluded'] for r in rows),repeat_checks=sum(r['repeat_checks'] for r in rows),proposers=proposers,fork_method='natural_prefix_replay',prefix_replay_checks=sum(r['prefix_replay_checks'] for r in rows),maximum_prefix_error=max(r['maximum_prefix_error'] for r in rows))
    summary.update(reused_groups=reused,new_groups=count-reused,actual_proposer_label_counts=dict(Counter(record['proposer'] for row in rows for record in row['records'])))
    save(OUT/f'gate_data_{split}_complete.json',summary);return summary


def collect(split,count=160,workers=8,proposers=None,behavior_specs=None):
    from ..round6.resource_control import validation_slot
    folder=OUT/'gate_data'/split;folder.mkdir(parents=True,exist_ok=True)
    with validation_slot(folder):
        marker=OUT/f'gate_data_{split}_complete.json'
        if marker.exists() and read(marker)['groups']==count:
            assert all((folder/f'group_{i:03d}.npz').exists() and (folder/f'group_{i:03d}.json').exists() for i in range(count))
            return read(marker)
        return _collect(split,count,workers,proposers,behavior_specs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','calibration'],required=True);p.add_argument('--groups',type=int,default=160);p.add_argument('--workers',type=int,default=8);p.add_argument('--proposers-json');p.add_argument('--behavior-specs-json');a=p.parse_args()
    if not (OUT/'h1_proposal_selection_conclusion.json').exists():raise RuntimeError('Close the proposal-quality stage before starting h2')
    print('GATE_DATA_COMPLETE',collect(a.split,a.groups,a.workers,read(a.proposers_json) if a.proposers_json else None,read(a.behavior_specs_json) if a.behavior_specs_json else None),flush=True)
