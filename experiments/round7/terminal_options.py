"""Finite extension: add whole-policy continuations without discarding short options.
Original archives are copied byte-for-byte before atomically replacing upgraded ones.
The full state is recreated by replaying the same native student and checking history.
"""
import time,argparse,shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from .common import OUT,config,read,save,silent,close_milestone,sha
from .episode import load_policy,action_from_delta
from .policy import canonical_risk,risk_features,propose,controller_descriptor
from .gate_data import near_boundary,compact
from .snapshot import Snapshot
from .branches import continuation,feedback_continuation,outcome_class
from ..round6.env import LandingEnv


def _upgrade_group(index,split):
    torch.set_num_threads(1);folder=OUT/'gate_data'/split;path=folder/f'group_{index:03d}.npz';meta=path.with_suffix('.json');old=read(meta)
    if old.get('terminal_option_upgrade'):return old['terminal_option_upgrade']
    started=time.time()
    with np.load(path) as a:z={k:a[k].copy() for k in a.files}
    records=old['records'];lookup={}
    for r in records:lookup.setdefault((r['level'],r['tick'],r['future']),r['base'])
    if not records:raise RuntimeError('Expected at least one paired query per original group')
    first=records[0];seed=old['seed'];model=load_policy(first['proposer']);risk=canonical_risk().double().eval();identity=controller_descriptor(first['proposer'])
    with silent():e=LandingEnv(condition=first['condition'],config=config(noise_multiplier=first['noise'],fault_clock=first['clock']))
    extra=[];labels=[];added=[];indices=[];starts=[];trace=[];checks=0;excluded=0;boundaries=[]
    levels=[.295,.296]+([0.,.25] if index%16==0 else [])
    try:
        for level in levels:
            with silent():obs,_=e.reset(seed=seed,options=dict(level=level))
            pm=rm=None;queries=0;counts=Counter();begin=len(trace);boundaries.append(begin);was_native=False;previous=-1000
            while True:
                trace.append(obs[-1].copy());q=len(trace)-1;x=torch.as_tensor(obs[-1:],dtype=torch.float64)
                with torch.no_grad():delta,native,pm=propose(model,x,pm);f,rm=risk_features(risk,x,rm)
                raw=delta[0].numpy() if bool(native[0]) else np.zeros(4);phase='terminal' if e.route_reached else 'route'
                due=((e.t>=48 and e.t%(48 if e.route_reached else 96)==0) or(e.t>=12 and bool(native[0]) and not was_native)) and e.t-previous>=24
                if due and bool(e.estimated_eta.min()<.98 or e.route_reached) and not e.touchdown_seen and queries<6 and counts[phase]<3:
                    tick=e.t;previous=tick;queries+=1;counts[phase]+=1;duration=1440-tick
                    with Snapshot(e) as snap:
                        for future in range(2):
                            future_seed=9000000+seed*31+tick*7+future;base=lookup.get((level,tick,future))
                            if base is None or(queries==1 and future==0):
                                snap.restore();e.sensor_rng=np.random.default_rng(future_seed);repeated=continuation(e)
                                if base is not None:
                                    assert repeated['outcome']==base['outcome'] and repeated['episode_steps']==base['episode_steps'] and repeated['stable_seconds']==base['stable_seconds'];checks+=1
                                base=compact(repeated)
                            snap.restore();e.sensor_rng=np.random.default_rng(future_seed);candidate=feedback_continuation(e,model,pm,raw,1.,duration)
                            if near_boundary(base) or near_boundary(candidate):excluded+=1;continue
                            label=outcome_class(base['stable_landing'],candidate['stable_landing'])
                            extra.append(np.r_[f[0].numpy(),raw,identity,duration/96].astype(np.float32));labels.append(label);indices.append(q);starts.append(begin)
                            added.append(dict(seed=seed,level=level,condition=first['condition'],noise=first['noise'],clock=first['clock'],tick=tick,phase=phase,future=future,option=3,duration=duration,budget_ticks=1440,alpha=1.,base=base,candidate=compact(candidate),label=label,proposer=first['proposer'],terminal_feedback_option=True))
                was_native=bool(native[0]);obs,_,d,t,_=e.step(action_from_delta(raw))
                if d or t:break
    finally:e.close()
    reconstructed=np.asarray(trace,np.float32);assert reconstructed.shape==z['trace'].shape
    error=float(np.max(np.abs(reconstructed-z['trace'])));assert error<2e-6,error
    np.testing.assert_array_equal(boundaries+[len(trace)],z['episode_boundaries'])
    backup=OUT/'gate_data_three_options'/split;backup.mkdir(parents=True,exist_ok=True)
    for original in [path,meta]:
        target=backup/original.name
        if not target.exists():shutil.copy2(original,target)
        assert sha(original)==sha(target)
    for i,r in enumerate(records):
        budget=r['duration'];r['budget_ticks']=budget;r['duration']=min(budget,1440-r['tick']);z['x'][i,-1]=r['duration']/96
    z['x']=np.concatenate([z['x'],np.asarray(extra,np.float32).reshape(-1,196)])
    z['y']=np.r_[z['y'],np.asarray(labels,np.int64)];z['seed']=np.r_[z['seed'],np.full(len(labels),seed,np.int64)]
    z['trace_index']=np.r_[z['trace_index'],np.asarray(indices,np.int64)];z['trace_start']=np.r_[z['trace_start'],np.asarray(starts,np.int64)];z['future']=np.asarray([r['future'] for r in records+added],np.int64)
    temporary=path.with_suffix('.upgrade.npz');np.savez_compressed(temporary,**z);temporary.replace(path)
    receipt=dict(seconds=time.time()-started,added_labels=len(labels),added_counts=dict(Counter(labels)),replay_observation_max_error=error,baseline_repeat_checks=checks,ambiguous_excluded=excluded,original_archive_sha=sha(backup/path.name))
    old.update(records=records+added,samples=len(z['y']),labels=dict(Counter(map(int,z['y']))),terminal_option_upgrade=receipt)
    save(meta,old);print('TERMINAL_OPTION_UPGRADE',split,index,receipt,flush=True);return receipt


def upgrade_group(index,split):
    from ..round6.resource_control import validation_slot
    folder=OUT/'terminal_option_locks'/split/f'group_{index:03d}';folder.mkdir(parents=True,exist_ok=True)
    with validation_slot(folder):return _upgrade_group(index,split)


def independent_split(split):
    while not (OUT/f'gate_data_{split}_complete.json').exists():time.sleep(5)
    start=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(upgrade_group,i,split) for i in range(160)]):rows.append(f.result())
    close_milestone('h2_terminal_'+split+'_acquisition_parallel','Independent split acquisition completed under per-group locks; the main extension driver alone updates the global data receipt.',seconds=time.time()-start,groups=160,new_labels=sum(r['added_labels'] for r in rows))


def extend(split,workers=8):
    while not (OUT/f'gate_data_{split}_complete.json').exists():time.sleep(5)
    start=time.time();rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(upgrade_group,i,split) for i in range(160)]):rows.append(f.result())
    total=Counter();sources=Counter();samples=0
    for i in range(160):
        r=read(OUT/'gate_data'/split/f'group_{i:03d}.json');total.update({int(k):v for k,v in r['labels'].items()});samples+=r['samples'];sources.update(x['proposer'] for x in r['records'])
    p=OUT/f'gate_data_{split}_complete.json';summary=read(p);summary.update(samples=samples,labels=dict(total),terminal_option_upgrade=True,terminal_option_upgrade_seconds=time.time()-start,actual_proposer_label_counts=dict(sources));save(p,summary)
    return dict(split=split,groups=160,seconds=time.time()-start,new_labels=sum(r['added_labels'] for r in rows),total_labels=dict(total),repeat_checks=sum(r['baseline_repeat_checks'] for r in rows),max_history_error=max(r['replay_observation_max_error'] for r in rows))


def main(pilot=False):
    if pilot:
        r=upgrade_group(-100,'pilot');close_milestone('h0_terminal_option_pilot','Whole-policy extension preserves student histories and paired baseline outcomes; pilot remains excluded.',result=r);return
    assert read(OUT/'h0_terminal_option_pilot_conclusion.json')['closed']
    rows=[extend(split) for split in ['train','calibration']]
    previous=OUT/'h2_terminal_labels_conclusion.json'
    if previous.exists():shutil.copy2(previous,OUT/'h2_terminal_labels_three_options.json')
    close_milestone('h2_terminal_labels','Full feedback and whole-policy continuations compared with DSLPID until strict terminal; all four options and route-separated calibration ready.',audits={split:read(OUT/f'gate_data_{split}_complete.json') for split in ['train','calibration']})
    close_milestone('h2_terminal_option_upgrade','All160 train and160 calibration groups include whole-policy versus whole-baseline terminal outcomes; short-option data preserved and originals backed up.',results=rows)
if __name__=='__main__':
    if OUT.name.endswith('_replay'):raise RuntimeError('Historical save/restore upgrade is disabled for natural replay datasets')
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--split',choices=['train','calibration']);a=p.parse_args();independent_split(a.split) if a.split else main(a.pilot)
