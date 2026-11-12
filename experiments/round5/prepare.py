"""Preparation pipeline; fail-fast and resumable at completed artifact boundaries."""
import json,time,subprocess,sys,datetime
from pathlib import Path
from .evaluate import OUT
from ..run import save

def stage(label,args):
    receipt=OUT/f'{label}_stage.json'
    if receipt.exists():
        print('PREPARATION_REUSE_COMPLETED_STAGE',label,flush=True);return
    start=time.time();log=OUT/f'{label}.log';save(OUT/'preparation_status.json',dict(stage=label,status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    with log.open('a',encoding='utf-8') as f:
        subprocess.run([sys.executable,*args],cwd=Path.cwd(),stdout=f,stderr=subprocess.STDOUT,check=True)
    save(OUT/f'{label}_stage.json',dict(seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('PREPARATION_COMPLETE',label,'seconds',round(time.time()-start),flush=True)

def audit_data():
    import numpy as np
    from .data import load
    data=load();records=data['records'];train={r['seed'] for r in records if r['split']=='train'};validation={r['seed'] for r in records if r['split']=='validation'}
    assert train<=set(range(526000,526048)) and validation<=set(range(527000,527012)) and not train&validation
    for key in ['s','ns','a','observer','executed','reward','eta']:assert np.isfinite(data[key]).all(),key
    for key in ['a','observer','executed']:assert np.max(np.abs(data[key]))<=1.+1e-6 and set(np.unique(data[key][:,5]))<=set([-1.,1.])
    for row in records:
        start,stop=row['start'],row['stop'];assert data['done'][stop-1] and not data['done'][start:stop-1].any()
        np.testing.assert_array_equal(data['ns'][start:stop-1],data['s'][start+1:stop])
        if row['condition']=='exact':assert np.all(data['a'][start:stop,5]==-1) and np.all(data['observer'][start:stop,5]==-1)
    from .data import SequenceData
    result=dict(episodes=len(records),transitions=len(data['s']),successes=sum(r['stable_landing'] for r in records),train_routes=len(train),validation_routes=len(validation),
        paired_proposals=len(data['branches']),residual_positive_labels=len(SequenceData(data).positive),observer_positive_labels=len(SequenceData(data,'effectiveness').positive),
        train_test_disjoint=True,transition_chain_verified=True,executed_actions_separate_from_teacher_labels=True)
    save(OUT/'data_audit.json',result);print('DATA_AUDIT',json.dumps(result),flush=True)
    return data

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    from .bootstrap import initialize
    initialize(OUT)
    target=OUT/'core_selection.json'
    if not target.exists() or json.loads(target.read_text()).get('engineering_version')!=4:stage('core_engineering',['-m','experiments.round5.select_core'])
    if not (OUT/'structural_challenger.json').exists():stage('governor_validation',['-m','experiments.round5.structural_challenger'])
    if not (OUT/'paired_physics_audit.json').exists() or not (OUT/'physics_precision_summary.json').exists():stage('physics_precision',['-m','experiments.round5.physics_audit'])
    if not (OUT/'observer_fallback_selection.json').exists():stage('observer_only_validation',['-m','experiments.round5.observer_fallback'])
    stage('collect_initial',['-m','experiments.round5.data','--workers','6','--count','60'])
    stage('bc_pilots',['-m','experiments.round5.bc','--phase','pilots'])
    stage('collect_dagger',['-m','experiments.round5.bc','--phase','dagger'])
    data=audit_data()
    from .judge import fit
    if not (OUT/'judge_complete.json').exists():fit(data);save(OUT/'judge_complete.json',dict(completed=True))
    del data
    stage('hpo',['-m','experiments.round5.train','--phase','hpo'])
    save(OUT/'preparation_complete.json',dict(seconds=time.time()-start,completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    save(OUT/'preparation_status.json',dict(stage='ready_for_full_training',status='complete'))
if __name__=='__main__':main()
