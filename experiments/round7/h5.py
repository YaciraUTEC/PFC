"""Frozen holdout evaluation; no hyperparameter selection occurs here."""
import argparse,time,json,platform,sys
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,OLD,LEVELS,read,save,sha,old_model,close_milestone
from .policy import TransformerPolicy
from .evaluation import cases,evaluate,key,compare
from .cache import baseline,run
from .power import awake


def preflight():
    if not (OUT/'h4_protected_finetuning_conclusion.json').exists():raise RuntimeError('Close development first')
    chosen=read(OUT/'selected_candidate.json')['entries'];cc=cases(range(719980,719984),levels=[0.,.295,.296],noises=[1.,2.])
    base=baseline(cc,OUT/'execution_preflight');checks=[]
    specs=chosen+[dict(proposer=str(old_model(s)),mode='native') for s in range(201,206)]
    for index,spec in enumerate(specs):
        folder=OUT/'execution_preflight'/str(index);cached=run(spec,base,folder);fresh=evaluate(TransformerPolicy.from_spec(spec),cc)
        aa={key(r):r for r in cached['episodes']};bb={key(r):r for r in fresh['episodes']};largest=0.
        for case in aa:
            a,b=aa[case],bb[case]
            assert a['outcome']==b['outcome'] and a['stable_landing']==b['stable_landing'],(index,case,a['outcome'],b['outcome'])
            assert a['episode_steps']==b['episode_steps']
            error=float(np.max(np.abs(np.asarray(a['terminal_position'])-b['terminal_position'])));largest=max(largest,error)
            assert error<1e-8,(index,case,error)
        from .composed_replay import certify
        histories=[]
        for row in base['episodes']:
            with np.load(row['trace_path']) as z:histories.append(z['observation'].copy())
        cpu_flags,_=certify(TransformerPolicy.from_spec(spec),histories)
        from .execution import settings
        proof_model=TransformerPolicy.from_spec(spec).to(settings()['certification_device']).double();other_flags,_=certify(proof_model,histories);assert np.array_equal(cpu_flags,other_flags),'Certification device disagrees';del proof_model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        checks.append(dict(certification_flags_equal=True,index=index,cases=len(cc),largest_terminal_position_difference=largest,off_reuse=cached['verified_off_reuse'],fresh=cached['fresh_simulations']))
    close_milestone('h5_execution_preflight','All five selected model replicas and all five reference replicas agree between conservative OFF reuse and fresh full simulations on development-only routes.',checks=checks)


def freeze():
    path=OUT/'frozen_before_final.json'
    if path.exists():verify();return
    if not (OUT/'h5_execution_preflight_conclusion.json').exists():raise RuntimeError('Run complete execution preflight first')
    tests=read(OUT/'h5_tests_receipt.json');assert tests['exit_code']==0
    sources={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.relative_to(ROOT).parts for x in ['runs','.git','__pycache__'])}
    selected=read(OUT/'selected_candidate.json');assert len(selected['entries'])==5
    files={str(OLD/'phase1_conclusion.json'),str(old_model(201)),str(OUT/'execution_settings.json')}
    for spec in selected['entries']:
        files.add(spec['proposer'])
        if spec.get('gate'):files.add(spec['gate'])
    files.update(str(old_model(s)) for s in range(201,206))
    artifacts={p:sha(p) for p in sorted(files)}
    from ..round4.landing import LandingCriteria
    save(path,dict(created_epoch=time.time(),sources=sources,artifacts=artifacts,selection_sha=sha(OUT/'selected_candidate.json'),
        selected=selected,criteria=LandingCriteria().dictionary(),clock_challenge=[4.,5.,7.,8.],python=sys.version,torch=torch.__version__,numpy=np.__version__,
        primary_routes=[740000,740064],confirmation_routes=[750000,750320],shift_routes=[760000,760032],
        primary='strict success at29.5/29.6 estimated, mean of both noises',acceptance='zero new losses in selected-family development; lower95CI gain>0 on primary AND confirmation; zero observed new losses in all protected tests/clock challenges; full-chain p95<20.833ms',
        statement='Reporting artifacts may be generated; frozen source and selected policy artifacts may not change. No final result is used for tuning.'))
    verify()


def verify():
    f=read(OUT/'frozen_before_final.json')
    for rel,value in f['sources'].items():assert sha(ROOT/rel)==value,'Frozen source changed '+rel
    for path,value in f['artifacts'].items():assert sha(path)==value,'Frozen model/input changed '+path
    assert sha(OUT/'selected_candidate.json')==f['selection_sha']
    return f


def job(name,spec,base,folder):
    path=folder/name/'results.json'
    if path.exists():return read(path)
    verify()
    for previous in sorted(folder.glob('candidate_*/results.json')):
        prior=read(previous)
        if prior.get('spec')!=spec:continue
        assert {key(r) for r in prior['episodes']}=={key(r) for r in base['episodes']}
        result=dict(prior,episodes=[dict(r,identical_policy_result_reused=True,baseline_reused_verified_off=False,fresh_simulation=False) for r in prior['episodes']],fresh_simulations=0,verified_off_reuse=0,identical_policy_reuse=len(prior['episodes']),seconds=0.,identical_policy_source=str(previous),proof_scope='Exactly the same frozen policy specification and cases were already evaluated; this is a named-reference reuse, not an additional independent flight.')
        save(path,result);save(folder/name/'identical_policy_proof.json',dict(spec=spec,source=str(previous),source_sha=sha(previous),proposer_sha=sha(spec['proposer']),cases=len(prior['episodes'])));verify();print('IDENTICAL_FROZEN_POLICY_REUSE',name,len(prior['episodes']),flush=True);return result
    result=run(spec,base,folder/name);verify();print('FINAL_MODEL_JOB',folder.name,name,result['fresh_simulations'],result['verified_off_reuse'],flush=True);return result


def main():
    start=time.time();freeze();selection=read(OUT/'selected_candidate.json')['entries']
    with awake():
        phases=[('primary',cases(range(740000,740064),levels=LEVELS),True),
            ('confirmation',cases(range(750000,750320),levels=[.295,.296]),False),
            ('shifted',cases(range(760000,760032),levels=[.295,.296],conditions=['estimated'],noises=[2.],clocks=[4.,5.,7.,8.]),True)]
        for label,cc,reference in phases:
            marker=OUT/('h5_'+label+'_complete.json')
            if marker.exists():continue
            before=time.time();folder=OUT/'final'/label
            save(OUT/'status.json',dict(stage='h5_'+label,status='running',started_epoch=before))
            base=baseline(cc,folder)
            entries=[]
            for i,spec in enumerate(selection):
                result=job('candidate_'+str(301+i),spec,base,folder);entries.append(dict(name='candidate_'+str(301+i),metrics=compare(result,base),fresh=result['fresh_simulations'],reused=result['verified_off_reuse'],identical_policy_reuse=result.get('identical_policy_reuse',0)))
            if reference:
                for s in range(201,206):
                    spec=dict(proposer=str(old_model(s)),mode='native');result=job('reference_'+str(s),spec,base,folder)
                    entries.append(dict(name='reference_'+str(s),metrics=compare(result,base),fresh=result['fresh_simulations'],reused=result['verified_off_reuse'],identical_policy_reuse=result.get('identical_policy_reuse',0)))
            save(marker,dict(complete=True,cases_per_model=len(cc),baseline_cases=len(cc),entries=entries,seconds=time.time()-before))
            print('FROZEN_PHASE_COMPLETE',label,round(time.time()-before),flush=True)
        verify();close_milestone('h5_frozen_test','All prospectively allocated untouched routes, confirmation and shifted-clock challenges completed with unchanged models and criteria.',seconds=time.time()-start)
        save(OUT/'status.json',dict(stage='h5_closed',status='complete'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args();preflight() if a.preflight else main()
