"""Sequential full-flight authorizer selection and physical-screen ablation."""
import argparse,time,hashlib
from .common import OUT,read,save,sha,old_model,close_milestone
from .policy import TransformerPolicy
from .evaluation import evaluate,cases,compare

RUNTIME=['common.py','episode.py','policy.py','proposal_model.py','screen.py','evaluation.py','cache.py','composed_replay.py','execution.py','fast_gtrxl.py','unchanged_controls.py']
def runtime_digest():
    here=__import__('pathlib').Path(__file__).parent
    return hashlib.sha256(''.join(sha(here/p) for p in RUNTIME).encode()).hexdigest()
def run(name,spec,cc,baseline,folder):
    dest=OUT/folder/(name+'.json');fingerprint=dict(spec=spec,proposer_sha=sha(spec['proposer']),gate_sha=sha(spec['gate']) if spec.get('gate') else None,runtime=runtime_digest(),execution=__import__('experiments.round7.execution',fromlist=['settings']).settings())
    if dest.exists():
        d=read(dest)
        if d.get('fingerprint')!=fingerprint:raise RuntimeError('Changed inputs for completed development job '+name)
        return dict(name=name,spec=spec,metrics=compare(d,baseline),path=str(dest))
    from .cache import run as cached_run
    from .unchanged_controls import reuse
    result=reuse(name,spec,cc,folder)
    if result is None:result=cached_run(spec,baseline,OUT/folder/name)
    result['fingerprint']=fingerprint;save(dest,result)
    metric=compare(result,baseline);print('FULL_FLIGHT_STUDY',folder,name,metric,flush=True)
    return dict(name=name,spec=spec,metrics=metric,path=str(dest))
def rank(row):
    m=row['metrics'];return (m['losses']==0,m['critical_difference'],m['success'],-m['losses'])
def proposers(seed=301):
    paths=read(OUT/'selected_proposal.json')['paths_by_seed']
    return dict(flashsac=str(old_model(seed-100)),bc=paths[str(seed)])

def baseline_for(cc,folder):
    from .cache import baseline
    from .unchanged_controls import baseline_for as retained_baseline
    existing=retained_baseline(cc,folder)
    return existing if existing is not None else baseline(cc,OUT/folder)

def gates():
    if not (OUT/'h2_gate_model_conclusion.json').exists():raise RuntimeError('Fit and assess gate model first')
    while not (OUT/'execution_settings.json').exists():time.sleep(5)
    assert read(OUT/'execution_settings.json')['validated']
    folder='gate_flights_verified_'+sha(OUT/'execution_settings.json')[:8]
    start=time.time();cc=cases(range(710004,710012),levels=[0.,.25,.295,.296]);base=baseline_for(cc,folder);rows=[];selected={}
    for source,path in proposers().items():
        rows.append(run(source+'_native',dict(proposer=path,mode='native'),cc,base,folder))
        local=[]
        for name,gain,harm,u in [('conservative',.02,.005,1.),('balanced',.005,.02,1.),('mean',0.,.02,0.)]:
            spec=dict(proposer=path,mode='counterfactual',gate=str(OUT/'outcome_gate.pt'),minimum_gain=gain,maximum_harm=harm,uncertainty_penalty=u)
            row=run(source+'_'+name,spec,cc,base,folder);rows.append(row);local.append(row)
        selected[source]=max(local,key=rank)
    rows.append(run('flashsac_raised_threshold',dict(proposer=proposers()['flashsac'],mode='native',native_threshold=.02),cc,base,folder))
    reference=read(OUT/'h2_original_cpu_evaluation_conclusion.json');original={r['name']:r for r in reference['results']};equivalence=[]
    for row in rows:
        current=read(row['path']);prior=read(original[row['name']]['path']);lookup={tuple(r[k] for k in ['seed','level','condition','noise','clock']):r for r in prior['episodes']};largest=0.
        for r in current['episodes']:
            old=lookup[tuple(r[k] for k in ['seed','level','condition','noise','clock'])]
            assert r['outcome']==old['outcome'] and r['episode_steps']==old['episode_steps'],('Execution optimization changed outcome',row['name'],r['seed'])
            error=max(abs(float(x)-float(y)) for x,y in zip(r['terminal_position'],old['terminal_position']));largest=max(largest,error);assert error<1e-8,('Execution optimization changed trajectory',row['name'],r['seed'],error)
        equivalence.append(dict(name=row['name'],cases=len(current['episodes']),maximum_position_error=largest))
    close_milestone('h2_execution_equivalence','All nine128-case controller comparisons preserve the completed original CPU evaluation under verified execution optimizations.',checks=equivalence)
    close_milestone('h2_counterfactual_gate','Paired-outcome authorization evaluated in full flights against the old gate and a raised-threshold control; advance to a separate filter ablation.',
        cases=len(cc),routes=8,selected_by_source=selected,results=rows,seconds=time.time()-start,
        no_formal_confidence_claim=True)

def filters():
    if not (OUT/'h2_counterfactual_gate_conclusion.json').exists():raise RuntimeError('Close gate flights first')
    start=time.time();cc=cases(range(710012,710020),levels=[0.,.25,.295,.296]);base=baseline_for(cc,'filter_flights');rows=[]
    g=read(OUT/'h2_counterfactual_gate_conclusion.json')['selected_by_source']
    for source,path in proposers().items():
        native=dict(proposer=path,mode='native');cf=dict(g[source]['spec']);forced=dict(proposer=path,mode='forced')
        for name,spec in [('forced',forced),('native',native),('counterfactual',cf)]:
            rows.append(run(source+'_'+name,spec,cc,base,'filter_flights'))
            rows.append(run(source+'_'+name+'_screen',dict(spec,screen=dict(allowance=0.,uncertainty_multiplier=1.)),cc,base,'filter_flights'))
    extra=dict(proposer=proposers()['flashsac'],mode='native',native_threshold=.02)
    rows.append(run('flashsac_raised_threshold',extra,cc,base,'filter_flights'))
    rows.append(run('flashsac_raised_threshold_screen',dict(extra,screen=dict(allowance=0.,uncertainty_multiplier=1.)),cc,base,'filter_flights'))
    best=max(rows,key=rank)
    close_milestone('h3_filter_ablation','Complete 3x2 ablation separates proposal, authorizer and actuator-aware screen. Selection remains subject to five-seed validation and timing.',
        cases=len(cc),routes=8,selected=best,results=rows,seconds=time.time()-start,screen_is_not_formal_reachability=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['gate','filter'],required=True);a=p.parse_args();gates() if a.phase=='gate' else filters()
