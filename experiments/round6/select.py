"""Independent validation closes each learning phase before the next begins."""
import argparse,json,time
import numpy as np
import torch
from .models import Agent,ARCHS
from .offpolicy import OffPolicy
from .evaluate import OUT,SEEDS,save
from .train import selected_config
from .phase1 import objective
from .replay import cache_baseline,evaluate_cached

SCHEDULES=['nominal','linear','adaptive','full']
def model_entries(phase):
    entries=[]
    if phase==2:
        for arch in ARCHS:
            for schedule in SCHEDULES:
                for seed in SEEDS:entries.append(dict(kind='ppo',arch=arch,schedule=schedule,seed=seed,path=str(OUT/'phase2'/arch/f'seed_{seed}'/schedule/'best.pt')))
    else:
        decision=json.loads((OUT/'phase2_conclusion.json').read_text())
        for arch in ARCHS:
            for method in ['flashsac','fql']:
                for seed in SEEDS:entries.append(dict(kind=method,arch=arch,schedule=decision['schedule_by_arch'][arch],seed=seed,path=str(OUT/'phase3'/method/arch/f'seed_{seed}'/'best.pt')))
    return entries

def validate_entry(entry):
    path=__import__('pathlib').Path(entry['path']);assert path.exists() and path.with_name('complete.json').exists(),str(path)
    model=Agent.load(path,'cpu').double() if entry['kind']=='ppo' else OffPolicy.load(path,'cpu').double()
    cfg=selected_config();allrows=[]
    for condition in ['exact','estimated']:
        for noise in [1.,2.]:
            rows,folder=cache_baseline('selected',cfg,condition,noise,range(611008,611016),[0.,.25,.28,.29,.295,.296],role='validation')
            label=f"{entry['kind']}_{entry['arch']}_{entry['schedule']}_{entry['seed']}_{condition}_noise{noise:g}"
            resultpath=OUT/'validation'/f'{label}.json'
            if not resultpath.exists():save(resultpath,dict(entry=entry,**evaluate_cached(model,rows,folder,cfg,condition,noise)))
            result=json.loads(resultpath.read_text());allrows.extend(dict(r,condition=condition,noise=noise,training_seed=entry['seed']) for r in result['episodes'])
    return dict(entry=entry,score=objective(allrows),episodes=allrows,mean_intervention=float(np.mean([r['applied_on_fraction'] for r in allrows])))

def phase2():
    results=[];start=time.time()
    for e in model_entries(2):
        print('PHASE2_VALIDATION',e['arch'],e['schedule'],e['seed'],flush=True);results.append(validate_entry(e))
    schedules={};summary={}
    for arch in ARCHS:
        summary[arch]={}
        for sch in SCHEDULES:
            rr=[r for r in results if r['entry']['arch']==arch and r['entry']['schedule']==sch]
            summary[arch][sch]=dict(mean_score=float(np.mean([r['score'] for r in rr])),sd_score=float(np.std([r['score'] for r in rr],ddof=1)),
                mean_intervention=float(np.mean([r['mean_intervention'] for r in rr])),training_seeds=len(rr))
        schedules[arch]=max(SCHEDULES,key=lambda sch:summary[arch][sch]['mean_score'])
    conclusion=dict(schedule_by_arch=schedules,summary=summary,results=results,finished_epoch=time.time(),seconds=time.time()-start,
        decision_basis='Independent validation routes611008..611015;five training seeds;all test routes still closed.',
        scope='Nominal/full-range controls distinguish augmentation exposure from the noise-range schedule.')
    save(OUT/'phase2_conclusion.json',conclusion)
    (OUT/'CONCLUSION_FASE2.txt').write_text('FASE 2 CERRADA\nSchedules seleccionados: '+json.dumps(schedules)+'\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('PHASE2_CONCLUSION',schedules,flush=True)

def phase3():
    results=[];start=time.time()
    for e in model_entries(3):
        print('PHASE3_VALIDATION',e['kind'],e['arch'],e['seed'],flush=True);results.append(validate_entry(e))
    p2=json.loads((OUT/'phase2_conclusion.json').read_text());combined=list(results)
    for r in p2['results']:
        if r['entry']['schedule']==p2['schedule_by_arch'][r['entry']['arch']]:combined.append(r)
    summary={}
    for arch in ARCHS:
        for method in ['ppo','flashsac','fql']:
            rr=[r for r in combined if r['entry']['arch']==arch and r['entry']['kind']==method]
            summary[f'{arch}_{method}']=dict(score=float(np.mean([r['score'] for r in rr])),sd=float(np.std([r['score'] for r in rr],ddof=1)),intervention=float(np.mean([r['mean_intervention'] for r in rr])))
    best=max(summary,key=lambda k:summary[k]['score']);selected=[r for r in combined if r['entry']['arch']+'_'+r['entry']['kind']==best]
    selected=sorted(selected,key=lambda r:r['score']);representative=selected[len(selected)//2]['entry']
    save(OUT/'phase3_conclusion.json',dict(best_family=best,representative=representative,summary=summary,results=results,finished_epoch=time.time(),seconds=time.time()-start,
        acceptance='Family is selected on validation; independent test/confirmation may accept it or retain the base. No further hyperparameter changes.'))
    (OUT/'CONCLUSION_FASE3.txt').write_text('FASE 3 CERRADA\nFamilia candidata: '+best+'\n'+json.dumps(summary,indent=2),encoding='utf-8')
    print('PHASE3_CONCLUSION',best,flush=True)
if __name__=='__main__':
    torch.set_num_threads(1);p=argparse.ArgumentParser();p.add_argument('--phase',type=int,choices=[2,3],required=True);a=p.parse_args();phase2() if a.phase==2 else phase3()
