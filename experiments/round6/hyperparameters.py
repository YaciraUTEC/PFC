"""Small prospective architecture-specific search, before the5seed final fits."""
import argparse,json,time,subprocess,sys
from .evaluate import OUT,save
from .train_offpolicy import fit
from .models import ARCHS

def specifications():
    jobs=[]
    for arch in ARCHS:
        for value in [.015,.05,.15]:jobs.append(dict(arch=arch,method='flashsac',tag=f'sigma_{value:g}',params=dict(target_sigma=value)))
        for value in [1.,10.,30.]:jobs.append(dict(arch=arch,method='fql',tag=f'alpha_{value:g}',params=dict(fql_alpha=value)))
    return jobs

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);a=p.parse_args()
    while not (OUT/'phase2_conclusion.json').exists():time.sleep(5)
    if a.worker is not None:
        for j,job in enumerate(specifications()):
            if j%2!=a.worker:continue
            dest=OUT/'hpo'/job['arch']/job['method']/job['tag'];dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():continue
            print('HPO_START',job,flush=True)
            fit(job['arch'],191,job['method'],dest,budget=65536,offline_updates=10000,prior_seed=201,validation_start=611064,validation_count=8,**job['params'])
        return
    start=time.time();logs=[];processes=[]
    save(OUT/'hpo_protocol.json',dict(jobs=specifications(),model_seed=191,prior_seed=201,online_steps=65536,offline_steps_fql=10000,validation_routes=list(range(611064,611072)),selected_before_final_model_fits=True))
    for worker in [0,1]:
        f=(OUT/f'hpo_worker_{worker}.log').open('a',encoding='utf-8');logs.append(f)
        processes.append(subprocess.Popen([sys.executable,'-m','experiments.round6.hyperparameters','--worker',str(worker)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)))
    for process in processes:process.wait()
    for f in logs:f.close()
    assert all(p.returncode==0 for p in processes),'HPO worker failed; see logs'
    selected={};results=[]
    for arch in ARCHS:
        selected[arch]={}
        for method in ['flashsac','fql']:
            rr=[]
            for job in specifications():
                if job['arch']!=arch or job['method']!=method:continue
                dest=OUT/'hpo'/arch/method/job['tag'];complete=json.loads((dest/'complete.json').read_text());rr.append(dict(**job,score=complete['best_score'],path=str(dest),seconds=complete['seconds']))
            # Parameter list order is the prospective reproducible tie-break.
            best=max(rr,key=lambda r:r['score']);selected[arch][method]=best['params'];results.extend(rr)
    save(OUT/'hpo_selection.json',dict(selected=selected,results=results,seconds=time.time()-start,finished_epoch=time.time(),final_fql_offline_updates=100000,
        interpretation='One development seed per architecture;final comparison uses5fresh head seeds. Native recurrent priors are from the closed phase2 choice. No final-test inputs.'))
    print('OFFPOLICY_HPO_SELECTED',selected,flush=True)
if __name__=='__main__':main()
