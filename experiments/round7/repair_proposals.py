"""Finite causal repair of the demonstrated mixed-supervision conflict."""
import time,math,argparse
import numpy as np
import torch
from ..round5.bc import setup,tensor
from ..round5.losses import clone_loss
from ..round6.models import configuration
from .common import OUT,save,read,close_milestone
from .data import load,ProposalSequences
from .proposal_model import ProposalAgent
from .proposal_validation import samples,fidelity
from .evaluation import evaluate,cases,compare

RECIPES={
 'original':dict(teacher_only=False,proposal_gate='learned',direct_motor_head=False),
 'teacher_gate':dict(teacher_only=True,proposal_gate='learned',direct_motor_head=False),
 'teacher_physical':dict(teacher_only=True,proposal_gate='observed_fault',direct_motor_head=False),
 'teacher_direct':dict(teacher_only=True,proposal_gate='observed_fault',direct_motor_head=True)}

def loss_for(model,mu,target,mask,valid,recipe):
    if recipe in ['original','teacher_gate']:return clone_loss(mu,target,mask,valid)
    actual=target[...,:4]*(target[...,4:5]+1)/2
    error=(model.physical_delta(mu)-actual).square().mean(-1)
    return 3*(error*mask).sum()/mask.sum().clamp_min(1)

def fit(recipe,groups,seed,updates=3600,phase='scale'):
    dest=OUT/'proposal_repair'/phase/recipe/f'groups_{groups}'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    start=time.time();setup(seed);data=load(groups)
    if RECIPES[recipe]['teacher_only']:data['records']=[r for r in data['records'] if r['behavior']!='baseline']
    dataset=ProposalSequences(data);cfg=dict(configuration('gtrxl'),proposal_repair=True,**RECIPES[recipe]);model=ProposalAgent(cfg).cuda();opt=model.optimizer(1e-4);rng=np.random.default_rng(seed)
    validation=samples();metrics=[];best=float('inf')
    checkpoints=[updates] if phase=='smoke' else [1200,2400,3600]
    for step in range(1,updates+1):
        lr=1e-4*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*step/updates)))
        for g in opt.param_groups:g['lr']=lr
        x,a,mask,valid=dataset.sample(rng);model.train();mu,_,_=model(tensor(x));loss=loss_for(model,mu,tensor(a),tensor(mask),tensor(valid),recipe)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite proposal repair')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
        if step%200==0:save(dest/'progress.json',dict(update=step,loss=float(loss),seconds=time.time()-start));print('PROPOSAL_REPAIR',recipe,groups,seed,step,round(float(loss),6),flush=True)
        if step in checkpoints:
            model.eval();path=dest/f'checkpoint_{step}.pt';model.save(path,recipe=recipe,groups=groups,seed=seed,updates=step)
            metric=fidelity(path,validation);metrics.append(dict(update=step,**metric));save(dest/'fidelity.json',metrics)
            if metric['active_mse']<best:best=metric['active_mse'];model.save(dest/'best.pt',recipe=recipe,groups=groups,seed=seed,updates=step)
            print('REPAIR_FIDELITY',recipe,groups,seed,step,metric,flush=True)
    model.save(dest/'final.pt',recipe=recipe,groups=groups,seed=seed,updates=updates)
    result=dict(recipe=recipe,groups=groups,seed=seed,updates=updates,seconds=time.time()-start,started_epoch=start,finished_epoch=time.time(),fidelity=metrics,
        final_active_mse=metrics[-1]['active_mse'],best_active_mse=best,teacher_episodes=len(dataset.records),active_episodes=len(dataset.active_records),path=str(dest/'final.pt'))
    save(dest/'complete.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','scale','complete'],required=True);a=p.parse_args()
    if a.phase=='smoke':
        rows=[fit(r,256,301,1200,'smoke') for r in RECIPES];best=min(rows,key=lambda r:r['final_active_mse'])
        close_milestone('h1_repair_screen','Four matched-budget recipes isolate conflicting demonstration targets, gate coupling and physical output parameterization.',selected_recipe=best['recipe'],results=rows)
    elif a.phase=='scale':
        recipe=read(OUT/'h1_repair_screen_conclusion.json')['selected_recipe'];rows=[fit(recipe,g,s) for g in [0,256] for s in [301,302,303]]
        means={g:float(np.mean([r['final_active_mse'] for r in rows if r['groups']==g])) for g in [0,256]};winner=min(means,key=means.get)
        close_milestone('h1_repaired_scale','Original and expanded demonstrations compared again with the measured best proposal-only recipe and the same training budget.',selected_recipe=recipe,selected_groups=winner,means=means,results=rows)
    else:
        selected=read(OUT/'h1_repaired_scale_conclusion.json');recipe=selected['selected_recipe'];g=selected['selected_groups']
        rows=[fit(recipe,g,s) for s in [301,302,303,304,305]]
        paths={str(r['seed']):r['path'] for r in rows}
        save(OUT/'selected_proposal.json',dict(recipe=recipe,groups=g,paths_by_seed=paths,selection='independent active-action fidelity; native flight and paired authorizer evaluation remain necessary'))
        close_milestone('h1_supervision_repair','Proposal supervision is consistent; five replicas are ready for independent terminal authorization learning.',selected_recipe=recipe,selected_groups=g,results=rows)
if __name__=='__main__':main()
