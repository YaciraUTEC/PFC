"""Matched GTrXL demonstration-scale study with an independent flight validation."""
import argparse,time,math
import numpy as np
import torch
from ..round5.bc import setup,tensor
from ..round5.losses import clone_loss
from ..round6.models import Agent,configuration
from .data import load,ProposalSequences
from .common import OUT,save,read,close_milestone,old_model
from .episode import load_policy
from .evaluation import cases,evaluate,compare


def _fit(groups,seed,updates=3600):
    dest=OUT/'bc_curve'/f'groups_{groups}'/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    setup(seed);start=time.time();data=load(groups);dataset=ProposalSequences(data)
    m=Agent(configuration('gtrxl')).cuda();opt=m.optimizer(1e-4);rng=np.random.default_rng(seed);logs=[];validation=[];best=-1e9
    cc=cases(range(710000,710004),levels=[0.,.25,.295,.296])
    basepath=OUT/'development/bc_baseline.json'
    base=read(basepath) if basepath.exists() else evaluate(None,cc,path=basepath)
    checkpoints=[600,1200,2400,3600]
    for it in range(1,updates+1):
        lr=1e-4*min(1.,it/200)*(.1+.9*.5*(1+math.cos(math.pi*it/updates)))
        for group in opt.param_groups:group['lr']=lr
        x,a,mask,valid=dataset.sample(rng);m.train();mu,_,_=m(tensor(x));loss=clone_loss(mu,tensor(a),tensor(mask),tensor(valid))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite proposal clone')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
        if it%200==0:
            row=dict(update=it,loss=float(loss),lr=lr,seconds=time.time()-start);logs.append(row);save(dest/'progress.json',row);print('GTRXL_BC',groups,seed,it,round(float(loss),5),flush=True)
        if it in checkpoints or it==updates:
            cpu=Agent(configuration('gtrxl')).double().eval();cpu.load_state_dict({k:v.detach().cpu().double() for k,v in m.state_dict().items()})
            result=evaluate(cpu,cc,path=dest/f'validation_{it}.json');metrics=compare(result,base);score=metrics['success']
            validation.append(dict(update=it,**metrics));save(dest/'validation.json',validation)
            if score>best:best=score;m.save(dest/'best.pt',groups=groups,seed=seed,updates=it,validation=metrics)
            print('GTRXL_BC_VALIDATION',groups,seed,it,metrics,flush=True);del cpu
    m.save(dest/'final.pt',groups=groups,seed=seed,updates=updates)
    result=dict(groups=groups,seed=seed,seconds=time.time()-start,updates=updates,validation=validation,best_success=best,
        data_episodes=len(data['records']),active_demonstration_episodes=len(dataset.active_records),path=str(dest/'best.pt'))
    save(dest/'complete.json',result);return result


def fit(groups,seed,updates=3600):
    from ..round6.resource_control import validation_slot
    folder=OUT/'gpu_training';folder.mkdir(parents=True,exist_ok=True)
    with validation_slot(folder):return _fit(groups,seed,updates)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--groups',nargs='+',type=int,default=[16,64,256]);parser.add_argument('--seeds',nargs='+',type=int,default=[301,302,303]);parser.add_argument('--updates',type=int,default=3600);a=parser.parse_args()
    if not (OUT/'h0_reproduction_conclusion.json').exists():raise RuntimeError('Close h0 first')
    results=[fit(g,s,a.updates) for g in a.groups for s in a.seeds]
    if a.groups==[16,64,256] and a.seeds==[301,302,303]:
        means={g:float(np.mean([r['best_success'] for r in results if r['groups']==g])) for g in a.groups};winner=max(a.groups,key=lambda g:(means[g],-g))
        close_milestone('h1_data_and_proposals','Feasible proposal supervision expanded independently of authorization; select scale by reserved flight validation.',
            means=means,selected_groups=winner,results=results,proposal_and_gate_targets_are_distinct=True)
if __name__=='__main__':main()
