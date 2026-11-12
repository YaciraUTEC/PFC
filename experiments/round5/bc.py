"""Shared supervised initialisation and student-state aggregation pilots."""
import argparse,json,time
import numpy as np
import torch
from .models import Agent,ARCHS
from .losses import clone_loss
from .data import load,SequenceData,collect
from .evaluate import OUT,close_pool
from ..run import save
SEEDS=[101,102,103,104,105]

def tensor(x):return torch.as_tensor(x,device='cuda',dtype=torch.float32)
def setup(seed):
    torch.set_num_threads(1);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)

def starting_config(arch):
    reference=__import__('pathlib').Path(__file__).with_name('configs')/'reference_inputs.json'
    return dict(json.loads(reference.read_text())['starting_architectures'][arch])

def pretrain(m,dataset,seed,dest,updates=1200,loss_function=None):
    loss_function=clone_loss if loss_function is None else loss_function
    opt=m.optimizer(3e-4);rng=np.random.default_rng(seed);start=time.time();log=[];m.train()
    for it in range(updates):
        x,a,mask,valid=dataset.sample(rng,batch=8,length=96,burnin=32)
        mu,_,_=m(tensor(x));loss=loss_function(mu,tensor(a),tensor(mask),tensor(valid))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite BC loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step()
        if it%200==0:
            row=dict(update=it,loss=float(loss),seconds=time.time()-start);log.append(row);save(dest/'bc_progress.json',row)
            print('BC',m.cfg['arch'],seed,dataset.mode,it,float(loss),'seconds',round(time.time()-start),flush=True)
    m.eval();m.save(dest/'bc.pt',mode=dataset.mode,updates=updates)
    result=dict(started_epoch=start,finished_epoch=time.time(),seconds=time.time()-start,updates=updates,mode=dataset.mode,positive_gate_labels=len(dataset.positive),verified_gate_labels=len(dataset.verified),losses=log)
    save(dest/'bc_timing.json',result);return result

def pilots():
    data=load(include_dagger=False)
    for mode in ['residual','effectiveness']:
        dataset=SequenceData(data,mode)
        for arch in ARCHS:
            dest=OUT/'pilots'/mode/arch;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'bc.pt').exists():continue
            from .models import clone_loss as historical_pilot_loss
            setup(97);m=Agent(starting_config(arch)).cuda();pretrain(m,dataset,97,dest,updates=400,loss_function=historical_pilot_loss);del m;torch.cuda.empty_cache()
    close_pool()

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilots','dagger','observers'],default='pilots');p.add_argument('--worker',type=int,default=0);args=p.parse_args()
    if args.phase=='pilots':pilots()
    elif args.phase=='dagger':
        config=json.loads((OUT/'core_selection.json').read_text())['config'];paths=[dict(path=str(OUT/'pilots'/mode/a/'bc.pt'),mode=mode) for mode in ['residual','effectiveness'] for a in ARCHS]
        collect(config,workers=6,dagger_models=paths)
    else:
        data=load();dataset=SequenceData(data,'effectiveness');jobs=[(a,s) for a in ARCHS for s in SEEDS]
        for index,(arch,seed) in enumerate(jobs):
            if index%2!=args.worker:continue
            dest=OUT/'observers'/arch/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
            if (dest/'bc_timing.json').exists():continue
            setup(seed);m=Agent(starting_config(arch)).cuda();pretrain(m,dataset,seed,dest);del m;torch.cuda.empty_cache()
if __name__=='__main__':main()
