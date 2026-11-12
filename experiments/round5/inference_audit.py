"""Audit every trained actor's dtype/device/batch and representative physics replay."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from .inventory import groups,core,sha
from .models import load_agent,tree_map
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def sequences():
    chunks=[]
    for index in range(48,52):
        path=OUT/'data'/'initial'/f'route_{index:03d}.npz';z=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
        for r in meta['episodes']:
            if r['level'] not in [0.,.25,.295]:continue
            first=max(r['start'],min(r['stop']-64,r['start']+int(.5*(r['stop']-r['start']))));x=z['s'][first:min(first+64,r['stop'])]
            if len(x)<64:x=np.concatenate([x,np.repeat(x[-1:],64-len(x),axis=0)])
            chunks.append(x)
    return np.stack(chunks[:16]).astype(np.float64)

@torch.no_grad()
def stream(m,x):
    state=None;outputs=[];proposals=[]
    for t in range(x.shape[1]):
        action,proposal,state=m.propose_actions(x[:,t],state);outputs.append(action.cpu());proposals.append(proposal.cpu())
    return torch.stack(outputs,1),torch.stack(proposals,1)

def main():
    torch.set_num_threads(1);start=time.time();x=sequences();records=[];physics=[]
    for group in groups():
        for entry in group['entries']:
            target=OUT/'inference_audit'/f"{group['id']}_{entry['seed']}.json"
            if target.exists():records.append(json.loads(target.read_text()));continue
            before=sha(entry['path']);cpu=load_agent(entry['path']);xx=torch.tensor(x,dtype=torch.float64)
            a,p=stream(cpu,xx[:8]);b,q=stream(cpu,xx);single,single_proposal=stream(cpu,xx[:1])
            assert torch.equal(a[:1].float(),single.float()) and float((p[:1]-single_proposal).abs().max())<1e-9
            batch_error=float((p-q[:8]).abs().max());assert torch.equal(a.float(),b[:8].float()) and batch_error<1e-9
            gpu=copy.deepcopy(cpu).cuda();c,r=stream(gpu,xx[:8].cuda());device_error=float((p-r).abs().max())
            assert torch.equal(a.float(),c.float()) and device_error<1e-9
            assert sha(entry['path'])==before
            row=dict(group=group['id'],seed=entry['seed'],batch_max_proposal_error=batch_error,device_max_proposal_error=device_error,canonical_actions_equal=True,checkpoint_sha256=before)
            save(target,row);records.append(row);del cpu,gpu;torch.cuda.empty_cache()
        # One representative of every approach/family,not onlyGTrXL.
        entry=group['entries'][0];target=OUT/'inference_audit'/f"{group['id']}_physics.json"
        if target.exists():physics.append(json.loads(target.read_text()));continue
        m=load_agent(entry['path']);cfg=dict(core(),intervention=group['intervention'])
        args=dict(condition='estimated',seeds=range(528024,528032),levels=[.25,.295],config=cfg)
        cpu8=evaluate(m,workers=8,**args);cpu16=evaluate(m,workers=16,**args);gpu8=evaluate(m.cuda(),workers=8,**args)
        comparisons=[]
        for other in [cpu16,gpu8]:
            assert len(other)==len(cpu8)
            comparisons.append(dict(outcomes_equal=all(a['outcome']==b['outcome'] for a,b in zip(cpu8,other)),steps_equal=all(a['episode_steps']==b['episode_steps'] for a,b in zip(cpu8,other)),
                maximum_rmse_difference=max(abs(a['tracking_rmse']-b['tracking_rmse']) for a,b in zip(cpu8,other))))
        single=evaluate(m.cpu(),condition='estimated',seeds=[528031],levels=[.25,.295],config=cfg,workers=1)
        selected=[r for r in cpu8 if r['seed']==528031]
        assert all(a['outcome']==b['outcome'] and a['episode_steps']==b['episode_steps'] and abs(a['tracking_rmse']-b['tracking_rmse'])<1e-8 for a,b in zip(selected,single))
        assert all(c['outcomes_equal'] and c['steps_equal'] and c['maximum_rmse_difference']<1e-8 for c in comparisons),group['id']
        row=dict(group=group['id'],cases=len(cpu8),comparisons=comparisons);save(target,row);physics.append(row);del m;torch.cuda.empty_cache()
        print('INFERENCE_AUDIT',group['id'],'actors',len(records),'elapsed',round(time.time()-start),flush=True)
    close_pool();save(OUT/'inference_audit_complete.json',dict(seconds=time.time()-start,actors=records,physical_groups=physics,passed=True))
if __name__=='__main__':main()
