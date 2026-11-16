"""Development-only CPU/CUDA float64 equivalence and cost; no final routes."""
import time
import numpy as np
import torch
from .common import OUT,LEGACY_OUT,read,save,old_model,close_milestone
from .policy import TransformerPolicy
from .evaluation import cases,evaluate,key
from .composed_replay import certify


def main():
    if not torch.cuda.is_available():raise RuntimeError('No CUDA device')
    torch.set_num_threads(1);start=time.time();theta=read(OUT/'selected_proposal.json')['paths_by_seed']['301']
    specs=[('flash_native',dict(proposer=str(old_model(201)),mode='native')),('bc_forced',dict(proposer=theta,mode='forced')),
        ('flash_counterfactual',dict(read(LEGACY_OUT/'h2_counterfactual_gate_conclusion.json')['selected_by_source']['flashsac']['spec'],continuation_gain=0.))]
    cc=cases([719982,719983],levels=[.295,.296],conditions=['estimated'],noises=[1.,2.]);results=[]
    base=read(LEGACY_OUT/'gate_flights/baseline.json');histories=[]
    for row in base['episodes'][:8]:
        with np.load(row['trace_path']) as z:histories.append(z['observation'].copy())
    for name,spec in specs:
        cpu=evaluate(TransformerPolicy.from_spec(spec),cc,workers=8,inference_device='cpu');gpu=evaluate(TransformerPolicy.from_spec(spec),cc,workers=8,inference_device='cuda')
        a={key(r):r for r in cpu['episodes']};largest=0.;equivalent=True;changed=[]
        for row in gpu['episodes']:
            old=a[key(row)];error=float(np.abs(np.asarray(old['terminal_position'])-row['terminal_position']).max());largest=max(largest,error)
            same=old['outcome']==row['outcome'] and old['episode_steps']==row['episode_steps'] and error<1e-8
            equivalent &=same
            if not same:changed.append(dict(case=list(key(row)),old=old['outcome'],new=row['outcome'],position_error=error,old_steps=old['episode_steps'],new_steps=row['episode_steps']))
        before=time.time();flags_cpu,_=certify(TransformerPolicy.from_spec(spec),histories);cpu_proof=time.time()-before
        model=TransformerPolicy.from_spec(spec).cuda().double();before=time.time();flags_gpu,_=certify(model,histories);torch.cuda.synchronize();gpu_proof=time.time()-before
        flags_equal=bool(np.array_equal(flags_cpu,flags_gpu));del model;torch.cuda.empty_cache()
        result=dict(name=name,cases=len(cc),physical_equivalent=bool(equivalent),maximum_position_error=largest,changed=changed,cpu_seconds=cpu['seconds'],cuda_seconds=gpu['seconds'],certification_flags_equal=flags_equal,cpu_certification_seconds=cpu_proof,cuda_certification_seconds=gpu_proof)
        results.append(result);save(OUT/'backend_pilot_progress.json',results);print('BACKEND_PILOT',result,flush=True)
    close_milestone('backend_pilot','CPU/CUDA float64 deployment-equivalence diagnostic completed before any final test; use CUDA execution only if exact outcome/steps and strict position tolerance pass.',results=results,all_physical_equivalent=all(r['physical_equivalent'] for r in results),all_certification_flags_equal=all(r['certification_flags_equal'] for r in results),seconds=time.time()-start)
if __name__=='__main__':main()
