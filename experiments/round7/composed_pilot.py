"""Verify aggregation follows the selected composed student's actual trajectory."""
import time
import numpy as np
import torch
from .common import OUT,read,close_milestone
from .gate_data import collect_group,case_settings
from .policy import TransformerPolicy
from .evaluation import evaluate,cases


def main():
    if not (OUT/'h3_filter_ablation_conclusion.json').exists():raise RuntimeError('Select composed student first')
    if not (OUT/'h3_continuation_ablation_conclusion.json').exists():raise RuntimeError('Close the continuation-control comparison first')
    torch.set_num_threads(1);start=time.time();spec=read(OUT/'h3_filter_ablation_conclusion.json')['selected']['spec']
    shadow=read(OUT/'selected_proposal.json')['paths_by_seed']['301'];index=-189;split='pilot_composed'
    acquired=collect_group(index,split,[shadow],behavior_spec=spec);cell=case_settings(index,1)
    cc=cases([acquired['seed']],levels=[.295,.296],conditions=[cell['condition']],noises=[cell['noise']],clocks=[cell['clock']])
    fresh=evaluate(TransformerPolicy.from_spec(spec),cc,workers=1,capture_directory=OUT/'composed_acquisition_pilot_fresh')
    path=OUT/'gate_data'/split/f'group_{index:03d}.npz';errors=[]
    with np.load(path) as z:
        for i,level in enumerate([.295,.296]):
            row=next(r for r in fresh['episodes'] if r['level']==level)
            with np.load(row['trace_path']) as a:actual=a['observation'].copy()
            begin,end=z['episode_boundaries'][i:i+2];stored=z['trace'][begin:end]
            assert actual.shape==stored.shape,(actual.shape,stored.shape)
            error=float(np.max(np.abs(actual-stored)));assert error==0,error;errors.append(error)
    close_milestone('h0_composed_acquisition_pilot','Shadow proposal labels and independent natural-prefix branches preserve the independently reproduced trajectory of the selected composed student. Pilot729811 is excluded from training/calibration/test.',
        student_spec=spec,shadow_proposal=shadow,maximum_history_error=max(errors),terminal_outcomes=[r['outcome'] for r in fresh['episodes']],seconds=time.time()-start,pilot_excluded=True)
if __name__=='__main__':main()
