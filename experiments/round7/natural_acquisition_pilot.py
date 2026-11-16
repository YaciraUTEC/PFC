"""Full acquisition itself must preserve the natural composed-policy trajectory."""
import time
import numpy as np
import torch
from .common import OUT,LEGACY_OUT,read,close_milestone
from .gate_data import collect_group,case_settings
from .policy import TransformerPolicy
from .evaluation import evaluate,cases


def main():
    start=time.time();torch.set_num_threads(1);spec=read(LEGACY_OUT/'h3_filter_ablation_conclusion.json')['selected']['spec'];shadow=read(OUT/'selected_proposal.json')['paths_by_seed']['301'];index=-189;split='pilot_composed_method'
    r=collect_group(index,split,[shadow],behavior_spec=spec);cell=case_settings(index,1);cc=cases([r['seed']],levels=[.295,.296],conditions=[cell['condition']],noises=[cell['noise']],clocks=[cell['clock']])
    fresh=evaluate(TransformerPolicy.from_spec(spec),cc,workers=1,capture_directory=OUT/'natural_acquisition_pilot_fresh');errors=[]
    with np.load(OUT/'gate_data'/split/f'group_{index:03d}.npz') as z:
        for i,level in enumerate([.295,.296]):
            row=next(x for x in fresh['episodes'] if x['level']==level)
            with np.load(row['trace_path']) as a:actual=a['observation'].copy()
            begin,end=z['episode_boundaries'][i:i+2];stored=z['trace'][begin:end];assert actual.shape==stored.shape
            error=float(np.abs(actual-stored).max());assert error==0.,error;errors.append(error)
    close_milestone('h0_composed_method','The complete label acquisition leaves the original composed student trajectory exactly identical to fresh natural execution, including four-option counterfactual queries.',maximum_history_error=max(errors),prefix_replay_checks=r['prefix_replay_checks'],maximum_prefix_error=r['maximum_prefix_error'],pilot_excluded=True,seconds=time.time()-start)
if __name__=='__main__':main()
