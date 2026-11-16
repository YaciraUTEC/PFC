"""Infrastructure pilot only: no data from this folder enter gate training."""
import time
import numpy as np
import torch
from .common import OUT,old_model,save,close_milestone
from .gate_data import collect_group
from .policy import canonical_risk,risk_features,controller_descriptor

def main():
    start=time.time();paths=[str(old_model(s)) for s in range(201,206)]+['unused_bc_controller_'+str(s) for s in range(301,306)]
    # Negative index chooses immutable FlashSAC201 and a separate729900 route.
    result=collect_group(-100,'pilot',paths);path=OUT/'gate_data/pilot/group_-100.npz'
    with np.load(path) as archive:z={k:archive[k] for k in archive.files}
    assert z['x'].shape[1]==196 and len(z['x'])==len(result['records'])
    np.testing.assert_array_equal(z['trace_index']-z['trace_start'],[r['tick'] for r in result['records']])
    encoder=canonical_risk().double().eval();encoded=np.empty((len(z['trace']),184),np.float64)
    with torch.no_grad():
        for a,b in zip(z['episode_boundaries'][:-1],z['episode_boundaries'][1:]):
            state=None
            for i in range(int(a),int(b)):
                f,state=risk_features(encoder,torch.as_tensor(z['trace'][i:i+1],dtype=torch.float64),state);encoded[i]=f[0].numpy()
    error=float(np.max(np.abs(encoded[z['trace_index']]-z['x'][:,:184]))) if len(z['x']) else 0.
    assert error<2e-6,error
    np.testing.assert_array_equal(z['x'][:,188:195],np.tile(controller_descriptor(paths[0]),(len(z['x']),1)))
    from .episode import fly,load_policy
    native_errors=[];settings=result['records'][0]
    for i,level in enumerate([.295,.296]):
        natural,rows=fly(result['seed'],level,settings['condition'],settings['noise'],settings['clock'],model=load_policy(paths[0]),collect=True)
        trace=np.asarray([r['s'] for r in rows]);a,b=z['episode_boundaries'][i:i+2];assert trace.shape==z['trace'][a:b].shape
        native_error=float(np.abs(trace-z['trace'][a:b]).max());assert native_error==0.,native_error;native_errors.append(native_error)
    close_milestone('h0_gate_pilot','Natural terminal labels, exact prefix replay, full student histories and known-controller features verified; pilot729900 is excluded from every fit/calibration/test split.',
        native_caller_history_error=max(native_errors),prefix_replay_checks=result['prefix_replay_checks'],maximum_prefix_error=result['maximum_prefix_error'],samples=len(z['y']),labels=result['labels'],feature_reconstruction_error=error,seconds=time.time()-start,pilot_excluded_from_training=True)
if __name__=='__main__':main()
