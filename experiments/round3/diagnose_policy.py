"""Behavioral fidelity of frozen policies on reserved teacher trajectories.
Diagnostic only: no update, no checkpoint selection, no flight-test reuse.
"""
import time,numpy as np,torch
from stable_baselines3 import PPO
from .common import OUT,SEEDS,load_data,histories,save

def effective(a):return .5*(a[:,4:5]+1)*a[:,:4]
def main():
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();data=load_data();ids=[]
    for r in data['records']:
        if r['split']=='reward_validation' and r['mode']=='teacher':ids.extend(range(r['start'],r['stop']))
    ids=np.array(ids);rng=np.random.default_rng(728151);ids=np.sort(rng.choice(ids,min(3000,len(ids)),False))
    h=histories(data,ids);target=effective(data['a'][ids]);active=np.any(data['s'][ids,31:35]>.001,axis=1);rows=[]
    for seed in SEEDS:
        for method in ['bc','manual','csil','meirl','rime']:
            path=OUT/f'seed_{seed}'/('bc.zip' if method=='bc' else method+'/final.zip')
            m=PPO.load(path,device='cuda');pred=[]
            for j in range(0,len(h),256):pred.append(m.predict(h[j:j+256],deterministic=True)[0])
            a=np.concatenate(pred);delta=effective(a);err=(delta-target).square() if hasattr(delta,'square') else np.square(delta-target)
            rows.append(dict(seed=seed,method=method,frames=len(ids),post_detection_frames=int(active.sum()),
                residual_rmse_post_detection=float(np.sqrt(err[active].mean())),residual_rms_before_detection=float(np.sqrt(np.square(delta[~active]).mean())),
                residual_rms_post_detection=float(np.sqrt(np.square(delta[active]).mean())),mean_gate_post_detection=float(np.mean((a[active,4]+1)/2))))
            print(rows[-1],flush=True);del m;torch.cuda.empty_cache()
    save(OUT/'policy_validation_diagnostics.json',dict(seconds=time.time()-start,rows=rows,interpretation='Open-loop command fidelity on the same3000 reserved teacher frames. Residual is gate*direction normalized by fixed wrench limits. These states come from demonstrations,not the final flight test. Fidelity alone is not flight success.'))
if __name__=='__main__':main()