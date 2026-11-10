"""Reward diagnostics on the six reserved demonstration-validation geometries.
These diagnostics never select policy checkpoints and never read flight tests.
"""
import json,numpy as np,torch
from scipy.stats import spearmanr
from .common import OUT,SEEDS,load_data,save
from .rewards import CSIL,MEIRL,RIME,tensor

def main():
    d=load_data();rng=np.random.default_rng(81623);groups={}
    for r in d['records']:
        if r['split']=='reward_validation':groups.setdefault((r['seed'],r['level']),[]).append(r)
    pairs=[];physical=[]
    for records in groups.values():
        for _ in range(30):
            a,b=rng.choice(records,2,replace=False);length=min(a['stop']-a['start'],b['stop']-b['start'])
            if length<32:continue
            k=int(rng.integers(length-31));ix1=np.arange(a['start']+k,a['start']+k+32);ix2=np.arange(b['start']+k,b['start']+k+32)
            pairs.append(np.stack([ix1,ix2]));physical.append(float(d['style'][ix1].sum()-d['style'][ix2].sum()))
    pairs=np.array(pairs);physical=np.array(physical);strict=np.abs(physical)>.01;rows=[]
    for seed in SEEDS:
        for method,cls in [('csil',CSIL),('meirl',MEIRL),('rime',RIME)]:
            path=OUT/f'seed_{seed}'/method/'reward.pt'
            if not path.exists():continue
            m=cls(d);ck=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(ck['state']);m.eval()
            if method=='meirl':m.elites=ck['elites']
            predicted=[]
            with torch.no_grad():
                for ix in np.array_split(np.arange(len(pairs)),max(1,len(pairs)//64)):
                    pp=pairs[ix];v=m(tensor(d['s'][pp]).reshape(-1,48),tensor(d['a'][pp]).reshape(-1,5)).reshape(len(ix),2,32).sum(-1)
                    predicted.extend((v[:,0]-v[:,1]).cpu().numpy())
                predicted=np.array(predicted)
                row=dict(seed=seed,method=method,strict_pairs=int(strict.sum()),style_preference_accuracy=float(np.mean((predicted[strict]>0)==(physical[strict]>0))),spearman=float(spearmanr(predicted[strict],physical[strict]).statistic))
                val=np.flatnonzero(d['validation']);val=val[::max(1,len(val)//10000)]
                if method=='csil':
                    teacher=np.concatenate([np.arange(r['start'],r['stop']) for r in d['records'] if r['split']=='reward_validation' and r['mode']=='teacher']);teacher=teacher[np.any(d['s'][teacher,31:35]>.001,axis=1)];mu,std=m.policy(m.norm(tensor(d['s'][teacher])))
                    row['cloned_action_mse_post_detection']=float((mu.tanh()-tensor(d['a'][teacher])).square().mean())
                if method=='meirl':
                    val=val[~d['done'][val]];s=tensor(d['s'][val]);a=tensor(d['a'][val]);target=m.norm(tensor(d['ns'][val]))-m.norm(s)
                    mean=[];variance=[]
                    for j in m.elites:
                        mu,lv=m.dynamics[j](torch.cat([m.norm(s),a],-1)).chunk(2,-1);mean.append(mu);variance.append(lv.clamp(-6,1).exp())
                    means=torch.stack(mean);vars=torch.stack(variance);mu=means.mean(0);var=(vars+means.square()).mean(0)-mu.square()
                    row.update(dynamics_normalized_mse=float((mu-target).square().mean()),dynamics_95percent_marginal_coverage=float(((mu-target).abs()<=1.96*var.clamp_min(1e-8).sqrt()).float().mean()))
                rows.append(row);print(row,flush=True)
            del m;torch.cuda.empty_cache()
    save(OUT/'reward_validation_diagnostics.json',dict(validation_route_seeds=list(range(300024,300030)),pairs=len(pairs),diagnostics=rows,interpretation='Style-only order according to scripted physical metrics,excluding near ties. Not human feedback,not the primary flight-success endpoint. RIME training-log clean_accuracy included neutral ties as binary and is not used for this diagnostic.'))
if __name__=='__main__':main()