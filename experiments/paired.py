"""Final paired scenarios, independent of episode durations and vector worker count."""
from pathlib import Path
import argparse,json,time
import numpy as np
from stable_baselines3 import PPO
from .run import ROOT,ARCHS,TEST_LEVELS,make_vec,save

PROTOCOL='paired_seed_per_episode_v1'

def evaluate_paired(vec,model,episodes,seed=70000,levels=TEST_LEVELS):
    if episodes<1:raise ValueError('At least one scenario is required')
    rows=[];t0=time.time()
    for level in levels:
        vec.env_method('set_fixed_fault',level);vec.seed(seed);obs=vec.reset()
        n=vec.num_envs;assigned=[i if i<episodes else None for i in range(n)]
        next_episode=min(n,episodes);collected=0
        while collected<episodes:
            actions=np.zeros((n,4),np.float32) if model is None or level==0 else model.predict(obs,deterministic=True)[0]
            obs,_,done,infos=vec.step(actions)
            for i,(d,info) in enumerate(zip(done,infos)):
                episode=assigned[i]
                if not d or episode is None:continue
                row={k:info.get(k) for k in ['outcome','mission_success','landed_near','exposed',
                     'pasos_post_falla','distance_to_goal','max_tilt','min_altitude_post_fault']}
                row.update(level=level,episode=episode,scenario_seed=seed+episode)
                rows.append(row);collected+=1
                if next_episode<episodes:
                    # VecEnv already auto-reset. Override that incidental episode
                    # with the explicitly assigned scenario and replace its obs.
                    replacement=vec.env_method('reset',seed=seed+next_episode,indices=i)[0]
                    obs[i]=replacement[0]
                    assigned[i]=next_episode;next_episode+=1
                else:assigned[i]=None
    vec.env_method('set_fixed_fault',None)
    rows.sort(key=lambda r:(r['level'],r['episode']))
    summary=[]
    for level in levels:
        rr=[r for r in rows if r['level']==level];ex=[r for r in rr if r['exposed']]
        assert [r['scenario_seed'] for r in rr]==list(range(seed,seed+episodes))
        summary.append(dict(level=level,n=len(rr),mission_success=sum(r['mission_success'] for r in rr)/len(rr),
          recovery=sum((r['mission_success'] or r['landed_near']) and (level==0 or r['exposed']) for r in rr)/len(rr),
          exposed_n=len(ex),unexposed_n=len(rr)-len(ex) if level>0 else 0,
          mission_given_exposure=sum(r['mission_success'] for r in ex)/len(ex) if ex else None,
          crash=sum(r['outcome']=='cayo' for r in rr)/len(rr)))
    return dict(protocol=PROTOCOL,seed_start=seed,episodes_per_level=episodes,
                seconds=time.time()-t0,summary=summary,episodes=rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'runs/dslpid_comparison_v2')
    p.add_argument('--architectures',nargs='*',default=ARCHS,choices=ARCHS)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--envs',type=int,default=8)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT.parent):raise ValueError('Outputs must stay in task folder')
    if not (out/'baseline_paired.json').exists():
        vec=make_vec(args.envs,1,levels=TEST_LEVELS,seed=70000,warm_start=True)
        try:
            baseline=evaluate_paired(vec,None,args.episodes)
            baseline['controller']='DSLPIDControl'
            save(out/'baseline_paired.json',baseline)
        finally:vec.close()
        print('PAIRED baseline complete',flush=True)
    summaries=[]
    for arch in args.architectures:
        directory=out/arch;dest=directory/'paired_test.json'
        if dest.exists():result=json.loads(dest.read_text(encoding='utf-8'))
        else:
            checkpoint=directory/'best.zip'
            model=PPO.load(checkpoint,device='cuda')
            vec=make_vec(args.envs,32,str(out/'reward.json'),TEST_LEVELS,seed=70000,warm_start=True)
            try:
                result=evaluate_paired(vec,model,args.episodes)
                result.update(architecture=arch,checkpoint=str(checkpoint),selection='best validation score, never selected on these tests')
                save(dest,result)
            finally:vec.close()
        summaries.append(dict(architecture=arch,summary=result['summary']))
        save(out/'paired_comparison.json',summaries)
        print('PAIRED',arch,json.dumps(result['summary']),flush=True)
if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support();main()