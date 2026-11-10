"""Reproducible local IRL -> six temporal PPO compensators, with held-out evaluation."""
from pathlib import Path
import argparse,json,time,os,sys,csv,hashlib,datetime,platform,subprocess
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from .env import FlightEnv,EXPERT_LEVELS,LEVELS
from .models import TemporalFeatures
from irl_features import FEATURE_NAMES,N_FEATURES,horizonte_efectivo
ROOT=Path(__file__).resolve().parents[1]
ARCHS=['lstm','mamba','mamba2','mamba3','gtrxl','ltc']
IRL_MASK=np.r_[np.ones(9),np.zeros(2)]
TEST_LEVELS=[0.,.02,.04,.06,.08,.10,.15,.20]
torch.set_num_threads(1)

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(tmp,path)

def make_vec(n,history=1,reward_path=None,levels=None,seed=0,theta=None,scale=None,warm_start=False):
    def init():
        torch.set_num_threads(1)
        env=FlightEnv(history=history,reward_path=reward_path,levels=levels,warm_start=warm_start)
        if theta is not None:env.configure_reward(theta,scale)
        return Monitor(env)
    vec=SubprocVecEnv([init for _ in range(n)],start_method='spawn') if 1<n<=16 else DummyVecEnv([init for _ in range(n)])
    vec.seed(seed)
    return vec

def feature_rollouts(vec,model,n_episodes,seed,keep=False):
    vec.seed(seed);obs=vec.reset();n=vec.num_envs
    acc=np.zeros((n,N_FEATURES));length=np.zeros(n,int);records=[];mus=[]
    t0=time.time();steps=0
    while len(records)<n_episodes:
        actions=np.zeros((n,4),np.float32) if model is None else model.predict(obs,deterministic=True)[0]
        obs,rewards,dones,infos=vec.step(actions);steps+=n
        for i,info in enumerate(infos):
            acc[i]+=(.99**length[i])*info['phi'];length[i]+=1
            if dones[i]:
                if len(records)<n_episodes:
                    mu=acc[i]/horizonte_efectivo(960,.99)
                    mus.append(mu.copy())
                    records.append({k:info.get(k) for k in ['outcome','mission_success','planned_fault_pct',
                           'exposed','pasos_post_falla','distance_to_goal','base_controller']})
                    if len(records)%100==0:print('features',len(records),'/',n_episodes,flush=True)
                acc[i]=0.;length[i]=0
    return np.asarray(mus),records,{'seconds':time.time()-t0,'sim_steps':steps}

def evaluate(vec,model,episodes,seed,levels=TEST_LEVELS):
    rows=[]
    for k,level in enumerate(levels):
        vec.env_method('set_fixed_fault',level);vec.seed(seed);obs=vec.reset();collected=0;n=vec.num_envs
        while collected<episodes:
            actions=np.zeros((n,4),np.float32) if model is None or level==0 else model.predict(obs,deterministic=True)[0]
            obs,rew,done,infos=vec.step(actions)
            for d,info in zip(done,infos):
                if d and collected<episodes:
                    row={key:info.get(key) for key in ['outcome','mission_success','landed_near','exposed',
                        'pasos_post_falla','distance_to_goal','max_tilt','min_altitude_post_fault']}
                    row.update(level=level,episode=collected)
                    rows.append(row);collected+=1
    summary=[]
    for level in levels:
        rr=[r for r in rows if r['level']==level];ex=[r for r in rr if r['exposed']]
        summary.append(dict(level=level,n=len(rr),mission_success=sum(r['mission_success'] for r in rr)/len(rr),
           recovery=sum((r['mission_success'] or r['landed_near']) and (level==0 or r['exposed']) for r in rr)/len(rr),
           exposed_n=len(ex),unexposed_n=len(rr)-len(ex) if level>0 else 0,
           mission_given_exposure=sum(r['mission_success'] for r in ex)/len(ex) if ex else None,
           crash=sum(r['outcome']=='cayo' for r in rr)/len(rr)))
    vec.env_method('set_fixed_fault',None)
    return {'summary':summary,'episodes':rows}

def schedule(initial):
    return lambda progress: initial*(.1+.9*progress)

def new_ppo(vec,arch,lr,seed,width=96,n_steps=256,epochs=5):
    if arch=='mlp':
        return PPO('MlpPolicy',vec,learning_rate=schedule(lr),n_steps=n_steps,batch_size=128,
          n_epochs=epochs,gamma=.99,gae_lambda=.95,ent_coef=.005,clip_range=.15,
          target_kl=.015,max_grad_norm=.5,seed=seed,device='cpu',
          policy_kwargs={'log_std_init':-2.,'net_arch':dict(pi=[64,64],vf=[64,64])},verbose=0)
    model=PPO('MlpPolicy',vec,learning_rate=schedule(lr),n_steps=n_steps,batch_size=128,
       n_epochs=epochs,gamma=.99,gae_lambda=.95,ent_coef=.005,clip_range=.15,
       target_kl=.015,max_grad_norm=.5,seed=seed,device='cuda',
       policy_kwargs=dict(features_extractor_class=TemporalFeatures,
          features_extractor_kwargs=dict(architecture=arch,width=width,layers=2),
          net_arch=dict(pi=[64],vf=[64]),ortho_init=False,log_std_init=-2.),verbose=0)
    torch.nn.init.zeros_(model.policy.action_net.weight);torch.nn.init.zeros_(model.policy.action_net.bias)
    return model

def irl(out,args):
    reward=out/'reward.json'
    if reward.exists():
        print('Reusing frozen reward',reward,flush=True);return reward
    save(out/'status.json',{'stage':'expert_demonstrations','controller':'DSLPIDControl'})
    env=make_vec(args.envs,levels=EXPERT_LEVELS,seed=1000)
    mus,records,timing=feature_rollouts(env,None,args.expert_episodes,1000)
    env.close()
    np.savez_compressed(out/'expert.npz',mu=mus)
    save(out/'expert_episodes.json',{'episodes':records,'timing':timing,'seed':1000,
         'controller':'DSLPIDControl','levels':EXPERT_LEVELS,'all_outcomes_retained':True})
    # Only training demonstrations define the scale; never use held-out tests.
    scale=np.maximum.reduce([mus.std(0),np.full(N_FEATURES,.01),.08*np.abs(mus.mean(0))])
    expert=mus.mean(0)/scale*IRL_MASK;theta=IRL_MASK/np.linalg.norm(IRL_MASK)
    bar=None;best=None;logs=[]
    for it in range(args.irl_iterations):
        save(out/'status.json',{'stage':'irl','iteration':it+1,'of':args.irl_iterations})
        train=make_vec(args.envs,levels=EXPERT_LEVELS,seed=2000+it,theta=theta,scale=scale)
        model=new_ppo(train,'mlp',3e-4,2000+it,epochs=5)
        t0=time.time();model.learn(args.irl_steps);train.close()
        ev=make_vec(args.envs,levels=EXPERT_LEVELS,seed=3000,theta=theta,scale=scale)
        cm,cr,ct=feature_rollouts(ev,model,args.irl_eval_episodes,3000);ev.close()
        candidate=cm.mean(0)/scale*IRL_MASK
        if bar is None:bar=candidate.copy()
        else:
            direction=candidate-bar;den=direction@direction
            if den>1e-12:bar=bar+np.clip((expert-bar)@direction/den,0.,1.)*direction
        margin=float(np.linalg.norm(expert-bar))
        entry=dict(iteration=it+1,margin=margin,candidate_gap=float(np.linalg.norm(expert-candidate)),theta_used=theta.tolist(),
          mission_success=float(np.mean([r['mission_success'] for r in cr])),
          exposed=int(sum(r['exposed'] for r in cr)),seconds=time.time()-t0)
        logs.append(entry);print('IRL',json.dumps(entry),flush=True)
        if best is None or margin<=best['margin']:
            best=entry.copy();model.save(out/'irl_candidate_best')
        gap=np.maximum(expert-bar,0.)
        if np.linalg.norm(gap)>1e-8:theta=gap/np.linalg.norm(gap)
        save(out/'irl_history.json',logs)
    artifact=dict(schema_version=2,feature_names=FEATURE_NAMES,weights=best['theta_used'],
       feature_scale=scale.tolist(),weights_space='scaled_features',shaping_coef=.01,
       base_controller='DSLPIDControl',residual_delta_max=3000,residual_scaling='fixed_post_fault',
       sign_constraint='nonnegative weights on declared signed features, enforced during learning',
       fixed_task_prior=dict(mission=100,crash=-120,landing_near=40,landing_far=-60,timeout=-30,alive=.02),
       best_iteration=best['iteration'],margin=best['margin'],selection='latest trained iterate tied at minimal mixture margin',source_seed=1000,
       expert_episodes=args.expert_episodes,irl_steps_per_iteration=args.irl_steps,
       learned_features=FEATURE_NAMES[:9],excluded_policy_relative_features=FEATURE_NAMES[9:],
       residual_regularization_coef=.02,
       training_levels=EXPERT_LEVELS,phase='full_episode_features; residual only during real fault')
    save(reward,artifact);return reward

def restore_checkpoint(directory,env,config,device='cuda'):
    """Resume weights and optimizer; a legacy snapshot precedes its last PPO update."""
    directory=Path(directory)
    previous=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    for key in ['architecture','width','layers','history','steps','learning_rate_start',
                'n_steps','n_envs','batch_size','n_epochs','seed','reward_sha256']:
        if previous[key]!=config[key]:
            raise ValueError(f'Cannot resume with changed {key}: {previous[key]} -> {config[key]}')
    model=PPO.load(directory/'latest.zip',env=env,device=device)
    saved_steps=int(model.num_timesteps)
    state_path=directory/'checkpoint_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    updated=state.get('steps')==saved_steps and state.get('rollout_updated',False)
    repeated=0 if updated else config['n_steps']*config['n_envs']
    model.num_timesteps=max(0,saved_steps-repeated)
    info=dict(checkpoint_steps=saved_steps,resume_from_steps=model.num_timesteps,
        repeated_unupdated_rollout_steps=repeated,
        note='Optimizer retained; environment streams restart at the configured seed.')
    return model,info

class TrainingLog(BaseCallback):
    def __init__(self,path,total,vec_eval,eval_every=131072):
        super().__init__();self.path=Path(path);self.total=total;self.vec_eval=vec_eval
        self.eval_every=eval_every;self.last_eval=0;self.last_log=0;self.start=None
        self.last_checkpoint=0;self.elapsed_before=0.
        self.outcomes=[];self.evaluations=[];self.best=-1.
        validation=self.path/'validation.json'
        if validation.exists():
            self.evaluations=json.loads(validation.read_text(encoding='utf-8'))
            self.best=max((x['score'] for x in self.evaluations),default=-1.)
            self.last_eval=max((x['steps'] for x in self.evaluations),default=0)
        progress=self.path/'progress.json'
        if progress.exists():
            self.elapsed_before=json.loads(progress.read_text(encoding='utf-8'))['seconds']
    def _on_training_start(self):
        self.start=time.time();self.last_log=self.num_timesteps
        self.last_checkpoint=self.num_timesteps
    def _checkpoint(self):
        temp=self.path/'latest_next.zip';self.model.save(temp)
        os.replace(temp,self.path/'latest.zip')
        save(self.path/'checkpoint_state.json',dict(steps=self.num_timesteps,
             rollout_updated=True,seconds=self.elapsed_before+time.time()-self.start))
        self.last_checkpoint=self.num_timesteps
    def _on_rollout_start(self):
        # Called after the previous rollout has actually updated the optimizer.
        if self.num_timesteps-self.last_checkpoint>=16384:self._checkpoint()
    def _on_training_end(self):self._checkpoint()
    def _on_step(self):
        for d,info in zip(self.locals['dones'],self.locals['infos']):
            if d:self.outcomes.append({k:info.get(k) for k in ['outcome','mission_success','exposed','planned_fault_pct']})
        frac=min(self.num_timesteps/(self.total*.4),1.)
        self.training_env.env_method('set_max_fault',.02+frac*.18)
        if self.num_timesteps-self.last_log>=16384:
            rr=self.outcomes[-200:];seconds=self.elapsed_before+time.time()-self.start
            status=dict(steps=self.num_timesteps,total=self.total,seconds=seconds,
             fps=self.num_timesteps/seconds,
             recent_mission_success=float(np.mean([r['mission_success'] for r in rr])) if rr else None,
             recent_episodes=len(rr))
            save(self.path/'progress.json',status);print(self.path.name,status,flush=True)
            self.last_log=self.num_timesteps
        if self.num_timesteps-self.last_eval>=self.eval_every:
            ev=evaluate(self.vec_eval,self.model,5,50000)
            score=float(np.mean([x['mission_success'] for x in ev['summary']]))
            self.evaluations.append(dict(steps=self.num_timesteps,score=score,summary=ev['summary']))
            save(self.path/'validation.json',self.evaluations)
            if score>self.best:self.best=score;self.model.save(self.path/'best')
            self.last_eval=self.num_timesteps
        return True


def choose_width(arch,history,target=150000):
    from gymnasium.spaces import Box
    best=None
    for width in [32,64,96,128,160,192]:
        m=TemporalFeatures(Box(-10,10,shape=(history,24),dtype=np.float32),arch,width,2)
        count=sum(p.numel() for p in m.parameters())
        if best is None or abs(count-target)<best[0]:best=(abs(count-target),width,count)
    return best[1],best[2]

def benchmark(out,args):
    from gymnasium.spaces import Box
    results=[]
    for arch in ARCHS:
        width,count=choose_width(arch,args.history)
        model=TemporalFeatures(Box(-10,10,shape=(args.history,24),dtype=np.float32),arch,width,2).cuda()
        x=torch.randn(8,args.history,24,device='cuda')
        for _ in range(2):model(x).sum().backward();model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();t=time.time()
        for _ in range(5):model(x).sum().backward();model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        row=dict(architecture=arch,width=width,extractor_parameters=count,batch8_forward_backward_ms=1000*(time.time()-t)/5)
        results.append(row);print('BENCHMARK',row,flush=True)
        del model;torch.cuda.empty_cache()
    save(out/'architecture_benchmark.json',results);return results

def train_one(out,reward,args,arch,lr,steps,tag,validation_episodes=5):
    directory=out/tag;directory.mkdir(exist_ok=True)
    if (directory/'test.json').exists():
        result=json.loads((directory/'test.json').read_text(encoding='utf-8'))
        expected=dict(architecture=arch,steps=steps,seed=args.seed,history=args.history,
          n_envs=args.policy_envs,learning_rate_start=lr,
          reward_sha256=hashlib.sha256(reward.read_bytes()).hexdigest())
        changed=[key for key,value in expected.items() if result['config'].get(key)!=value]
        if changed or any(s['n']!=validation_episodes for s in result['summary']):
            raise ValueError('Completed run has a different request; use a separate output folder: '+str(changed))
        return result
    width,count=choose_width(arch,args.history)
    train=make_vec(args.policy_envs,args.history,str(reward),LEVELS,seed=args.seed,warm_start=True)
    train.env_method('set_max_fault',.02)
    ev=make_vec(max(1,args.envs//2),args.history,str(reward),LEVELS,seed=50000,warm_start=True)
    rollout_steps=max(32,2048//args.policy_envs)
    config=dict(architecture=arch,width=width,layers=2,history=args.history,steps=steps,
      learning_rate_start=lr,learning_rate_end=lr*.1,n_steps=rollout_steps,n_envs=args.policy_envs,batch_size=128,
      n_epochs=5,entropy_coef=.005,gamma=.99,gae_lambda=.95,clip_range=.15,target_kl=.015,
      max_grad_norm=.5,seed=args.seed,reward_sha256=hashlib.sha256(reward.read_bytes()).hexdigest(),
      controller='DSLPIDControl',extractor_parameters=count,task_prior=True,oracle_fault_information=True,
      prefix='exact DSLPIDControl physics/history warmup to fault boundary; prefix failures retained',
      training_fault_only=True,
      numerical_backend='parallel affine scan' if arch=='mamba' else
         'compiled ncps LTC cell (same ODE)' if arch=='ltc' else 'PyTorch reference')
    if (directory/'latest.zip').exists():
        model,resume_info=restore_checkpoint(directory,train,config)
        previous=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        config['resume_history']=previous.get('resume_history',[])+[resume_info]
        print('RESUME',arch,resume_info,flush=True)
    else:model=new_ppo(train,arch,lr,args.seed,width,n_steps=rollout_steps)
    start_steps=int(model.num_timesteps)
    train.env_method('set_max_fault',.02+min(start_steps/(steps*.4),1.)*.18)
    save(directory/'config.json',config)
    callback=TrainingLog(directory,steps,ev,eval_every=max(131072,steps//4))
    t=time.time()
    try:
        if start_steps<steps:
            model.learn(steps-start_steps,callback=callback,reset_num_timesteps=(start_steps==0))
        model.save(directory/'final')
        if (directory/'best.zip').exists():model=PPO.load(directory/'best.zip',device='cuda')
        result=evaluate(ev,model,validation_episodes,50000 if tag.startswith('pilot_') else 70000)
        result.update(config=config,training_seconds=callback.elapsed_before+time.time()-t,actual_steps=int(max(model.num_timesteps,callback.num_timesteps,start_steps)))
        save(directory/'test.json',result)
    finally:
        train.close();ev.close()
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['irl','benchmark','pilot','train','all','baseline'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'runs'/'dslpid_comparison_v2')
    parser.add_argument('--envs',type=int,default=8)
    parser.add_argument('--policy-envs',type=int,default=32)
    parser.add_argument('--history',type=int,default=32)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--expert-episodes',type=int,default=1600)
    parser.add_argument('--irl-iterations',type=int,default=8)
    parser.add_argument('--irl-steps',type=int,default=100000)
    parser.add_argument('--irl-eval-episodes',type=int,default=200)
    parser.add_argument('--pilot-steps',type=int,default=32768)
    parser.add_argument('--steps',type=int,default=1000000)
    parser.add_argument('--test-episodes',type=int,default=50)
    parser.add_argument('--architectures',nargs='+',default=ARCHS,choices=ARCHS)
    args=parser.parse_args();out=args.output.resolve()
    allowed=Path(r'E:\UTEC\Tesis\TesisNicol').resolve()
    if not out.is_relative_to(allowed):raise ValueError("All outputs must remain inside the task folder")
    out.mkdir(parents=True,exist_ok=True)
    save(out/f'request_config_{args.stage}.json',vars(args))
    if args.stage=='benchmark':benchmark(out,args);return
    if args.stage=='baseline':
        vec=make_vec(args.envs,1,levels=LEVELS)
        try:save(out/'baseline.json',evaluate(vec,None,args.test_episodes,70000))
        finally:vec.close()
        return
    reward=irl(out,args)
    if args.stage=='irl':save(out/'status.json',{'stage':'irl_complete','reward':str(reward)});return
    if not (out/'architecture_benchmark.json').exists():benchmark(out,args)
    if args.stage in ('pilot','all'):
        pilots=[]
        for lr in [1e-4,3e-4]:
            result=train_one(out,reward,args,'lstm',lr,args.pilot_steps,f'pilot_lstm_lr{lr}',5)
            score=np.mean([s['mission_success'] for s in result['summary']])
            pilots.append(dict(lr=lr,score=float(score)))
        chosen=max(pilots,key=lambda x:x['score'])['lr']
        save(out/'hyperparameter_selection.json',dict(candidates=pilots,chosen_lr=chosen,
          method='validation only, shared starting worker seeds, no test-set tuning',
          validation_sampling='worker RNG streams with automatic resets; final tests explicitly paired',
          caveat='small pilot chooses a starting configuration, not proof of global optimality'))
        if args.stage=='pilot':return
    hp=out/'hyperparameter_selection.json'
    lr=json.loads(hp.read_text())['chosen_lr'] if hp.exists() else 1e-4
    summaries=[]
    for arch in args.architectures:
        save(out/'status.json',{'stage':'training','architecture':arch,'steps':args.steps})
        result=train_one(out,reward,args,arch,lr,args.steps,arch,args.test_episodes)
        summaries.append(dict(architecture=arch,summary=result['summary'],
                             training_seconds=result['training_seconds'],actual_steps=result['actual_steps']))
        save(out/'comparison.json',summaries)
    save(out/'status.json',{'stage':'complete','architectures':args.architectures,'summary':summaries})
    print('COMPLETE',out,flush=True)

if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
