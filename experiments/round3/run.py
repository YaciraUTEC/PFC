"""Fixed first-round protocol. All selection uses validation; test is opened last."""
import argparse,json,time,datetime,hashlib,os
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from .common import OUT,SEEDS,LEVELS,save,load_data,make_vec,new_policy,pretrain_policy,evaluate
from .rewards import CSIL,RIME,MEIRL,tensor
from .evaluate import batched_evaluate as evaluate,close_pool

METHODS=['manual','csil','meirl','rime'];BUDGET=262144

class RewardVec(VecEnvWrapper):
    def __init__(self,v,reward=None,manual_stats=(0.,1.),capture=False):
        super().__init__(v);self.reward=reward;self.stats=manual_stats;self.capture=capture;self.buffer=[]
    def reset(self):return self.venv.reset()
    def step_wait(self):
        o,r,d,infos=self.venv.step_wait()
        s=np.stack([i['state'] for i in infos]);a=np.stack([i['action'] for i in infos]);ns=np.stack([i['next_state'] for i in infos])
        with torch.no_grad():
            if self.reward is None:style=np.clip((np.array([i['manual_style'] for i in infos])-self.stats[0])/self.stats[1],-5,5)
            else:style=self.reward(tensor(s),tensor(a),tensor(ns),tensor(d)).detach().cpu().numpy()
        r=np.array([i['task_reward'] for i in infos],np.float32)+.02*style
        if self.capture:self.buffer.append(dict(s=s,a=a,ns=ns,done=np.asarray(d)))
        return o,r.astype(np.float32),d,infos
    def pop(self):
        result={k:np.concatenate([x[k] for x in self.buffer]) for k in ['s','a','ns','done']};self.buffer=[];return result

class Discover(BaseCallback):
    def __init__(self,reward,data,out):super().__init__();self.r=reward;self.data=data;self.out=out;self.logs=[]
    def _on_step(self):return True
    def _on_rollout_end(self):
        learner=self.training_env.pop()
        loss=self.r.adversarial_update(self.data,learner,self.model.policy,self.model.seed+self.num_timesteps)
        cal=self.r.calibrate(self.data)
        self.logs.append(dict(steps=self.num_timesteps,discriminator_loss=loss,calibration=cal))
        save(self.out/'adversarial.json',self.logs)
        if len(self.logs)%8==0:print('ME-IRL discovery',self.num_timesteps,loss,flush=True)

class Progress(BaseCallback):
    def __init__(self,out,method,seed):
        super().__init__();self.out=out;self.method=method;self.seed=seed;self.last=0;self.validation=[];self.best=-np.inf;self.start=time.time();self.completed=[];self.validation_seconds=0.
    def _on_training_start(self):self.validate(0)
    def _on_step(self):
        for d,i in zip(self.locals['dones'],self.locals['infos']):
            if d:self.completed.append(bool(i['mission_success']))
        return True
    def _on_rollout_end(self):
        if self.num_timesteps-self.last>=16384:
            self.last=self.num_timesteps
            save(self.out/'progress.json',dict(steps=self.num_timesteps,total=BUDGET,seconds=time.time()-self.start,train_success_recent=float(np.mean(self.completed[-100:])) if self.completed else None))
            print(self.method,self.seed,self.num_timesteps,'/',BUDGET,'seconds',round(time.time()-self.start),flush=True)
    def _on_rollout_start(self):
        if self.num_timesteps>0 and self.num_timesteps%65536==0:self.validate(self.num_timesteps)
    def _on_training_end(self):self.validate(self.num_timesteps)
    def validate(self,steps):
        vstart=time.time()
        rows=evaluate(self.model,seeds=range(100010,100015))
        rate=float(np.mean([r['mission_success'] for r in rows]));effort=float(np.mean([r['effort'] for r in rows]))
        mild=[r['mission_success'] for r in rows if r['level']<=.05]
        # Primary validation completion; tiny effort tie-breaker. No forced gate.
        score=rate-.001*effort
        seconds=time.time()-vstart;self.validation_seconds+=seconds
        self.validation.append(dict(seconds=seconds,steps=steps,score=score,success=rate,mild=float(np.mean(mild)),effort=effort,episodes=rows))
        save(self.out/'validation.json',self.validation)
        if score>self.best:
            self.best=score;self.model.save(self.out/'best');save(self.out/'selection.json',dict(steps=steps,score=score,rule='validation completion minus 0.001 effort'))


def protocol():
    p=OUT/'protocol.json'
    if p.exists():return
    save(p,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_seeds=SEEDS,
        methods=METHODS,controller='updated physical DSLPID',policy='bounded tanh Gaussian LSTM64x1, history32 at48Hz',
        train_data_seeds=list(range(300000,300024)),reward_validation_data_seeds=list(range(300024,300030)),
        controller_validation_seeds=list(range(100010,100015)),final_test_seeds=list(range(200000,200050)),
        levels=LEVELS,condition='RPM-loss estimate underestimates fault by factor U(0.45,0.9), common detection delay2..6 control ticks',
        common_initialization='same seed-specific behavioral clone for all four PPO treatments',
        ppo=dict(steps_per_seed=BUDGET,n_envs=8,n_steps=256,batch_size=256,n_epochs=5,lr_start=2e-4,lr_end=2e-5,gamma=.995,gae_lambda=.95,clip=.15,target_kl=.015,entropy=.001),
        expert='updated DSLPID with true effectiveness after shared delay; deployment has biased estimate only',
        common_task_anchors=dict(mission=10,crash=-10,landing_or_timeout=-5,effort_coefficient=.002),
        style_coefficient=.02,style_normalization='training-only mean/std; clip[-5,5], common to learned and manual',
        adaptations=dict(meirl='transition-aware causal-entropy discriminator,7 Gaussian dynamics/5 elites; PPO replaces SAC; real rollouts only,no synthetic rollouts; freeze state reward after65536 discovery steps',
            csil='Torch port stationary periodic PSD heteroscedastic tanh head128features;NLL+L2;fixed log-density-ratio shaped reward;common PPO replaces native Acme/JAX actor-critic',
            rime='Torch ensemble3 BradleyTerry,entropy reward warmstart,official variance-based KL filter and flips;fixed batch feedback,scripted metrics teacher with10% flips;common PPO replaces native SAC;no human labels'),
        decision='Primary mean success at10,15,20,25% over routes and training seeds. Paired hierarchical bootstrap across training seeds and route IDs,95%CI. Recommend learned reward only if lowerCI gain over manual PPO>0 and mild0,2,5% safety loss upperCI<5 percentage points. Compare BC-only and zero-residual updated controller. If no method clears both gates, do not declare an IRL winner.',
        limitations='route completion is original waypoint criterion,not stable touchdown;50%RPM is outside normal-yaw static hover feasibility and is diagnostic only;this is a local screening of explicit adaptations,not a replication of paper benchmarks',
        sources=dict(meirl='https://proceedings.mlr.press/v331/zhan26a.html',csil='https://joemwatson.github.io/csil/',rime='https://proceedings.mlr.press/v235/cheng24k.html')))

def run(seeds=SEEDS,phase="all"):
    worker="_".join(map(str,seeds))
    status=OUT/'status.json'
    if phase=='test' and status.exists() and json.loads(status.read_text()).get('stage')=='complete':
        required=[OUT/'baselines_test.json',OUT/'baseline_exact_secondary.json']
        for sid in SEEDS:
            sd=OUT/f'seed_{sid}';required.extend([sd/'bc_test.json',sd/'bc_exact_test.json'])
            for name in METHODS:required.extend([sd/name/'test.json',sd/name/'exact_test.json',sd/name/'final_test.json'])
        if all(p.exists() for p in required):
            frozen=json.loads((OUT/'frozen_before_test.json').read_text())
            for item in frozen['checkpoint_entries']:
                for relative,expected in item['weights_sha256'].items():assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==expected,relative
            print('Final evaluation already complete;frozen artifacts verified.',flush=True);return
    protocol();data=load_data();manual=data['style'][data['train']];stats=(float(manual.mean()),max(float(manual.std()),.1))
    start=time.time();save(OUT/f'execution_{worker}.json',dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=phase,evaluation_workers=int(os.environ.get('NICOL_EVAL_WORKERS','16'))))
    for seed in ([] if phase=="test" else seeds):
        seed_dir=OUT/f'seed_{seed}';seed_dir.mkdir(parents=True,exist_ok=True)
        v=make_vec(seed=seed);m=new_policy(v,seed)
        if not (seed_dir/'bc.zip').exists():
            timing=pretrain_policy(m,data,seed);m.save(seed_dir/'bc');save(seed_dir/'bc_timing.json',timing)
        else:m.set_parameters(seed_dir/'bc.zip')
        v.close();del m
        for method in METHODS:
            d=seed_dir/method;d.mkdir(exist_ok=True)
            if (d/'complete.json').exists():continue
            t0=time.time();torch.manual_seed(seed);np.random.seed(seed);save(OUT/f'status_{worker}.json',dict(stage='reward',method=method,seed=seed))
            reward=None;reward_timing={};reward_reused=(d/'reward.pt').exists()
            if method!='manual':
                reward={'csil':CSIL,'meirl':MEIRL,'rime':RIME}[method](data)
                if (d/'reward.pt').exists():
                    ck=torch.load(d/'reward.pt',map_location='cuda',weights_only=False);reward.load_state_dict(ck['state']);reward_timing=ck['timing']
                    if method=='meirl':reward.elites=ck['elites']
                else:
                    if method=='meirl':
                        dynamics=reward.fit_dynamics(data,seed)
                        discovery=RewardVec(make_vec(seed=400000+seed,flat=True),reward,capture=True)
                        actor=new_policy(discovery,seed,flat=True)
                        bc=pretrain_policy(actor,data,seed,updates=1200,flat=True)
                        actor.ent_coef=.02
                        callback=Discover(reward,data,d)
                        discover_start=time.time();actor.learn(65536,callback=callback);actor.save(d/'discovery_policy');discovery.close()
                        reward_timing=dict(dynamics=dynamics,behavioral_clone=bc,discovery_seconds=time.time()-discover_start,discovery_steps=65536,calibration=reward.calibrate(data))
                        del actor
                    else:reward_timing=reward.fit(data,seed)
                    torch.save(dict(state=reward.state_dict(),timing=reward_timing,elites=getattr(reward,'elites',None)),d/'reward.pt')
                reward.eval()
                for p in reward.parameters():p.requires_grad_(False)
            save(d/'reward_timing.json',reward_timing)
            v=RewardVec(make_vec(seed=500000+seed),reward,stats);m=new_policy(v,seed);m.set_parameters(seed_dir/'bc.zip')
            callback=Progress(d,method,seed);train_start=time.time()
            save(OUT/f'status_{worker}.json',dict(stage='policy_training',method=method,seed=seed,budget=BUDGET))
            m.learn(BUDGET,callback=callback);m.save(d/'final');v.close()
            save(d/'complete.json',dict(total_seconds=time.time()-t0,policy_seconds=time.time()-train_start,validation_seconds=callback.validation_seconds,policy_excluding_validation_seconds=time.time()-train_start-callback.validation_seconds,reward_reused=reward_reused,reward_timing=reward_timing,actual_steps=m.num_timesteps))
            del m,v,reward;torch.cuda.empty_cache()
    if phase=="train":
        save(OUT/f'status_{worker}.json',dict(stage="training_complete"));close_pool();return
    assert all((OUT/f"seed_{seed}"/method/"complete.json").exists() for seed in SEEDS for method in METHODS), "Training / validation selection must finish before final test"
    if not (OUT/'frozen_before_test.json').exists():
        from .freeze_checkpoints import main as freeze_checkpoints
        freeze_checkpoints()
    frozen=json.loads((OUT/'frozen_before_test.json').read_text())
    choices={(x['seed'],x['method']):x for x in frozen['checkpoint_entries']}
    for item in frozen['checkpoint_entries']:
        for relative,expected in item['weights_sha256'].items():
            assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==expected,'Frozen model changed: '+relative
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    # Test split is opened only after all reward fits and checkpoint choices finish.
    save(OUT/'status.json',dict(stage='final_test'))
    if not (OUT/'baselines_test.json').exists():
        rows=[]
        for controller in ['legacy','robust','updated']:
            for condition in ['exact','uncertain']:
                print('test baseline',controller,condition,flush=True)
                rr=evaluate(controller=controller,condition=condition)
                rows.extend([dict(r,controller=controller,condition=condition) for r in rr])
        rows.extend([dict(r,controller='oracle_teacher',condition='uncertain') for r in evaluate(mode='teacher')])
        save(OUT/'baselines_test.json',rows)
    if not (OUT/'baseline_exact_secondary.json').exists():
        save(OUT/'baseline_exact_secondary.json',evaluate(controller='updated',condition='exact',seeds=range(200100,200120)))
    for seed in SEEDS:
        sd=OUT/f'seed_{seed}'
        for method in ['bc']+METHODS:
            d=sd if method=='bc' else sd/method;p=d/('bc_test.json' if method=='bc' else 'test.json')
            ep=d/('bc_exact_test.json' if method=='bc' else 'exact_test.json')
            if method!='bc' and choices[(seed,method)]['selected_actor_equals_bc']:
                reused=[]
                for target,source in [(p,sd/'bc_test.json'),(ep,sd/'bc_exact_test.json')]:
                    if not target.exists():
                        rr=json.loads(source.read_text());save(target,rr);reused.append(str(source.relative_to(OUT)))
                        save(d/(target.stem+'_timing.json'),dict(seconds=0.,cases=len(rr),simulated_cases=0,reused_from=str(source.relative_to(OUT))))
                if reused:
                    save(d/'evaluation_reuse.json',dict(reason='Byte-identical actor parameters,architecture,precision and same seeded cases;critic is not used for deterministic action prediction.',actor_sha256=choices[(seed,method)]['selected_actor_sha256'],sources=reused))
                    print('cached identical actor',seed,method,flush=True)
            if not p.exists() or not ep.exists():
                print('test selected model',seed,method,flush=True)
                m=__import__('stable_baselines3').PPO.load(sd/'bc.zip' if method=='bc' else d/'best.zip',device='cuda')
                if not p.exists():
                    test_start=time.time();rr=evaluate(m);save(p,rr)
                    save(d/(p.stem+'_timing.json'),dict(seconds=time.time()-test_start,cases=len(rr),simulated_cases=len(rr)))
                if not ep.exists():
                    test_start=time.time();rr=evaluate(m,condition='exact',seeds=range(200100,200120));save(ep,rr)
                    save(d/(ep.stem+'_timing.json'),dict(seconds=time.time()-test_start,cases=len(rr),simulated_cases=len(rr)))
                del m
            if method!='bc' and not (d/'final_test.json').exists():
                print('test final-budget model',seed,method,flush=True)
                m=__import__('stable_baselines3').PPO.load(d/'final.zip',device='cuda')
                test_start=time.time();rr=evaluate(m,seeds=range(200000,200030));save(d/'final_test.json',rr)
                save(d/'final_test_timing.json',dict(seconds=time.time()-test_start,cases=len(rr),simulated_cases=len(rr)));del m
    close_pool()
    save(OUT/'status.json',dict(stage='complete',seconds_this_process=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='+',default=SEEDS);parser.add_argument('--phase',choices=['all','train','test'],default='all');args=parser.parse_args();run(args.seeds,args.phase)