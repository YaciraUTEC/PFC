from pathlib import Path
from types import SimpleNamespace,ModuleType
import ast,importlib.util,sys,json,math
import numpy as np
import pytest
import torch
from einops import repeat,rearrange
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'fallas'))
from experiments.models import Mamba1,Mamba2,Mamba3,mamba3_scan,TemporalFeatures
from experiments.env import FlightEnv
from reward_contract import load_linear_weights
from irl_features import phi,cargar_escalas,FEATURE_NAMES

def extract(path,name,cls=None,namespace=None):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
    if cls:tree=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    ns={'np':np}
    if namespace:ns.update(namespace)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/path),'exec'),ns)
    return ns[name]

def test_pid_factory_preserves_fault_levels(monkeypatch):
    class Capture:
        def __init__(self,**kw):self.kw=kw
    for name in ['fault_env_residual_irl','fault_env_residual_irl_mamba']:
        mod=ModuleType(name);mod.FaultResidualEnvIRL=Capture;monkeypatch.setitem(sys.modules,name,mod)
    fn=extract('fallas/entrenar_rl.py','_env_irl')
    for arch in ['pid','mamba']:
        got=fn(arch,'irl',None,None,'aleatorias',None,.05,niveles_falla_pct=[.02,.05,.10,.15,.20])
        assert got.kw['niveles_falla_pct'][-1]==.20

def test_reward_coordinate_invariance(tmp_path):
    p=tmp_path/'reward.json'
    theta=np.array([.2,.8]);scale=np.array([.01,10.]);features=np.array([-4.,-2.])
    p.write_text(json.dumps(dict(weights=theta.tolist(),feature_scale=scale.tolist(),
       weights_space='scaled_features',feature_names=['a','b'])))
    w,_=load_linear_weights(p)
    np.testing.assert_allclose(w@features,theta@(features/scale))

def test_irl_projection_keeps_physical_signs():
    fn=extract('fallas/entrenar_irl_apprenticeship.py','restringir_w')
    w=fn(np.array([-4.,2.,-1.]),np.ones(3)/np.sqrt(3))
    assert np.all(w>=0) and np.isclose(np.linalg.norm(w),1.)

def test_unexposed_landing_is_not_recovery():
    fn=extract('fallas/evaluar_rl_hibrido.py','correr_episodio',namespace={'DIST_ATERRIZO_B':.3})
    class Env:
        gui=False;punto_B=np.array([0.,0,.1])
        def reset(self,seed=None):
            self.obs_raw=np.zeros((1,20));return np.zeros(24),{}
        def step(self,a):return np.zeros(24),0.,True,False,{'outcome':'aterrizo','t_falla_seg':None}
    assert fn(Env(),None,42)[0]=='sin_exposicion_aterrizo'

def test_real_env_seed_exposure_and_action_contract():
    e=FlightEnv(history=4,fixed_fault=.04)
    try:
        a,_=e.reset(seed=7);point=e.punto_A.copy()
        b,_=e.reset(seed=7)
        np.testing.assert_equal(a,b);np.testing.assert_equal(point,e.punto_A)
        for _ in range(960):
            obs,r,d,t,info=e.step(np.zeros(4))
            assert np.isfinite(obs).all() and np.isfinite(r)
            if d or t:break
        assert info['base_controller']=='DSLPIDControl'
        assert info['exposed'] and info['pasos_post_falla']>0
        assert info['planned_fault_pct']==pytest.approx(.04)
    finally:e.close()

@pytest.mark.parametrize('arch',['lstm','mamba','mamba2','mamba3','gtrxl','ltc'])
def test_model_forward_backward_and_sample_independence(arch):
    from gymnasium.spaces import Box
    torch.manual_seed(1)
    m=TemporalFeatures(Box(-10,10,shape=(16,24),dtype=np.float32),arch,32,1)
    x=torch.randn(2,16,24,requires_grad=True);y=m(x)
    assert y.shape==(2,64) and torch.isfinite(y).all()
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    m.eval()
    with torch.no_grad():
        before=m(x.detach()).clone();m(torch.randn(3,16,24));after=m(x.detach())
    torch.testing.assert_close(before,after)

def test_mamba2_matches_recurrent_definition():
    torch.manual_seed(2);m=Mamba2(32,state=16);u=torch.randn(2,12,32)
    z,xbc,dt=torch.split(m.in_proj(u),[m.inner,m.inner+2*m.n,m.h],-1)
    dt=F.softplus(dt+m.dt_bias)
    xbc=F.silu(m.conv(xbc.transpose(1,2))[...,:u.shape[1]].transpose(1,2))
    x,B,C=torch.split(xbc,[m.inner,m.n,m.n],-1);x=x.reshape(2,12,m.h,m.p)
    state=torch.zeros(2,m.h,m.p,m.n);ys=[]
    for t in range(12):
        state=torch.exp(-m.A_log.exp()*dt[:,t])[:,:,None,None]*state
        state=state+dt[:,t,:,None,None]*x[:,t,:,:,None]*B[:,t,None,None,:]
        ys.append((state*C[:,t,None,None,:]).sum(-1)+m.D[None,:,None]*x[:,t])
    expected=m.out_proj(m.norm(torch.stack(ys,1).flatten(-2)*F.silu(z)))
    torch.testing.assert_close(m(u),expected,atol=2e-5,rtol=2e-5)

def test_mamba3_matches_authors_reference_and_gradients():
    vendor=ROOT.parent/'vendor/mamba/tests/ops/triton/test_mamba3_siso.py'
    tree=ast.parse(vendor.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='mamba3_siso_step_ref')
    from typing import Optional,Tuple
    ns=dict(torch=torch,F=F,repeat=repeat,rearrange=rearrange,math=math,Optional=Optional,Tuple=Tuple)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(vendor),'exec'),ns)
    torch.manual_seed(3);b,l,h,n,p=2,11,4,16,8
    tensors=[torch.randn(b,l,1,n),torch.randn(b,l,1,n),torch.randn(b,l,h,p),
             -torch.rand(b,h,l),torch.rand(b,h,l)*.1,torch.randn(b,h,l),
             torch.randn(h,n),torch.randn(h,n),torch.randn(b,l,h,n//4),torch.randn(h),torch.randn(b,l,h,p)]
    inputs=[x.requires_grad_() for x in tensors]
    actual=mamba3_scan(*inputs);expected,_=ns['mamba3_siso_step_ref'](*inputs)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    ga=torch.autograd.grad(actual.square().mean(),inputs,retain_graph=True)
    ge=torch.autograd.grad(expected.square().mean(),inputs)
    for x,y in zip(ga,ge):torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-4)

def test_recovery_feature_active_above_ground_and_ignores_landing():
    stats=json.loads((ROOT/'results/stats_normalizacion.json').read_text())
    scales=cargar_escalas(stats)
    args=dict(pos=np.array([0.,0,.8]),rpy=np.zeros(3),ang_vel=np.zeros(3),vel=np.array([0.,0,.2]),
      wp_actual=np.array([0.,0,1.2]),punto_final=np.array([2.,0,.1]),rpm=np.full(4,14300.),
      rpm_anterior=np.full(4,14300.),vel_z_anterior=.2,dist_prev_z=2.,escalas=scales)
    vec,_=phi(**args);i=FEATURE_NAMES.index('recuperacion_altura')
    assert vec[i]>0
    vec,_=phi(**{**args,'wp_actual':np.array([0.,0,.1])})
    assert vec[i]==0

def test_actual_step_action_units_are_equal_after_fault():
    from types import SimpleNamespace
    ns=dict(CTRL_FREQ=48,MIN_RPM=9440,MAX_RPM=21700,DELTA_MAX=3000,MOTOR_FALLA=0,UMBRAL_WAYPOINT=.25,
       phi=lambda *a,**kw:(np.zeros(11),0.),_fase_de_waypoint=lambda *a:1,
       es_caida=lambda *a:False,es_aterrizaje=lambda *a:False)
    class Physical:
        def step(self,a):
            self.action=a.copy()
            return np.zeros((1,20)),0.,False,False,{}
    actions=[]
    for path,cls in [('fallas/nominal_flight_env.py','NominalFlightEnv'),
                     ('fallas/fault_env_residual_irl.py','FaultResidualEnvIRL')]:
        physical=Physical()
        state=SimpleNamespace(paso=0,wp_idx=3,waypoints=[np.array([2.,0,1.2])]*8,wp_medio_crucero=3,
            falla_activada=False,falla_activa=False,fault_aware=False,severidad=.96,obs_raw=np.zeros((1,20)),
            pid_controller=SimpleNamespace(computeControlFromState=lambda **kw:(np.full(4,14300.),0,0)),
            _env=physical,punto_B=np.array([2.,0,.1]),rpm_anterior=np.full(4,14300.),vel_z_anterior=0.,
            dist_prev_z=2.,escalas={},w=np.zeros(11),max_pasos=960,niveles=[.02,.05,.1,.15,.2],
            _delta_acum=[],modo_reward='irl',_build_obs=lambda *a:np.zeros(24))
        extract(path,'step',cls,ns)(state,np.full(4,.5))
        actions.append(physical.action)
    np.testing.assert_allclose(actions[0],actions[1])
    assert actions[0][0,1]==15800.

def test_policy_relative_features_cannot_dominate_irl():
    fn=extract('fallas/entrenar_irl_apprenticeship.py','restringir_w')
    gap=np.ones(11);gap[9]=1e6
    w=fn(gap,np.ones(11)/np.sqrt(11))
    assert np.all(w[9:]==0)
    assert np.isclose(np.linalg.norm(w),1.)

def test_warm_start_is_exact_full_flight_prefix():
    full=FlightEnv(history=16,fixed_fault=.04,warm_start=False)
    warm=FlightEnv(history=16,fixed_fault=.04,warm_start=True)
    try:
        obs,_=full.reset(seed=21)
        while full.wp_idx<full.wp_medio_crucero:
            obs,_,d,t,_=full.step(np.zeros(4))
            assert not d and not t
        fast,_=warm.reset(seed=21)
        assert full.paso==warm.paso and not warm.falla_activada
        np.testing.assert_allclose(full.obs_raw,warm.obs_raw,atol=1e-9)
        np.testing.assert_allclose(obs,fast,atol=1e-6)
        for _ in range(10):
            a=full.step(np.zeros(4));b=warm.step(np.zeros(4))
            np.testing.assert_allclose(a[0],b[0],atol=1e-6)
            assert a[1:4]==b[1:4]
    finally:
        full.close();warm.close()

def test_mamba1_matches_authors_selective_scan():
    vendor=ROOT.parent/'vendor/mamba/mamba_ssm/ops/selective_scan_interface.py'
    tree=ast.parse(vendor.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='selective_scan_ref')
    ns=dict(torch=torch,F=F,repeat=repeat,rearrange=rearrange)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(vendor),'exec'),ns)
    torch.manual_seed(4);m=Mamba1(32)
    u=torch.randn(2,13,32)
    x,z=m.in_proj(u).chunk(2,-1)
    x=F.silu(m.conv(x.transpose(1,2))[...,:13].transpose(1,2))
    dt,B,C=torch.split(m.x_proj(x),[m.rank,m.n,m.n],-1)
    dt=F.linear(dt,m.dt_proj.weight)
    y=ns['selective_scan_ref'](x.transpose(1,2),dt.transpose(1,2),-m.A_log.exp(),
       B.transpose(1,2),C.transpose(1,2),m.D,z=z.transpose(1,2),
       delta_bias=m.dt_proj.bias,delta_softplus=True)
    expected=m.out_proj(y.transpose(1,2))
    torch.testing.assert_close(m(u),expected,atol=2e-5,rtol=2e-5)


def test_affine_scan_outputs_and_gradients_match_recurrence():
    from experiments.scan import affine_scan
    torch.manual_seed(19)
    for length in [1,13,32]:
        a=(torch.rand(2,length,4,3)*.95).requires_grad_()
        b=torch.randn(2,length,4,3,requires_grad=True)
        state=torch.zeros_like(b[:,0]);expected=[]
        for t in range(length):
            state=a[:,t]*state+b[:,t];expected.append(state)
        ref=torch.stack(expected,1);actual=affine_scan(a,b)
        torch.testing.assert_close(actual,ref)
        ga=torch.autograd.grad(actual.square().sum(),(a,b),retain_graph=True,allow_unused=True,materialize_grads=True)
        gr=torch.autograd.grad(ref.square().sum(),(a,b))
        for x,y in zip(ga,gr):torch.testing.assert_close(x,y)


def test_completed_update_checkpoint_resumes_optimizer_and_budget(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from experiments.run import TrainingLog,restore_checkpoint,schedule
    class Task(gym.Wrapper):
        def set_max_fault(self,value):self.max_fault=value
    env=DummyVecEnv([lambda:Task(gym.make('CartPole-v1'))])
    config=dict(architecture='test',width=0,layers=0,history=1,steps=128,
       learning_rate_start=1e-4,n_steps=32,n_envs=1,batch_size=16,n_epochs=1,seed=19,reward_sha256='test')
    (tmp_path/'config.json').write_text(json.dumps(config))
    try:
        model=PPO('MlpPolicy',env,n_steps=32,batch_size=16,n_epochs=1,learning_rate=schedule(1e-4),seed=19,device='cpu')
        callback=TrainingLog(tmp_path,128,None,eval_every=1000000)
        model.learn(64,callback=callback)
        restored,info=restore_checkpoint(tmp_path,env,config,device='cpu')
        assert info['resume_from_steps']==64 and info['repeated_unupdated_rollout_steps']==0
        for name,param in model.policy.state_dict().items():
            torch.testing.assert_close(param,restored.policy.state_dict()[name],rtol=0,atol=0)
        original=model.policy.optimizer.state_dict()['state']
        loaded=restored.policy.optimizer.state_dict()['state']
        for i,state in original.items():
            for key,value in state.items():torch.testing.assert_close(value,loaded[i][key],rtol=0,atol=0)
        restored.learn(64,reset_num_timesteps=False)
        assert restored.num_timesteps==128
        assert restored._total_timesteps==128
        assert restored.policy.optimizer.param_groups[0]['lr']==pytest.approx(1e-5)
        (tmp_path/'checkpoint_state.json').unlink()
        _,legacy=restore_checkpoint(tmp_path,env,config,device='cpu')
        assert legacy['resume_from_steps']==32 and legacy['repeated_unupdated_rollout_steps']==32
        with pytest.raises(ValueError,match='reward_sha256'):
            restore_checkpoint(tmp_path,env,{**config,'reward_sha256':'changed'},device='cpu')
    finally:env.close()
def test_paired_scenarios_do_not_depend_on_episode_length_or_workers():
    from experiments.paired import evaluate_paired
    class FakeVec:
        def __init__(self,n):self.num_envs=n;self.seeds=[0]*n;self.t=[0]*n;self.level=0.
        def seed(self,seed):self.seeds=[seed+i for i in range(self.num_envs)]
        def reset(self):self.t=[0]*self.num_envs;return np.asarray(self.seeds,dtype=float)[:,None]
        def env_method(self,name,*args,**kwargs):
            if name=='set_fixed_fault':self.level=args[0];return [None]*self.num_envs
            i=kwargs['indices'];self.seeds[i]=kwargs['seed'];self.t[i]=0
            return [(np.array([self.seeds[i]],dtype=float),{})]
        def step(self,actions):
            done=[];infos=[]
            for i in range(self.num_envs):
                self.t[i]+=1;finished=self.t[i]>=1+self.seeds[i]%5
                done.append(finished)
                infos.append(dict(outcome='llego' if self.seeds[i]%3 else 'cayo',
                  mission_success=bool(self.seeds[i]%3),landed_near=False,exposed=self.level>0,
                  pasos_post_falla=self.t[i],distance_to_goal=self.seeds[i],max_tilt=0.,min_altitude_post_fault=1.))
                if finished:self.seeds[i]+=1000;self.t[i]=0
            return np.asarray(self.seeds,dtype=float)[:,None],np.zeros(self.num_envs),np.asarray(done),infos
    one=evaluate_paired(FakeVec(1),None,7,seed=70000,levels=[0.,.1])
    four=evaluate_paired(FakeVec(4),None,7,seed=70000,levels=[0.,.1])
    assert one['episodes']==four['episodes']
    assert one['summary']==four['summary']
    assert all(r['scenario_seed']==r['distance_to_goal'] for r in four['episodes'])
def test_completed_run_cannot_silently_reuse_a_different_seed_or_budget(tmp_path):
    import hashlib
    from experiments.run import train_one
    directory=tmp_path/'lstm';directory.mkdir()
    reward=tmp_path/'reward.json';reward.write_text('{}')
    config=dict(architecture='lstm',steps=524288,seed=42,history=32,n_envs=32,
       learning_rate_start=1e-4,reward_sha256=hashlib.sha256(reward.read_bytes()).hexdigest())
    result=dict(config=config,summary=[dict(n=50)])
    (directory/'test.json').write_text(json.dumps(result))
    args=SimpleNamespace(seed=42,history=32,policy_envs=32)
    assert train_one(tmp_path,reward,args,'lstm',1e-4,524288,'lstm',50)==result
    with pytest.raises(ValueError,match='different request'):
        train_one(tmp_path,reward,args,'lstm',1e-4,1000000,'lstm',50)
    args.seed=43
    with pytest.raises(ValueError,match='different request'):
        train_one(tmp_path,reward,args,'lstm',1e-4,524288,'lstm',50)