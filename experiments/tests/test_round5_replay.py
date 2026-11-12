"""Full-simulator differential verification of exact zero-action replay."""
import json
import numpy as np
import pytest
import torch
from experiments.round5 import replay_cache as cache
from experiments.round5.evaluate import evaluate,close_pool
from experiments.round5.models import Agent,configurations,ARCHS
from experiments.round5.inventory import always_off
CFG=dict(estimator='adaptive',readiness=True,emergency='damped',reserve=.003)
SEEDS=[525000,525001];LEVELS=[0.,.295]
PAIRS=[(s,l) for l in LEVELS for s in SEEDS]

@pytest.fixture(scope='module')
def base():
    torch.set_num_threads(1)
    try:return evaluate(condition='estimated',case_pairs=PAIRS,config=CFG,capture=True,workers=8,fixed_workers=True)
    finally:close_pool()

def install(monkeypatch,tmp_path,base,config=CFG):
    monkeypatch.setattr(cache,'BANK',tmp_path/'bank');cache._MEMORY.clear()
    cache.remember('estimated',config,*base)

def model(arch):
    torch.manual_seed(817);m=Agent(configurations(arch)[0]).double().eval()
    with torch.no_grad():
        m.actor[-1].weight.zero_();m.actor[-1].bias[:4]=.15
        m.actor[-1].weight[5,0]=10.;m.actor[-1].bias[5]=-9.9999
    return m

def assert_same(a,b):
    assert len(a)==len(b)
    def compare(x,y):
        if isinstance(x,dict):
            x={k:v for k,v in x.items() if k!='zero_replay_certified'};y={k:v for k,v in y.items() if k!='zero_replay_certified'}
            assert set(x)==set(y)
            for k in x:compare(x[k],y[k])
        elif isinstance(x,list):
            assert len(x)==len(y)
            for xx,yy in zip(x,y):compare(xx,yy)
        elif isinstance(x,float):np.testing.assert_allclose(x,y,atol=1e-10,rtol=1e-12)
        else:assert x==y
    for x,y in zip(a,b):compare(x,y)

@pytest.mark.parametrize('arch',ARCHS)
def test_all_off_trace_is_exact_even_without_global_gate_bound(arch,tmp_path,monkeypatch,base):
    install(monkeypatch,tmp_path,base);m=model(arch);assert not always_off(m)[0]
    try:
        fast,meta=cache.accelerated(m,'estimated',SEEDS,LEVELS,CFG)
        full=evaluate(m,condition='estimated',case_pairs=PAIRS,config=CFG,workers=8,fixed_workers=True)
        assert meta['trajectory_certified_zero_cases']==4 and meta['fully_simulated_policy_cases']==0
        assert_same(fast,full)
    finally:close_pool()

@pytest.mark.parametrize('arch',ARCHS)
def test_any_nonzero_proposal_forces_entire_case_simulation(arch,tmp_path,monkeypatch,base):
    install(monkeypatch,tmp_path,base);m=model(arch)
    with torch.no_grad():m.actor[-1].weight[5,0]=1.;m.actor[-1].bias[5]=0.
    maxima=[]
    for frames in base[1]:
        scores=[];hook=m.actor.register_forward_hook(lambda module,args,out:scores.append(float(out[0,5].detach())))
        state=None
        with torch.no_grad():
            for frame in frames:_,state=m.predict_actions(torch.tensor(frame[None],dtype=torch.float64),state)
        hook.remove();maxima.append(max(scores))
    assert max(maxima)-min(maxima)>1e-6
    m.gate_logit_threshold=float((min(maxima)+max(maxima))/2)
    try:
        fast,meta=cache.accelerated(m,'estimated',SEEDS,LEVELS,CFG)
        full=evaluate(m,condition='estimated',case_pairs=PAIRS,config=CFG,workers=8,fixed_workers=True)
        assert 0<meta['trajectory_certified_zero_cases']<4 and meta['fully_simulated_policy_cases']>0
        assert_same(fast,full)
    finally:close_pool()

def test_effectiveness_mode_replays_its_observation_context(tmp_path,monkeypatch,base):
    install(monkeypatch,tmp_path,base);m=model('lstm');config=dict(CFG,intervention='effectiveness')
    try:
        fast,meta=cache.accelerated(m,'estimated',SEEDS,LEVELS,config)
        full=evaluate(m,condition='estimated',case_pairs=PAIRS,config=config,workers=8,fixed_workers=True)
        assert meta['trajectory_certified_zero_cases']==4;assert_same(fast,full)
    finally:close_pool()

def test_baseline_capture_matches_ordinary_evaluation(base):
    try:plain=evaluate(condition='estimated',case_pairs=PAIRS,config=CFG,workers=8,fixed_workers=True);assert_same(base[0],plain)
    finally:close_pool()
    assert all(x.shape==(r['episode_steps'],80) for r,x in zip(*base))

def test_corrupted_trace_is_rejected(tmp_path,monkeypatch,base):
    install(monkeypatch,tmp_path,base);key,payload=cache.identity('estimated',CFG);path=cache.filename(key,0.,525000)
    with np.load(path) as z:frames=z['frames'].copy();record=str(z['record'])
    frames[0,0]+=.1;np.savez_compressed(path,frames=frames,record=np.asarray(record));cache._MEMORY.clear()
    with pytest.raises(AssertionError,match='provenance'):cache.load_case(key,payload,0.,525000)

def test_parallel_atomic_writes_are_readable(tmp_path,monkeypatch,base):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(cache,'BANK',tmp_path/'atomic');key,payload=cache.identity('estimated',CFG);row,frames=base[0][0],base[1][0]
    with ThreadPoolExecutor(2) as pool:list(pool.map(lambda _:cache.store_case(key,payload,row,frames),range(10)))
    cache._MEMORY.clear();loaded=cache.load_case(key,payload,row['level'],row['seed']);np.testing.assert_array_equal(loaded[1],frames)


def test_inventory_uses_verified_replay_and_preserves_case_coverage(tmp_path,monkeypatch,base):
    from experiments.round5.inventory import cached_evaluation
    install(monkeypatch,tmp_path,base);m=model('lstm');checkpoint=tmp_path/'policy.pt';m.save(checkpoint);target=tmp_path/'cases.json'
    rows=cached_evaluation(checkpoint,target,'estimated',SEEDS,LEVELS,CFG)
    meta=json.loads(target.with_suffix('.meta.json').read_text())
    assert len(rows)==4 and meta['zero_action_replay']['trajectory_certified_zero_cases']==4
    assert {(r['seed'],r['level']) for r in rows}==set(PAIRS)
