"""Verify complete paired test coverage and the recorded touchdown contract."""
import hashlib,json
import numpy as np
from .models import ARCHS
from .train import SEEDS
from .env import LEVELS,STRESS
from .landing import LandingCriteria
from .evaluate import OUT
from ..run import save

def verify():
    c=LandingCriteria();expected=[]
    for controller in ['updated','updated_round3','robust','legacy']:
        for condition in ['exact','estimated','biased']:
            expected.append((OUT/'test'/f'baseline_{controller}_{condition}.json',LEVELS+STRESS))
    for condition in ['exact','estimated']:
        expected.append((OUT/'test'/f'metric_only_original_profile_{condition}.json',[0.,.25,.29]))
        for arch in ARCHS:
            for seed in SEEDS:
                for mode in ['bc','best','final']:
                    expected.append((OUT/'test'/f'{arch}_{seed}_{mode}_{condition}.json',LEVELS+STRESS))
            expected.append((OUT/'test'/f'{arch}_71_no_guard_{condition}.json',LEVELS))
            for suffix in ['', '_final']:
                expected.append((OUT/'historical'/f'v2_{arch}{suffix}_{condition}.json',LEVELS+STRESS))
        for seed in [11,22,33,44,55]:
            for name in ['bc','manual','csil','meirl','rime','manual_final','csil_final','meirl_final','rime_final']:
                expected.append((OUT/'historical'/f'round3_{seed}_{name}_{condition}.json',LEVELS+STRESS))
    total=0;reused=0;stable=0
    for path,levels in expected:
        rows=json.loads(path.read_text(encoding='utf-8'))
        keys=[(r['level'],r['seed']) for r in rows]
        assert len(keys)==len(set(keys))==len(levels)*30,path
        assert set(keys)=={(l,s) for l in levels for s in range(440000,440030)},path
        for r in rows:
            assert r['mission_success']==r['stable_landing'],path
            assert 0<r['episode_steps']<=1440,path
            assert np.isfinite([r['tracking_rmse'],r['mean_gate'],r['eta_mae'],r['stable_seconds']]).all(),path
            assert 0<=r['motor']<4,path
            if r['stable_landing']:
                f=r['first_contact'];assert r['outcome']=='stable_landing',path
                assert f and f['vertical_speed']<=c.first_vertical_speed and f['horizontal_speed']<=c.first_horizontal_speed,path
                assert f['tilt']<=c.first_tilt and f['angular_speed']<=c.first_angular_speed and f['radius']<=c.radius,path
                assert r['stable_seconds']>=c.hold_seconds-1e-9 and r['distance_to_goal']<=c.radius,path
                assert np.linalg.norm(r['terminal_velocity'])<=c.settled_speed,path
                assert np.linalg.norm(r['terminal_angular_velocity'])<=c.settled_angular_speed,path
                assert r['terminal_position'][2]<=c.settled_height,path
                stable+=1
        total+=len(rows)
        if path.with_suffix('.reuse.json').exists():
            source=json.loads(path.with_suffix('.reuse.json').read_text())['source']
            assert json.loads(open(source,encoding='utf-8').read())==rows,path
            reused+=len(rows)
    frozen=json.loads((OUT/'frozen_before_test.json').read_text())
    assert len(frozen['checkpoints'])==147
    for record in frozen['checkpoints']:
        assert hashlib.sha256((OUT/record['path']).read_bytes()).hexdigest()==record['sha256']
    assert len(expected)==320 and total==122760
    result=dict(result='passed',result_files=len(expected),logical_model_cases=total,
        cached_actor_cases=reused,simulated_case_records=total-reused,stable_model_cases=stable,
        frozen_checkpoints_verified=147,scope='coverage,unique paired cases,finite metrics,recorded strict landing contract,cache copies and checkpoint SHA')
    save(OUT/'results_integrity.json',result)
    return result
if __name__=='__main__':print(verify())
