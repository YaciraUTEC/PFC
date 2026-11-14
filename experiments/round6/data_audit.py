"""Transition/episode contracts checked before any round6 learner sees data."""
from collections import Counter
import numpy as np
from .data import load
from .evaluate import OUT,save
from .env import ZERO

def audit():
    data=load();n=len(data['s']);assert n>0
    for key,value in data.items():
        if isinstance(value,np.ndarray):assert np.isfinite(value).all(),key
    assert data['s'].shape==data['ns'].shape==(n,88)
    assert data['executed'].shape==(n,6)
    assert (np.abs(data['executed'])<=1+1e-6).all()
    assert (data['motor_command_rpm']>=-1e-6).all() and (data['motor_command_rpm']<=21700+1e-3).all()
    assert (np.abs(data['projected_motor_delta'])<=1+1e-4).all()
    offset=0;keys=set();levels=Counter();experts=0;positives=0
    for r in data['records']:
        a,b=r['start'],r['stop'];assert a==offset and a<b;offset=b
        assert not data['done'][a:b-1].any() and data['done'][b-1]
        np.testing.assert_array_equal(data['ns'][a:b-1],data['s'][a+1:b])
        key=tuple(r[k] for k in ['seed','level','condition','noise','behavior']);assert key not in keys;keys.add(key)
        assert 620000<=r['seed']<622000
        if r['expert']:
            assert r['stable_landing'];np.testing.assert_array_equal(data['a'][a:b],data['executed'][a:b]);experts+=1
        off=data['executed'][a:b,5]<0
        if off.any():
            np.testing.assert_array_equal(data['executed'][a:b][off],np.tile(ZERO,(int(off.sum()),1)))
            np.testing.assert_array_equal(data['ns'][a:b][off,35:39],np.zeros((int(off.sum()),4),np.float32))
        levels[str(r['level'])]+=b-a;positives+=int((data['valid'][a:b]&(data['a'][a:b,5]>0)).sum())
    assert offset==n and experts>0
    result=dict(passed=True,transitions=n,episodes=len(data['records']),expert_episodes=experts,positive_bc_interventions=positives,steps_by_level=dict(levels),
        states_finite=True,next_state_continuity=True,one_terminal_per_episode=True,actuator_limits_preserved=True,actions_are_executed_requests_not_counterfactual_teacher=True)
    save(OUT/'data_audit.json',result);print('DATA_AUDIT',result,flush=True);return result
if __name__=='__main__':audit()
