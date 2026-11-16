"""Reproduce old policy and verify branches against natural execution itself."""
import time
import numpy as np
import torch
from .common import OUT,OLD,config,read,sha,silent,old_model,close_milestone
from .protocol import create
from .episode import fly,load_policy
from .replay_fork import ReplayFork
from ..round6.env import LandingEnv,ZERO


def main():
    create();torch.set_num_threads(1);start=time.time()
    if (OUT/'h0_reproduction_conclusion.json').exists():return
    row=fly(612034,.296,'exact',1.,model=load_policy(old_model()));old=read(OLD/'test/flashsac_gtrxl_linear_201_exact_noise1.json');expected=next(r for r in old['episodes'] if r['seed']==612034 and r['level']==.296)
    assert row['outcome']==expected['outcome'];err=abs(row['first_contact']['vertical_speed']-expected['first_contact']['vertical_speed']);assert err<1e-10
    with silent():e=LandingEnv(condition='estimated',config=config(noise_multiplier=2.))
    try:
        with silent():obs,_=e.reset(seed=710000,options=dict(level=.296))
        history=[]
        for _ in range(300):history.append(ZERO.copy());obs,_,d,t,_=e.step(ZERO);assert not(d or t)
        with ReplayFork(e,history) as fork:
            outputs=[]
            for trial in range(3):
                target=e if trial==0 else fork.restore()
                one=[]
                for k in range(64):
                    a=np.array([.04,-.02,.03,-.01,-.6,1.],np.float32) if k<6 else ZERO
                    oo,r,d,t,ii=target.step(a);one.append(np.r_[oo[-1],r,target.estimated_eta,target.c.integral_pos_e,target.c.integral_rpy_e]);assert not(d or t)
                outputs.append(np.asarray(one))
            difference=float(max(np.max(np.abs(x-outputs[0])) for x in outputs[1:]));assert difference==0.,difference
            assert fork.env.descent.controller is fork.env.c
        assert e.descent.controller is e.c
    finally:e.close()
    close_milestone('h0_reproduction','Frozen GTrXL reproduced; independent prefix replays match the natural original trajectory exactly. PyBullet save/restore is rejected for training labels.',seconds=time.time()-start,old_model_sha256=sha(old_model()),reproduction_vertical_speed_error=err,natural_replay_max_error=difference,baseline_config=config(),scope='Original results, demonstrations and proposal models remain immutable')
    close_milestone('h0_natural_fork','Natural caller and two independently reconstructed branches agree exactly, including observer and controller state.',maximum_error=difference,method='natural_prefix_replay')
if __name__=='__main__':main()
