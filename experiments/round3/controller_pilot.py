from pathlib import Path
import json,time,numpy as np
from .env import TrialEnv
from ..run import save

def main():
    rows=[];start=time.time()
    for controller in ['legacy','robust','updated']:
        for condition in ['exact','uncertain']:
            e=TrialEnv(controller=controller,condition=condition)
            try:
                for level in [0.,.02,.1,.2,.25,.3,.5]:
                    e.set_fixed_fault(level);success=0
                    for seed in range(100000,100010):
                        o,_=e.reset(seed=seed)
                        while True:
                            o,_,d,t,info=e.step(np.r_[np.zeros(4),-1.])
                            if d or t:break
                        success+=info['mission_success'];rows.append(dict(controller=controller,condition=condition,level=level,seed=seed,**{k:v for k,v in info.items() if k not in ['state','action','next_state']}))
                    print(controller,condition,level,success,'/10',flush=True)
            finally:e.close()
    save(Path(__file__).resolve().parents[3]/'runs/controller_round3_pilot.json',dict(seconds=time.time()-start,episodes=rows))
if __name__=='__main__':main()