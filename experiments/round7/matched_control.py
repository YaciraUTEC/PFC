"""Matched no-new-data control fills the acquisition wait without GPU overlap."""
import time
from .common import OUT,read,save,close_milestone
from .power import awake
from .bc import fit
if __name__=='__main__':
    with awake():
        while not (OUT/'h1_scale_64_conclusion.json').exists():time.sleep(5)
        start=time.time();rows=[fit(0,s) for s in [301,302,303]]
        close_milestone('h1_matched_original_data','The original demonstrations were trained with exactly the new optimizer, sampler, seeds and update budget; quantity comparison is not confounded by a changed training recipe.',results=rows,seconds=time.time()-start)
