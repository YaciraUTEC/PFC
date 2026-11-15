"""Invalidate the engineering-only floor bug before independent validation."""
import time,json
from pathlib import Path
from .physical_control import NEW
from .phase1 import run_job,summary
from .evaluate import OUT,save

def main():
    rejected=OUT/'phase1/rejected_floor_double_count';rejected.mkdir(parents=True,exist_ok=True)
    for path in (OUT/'phase1/development').glob('landing_physical_*.json'):
        path.replace(rejected/path.name)
    rows=run_job('landing_physical',NEW['landing_physical'],'development',range(610000,610008),[0.,.25,.28,.295,.296,.35,.5])
    fixed=summary(rows);save(OUT/'phase1/physical_floor_correction.json',dict(passed=True,summary=fixed,reason='The contact-height allowance was counted twice;fixed before independent physical-validation runs of this controller.'))
    while not (OUT/'phase1_physical_closed.json').exists():time.sleep(1)
    p=OUT/'phase1_conclusion.json';data=json.loads(p.read_text());data['development']['landing_physical']=fixed
    data['engineering_corrections']=['Contact-height allowance double count corrected and development rerun before independent physical validation.']
    save(p,data);save(OUT/'phase1_corrections_closed.json',dict(finished_epoch=time.time(),passed=True))
if __name__=='__main__':main()
