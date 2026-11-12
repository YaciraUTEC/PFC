"""Create registered inputs and engineering prerequisites from a clean checkout."""
import json,sys,platform,datetime,hashlib,importlib.metadata
from pathlib import Path
from .evaluate import OUT
from .confirm import build_plan
from ..run import save
CONFIG=Path(__file__).with_name('configs')

def initialize(out=None):
    target=Path(OUT if out is None else out);target.mkdir(parents=True,exist_ok=True)
    copied=[]
    for template in (CONFIG/'protocols').glob('*.json'):
        destination=target/template.name
        if not destination.exists():destination.write_text(template.read_text(),encoding='utf-8');copied.append(template.name)
    cases=build_plan();digest=hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    reference=json.loads((CONFIG/'reference_inputs.json').read_text());assert digest==reference['case_plan_sha256'],'Registered case plan changed'
    if not (target/'confirmation_case_plan.json').exists():save(target/'confirmation_case_plan.json',cases)
    if not (target/'runtime_environment.json').exists():
        import torch
        packages=['torch','numpy','scipy','gymnasium','stable-baselines3','pybullet','ncps','matplotlib','pytest']
        save(target/'runtime_environment.json',dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.version,executable=sys.executable,platform=platform.platform(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,torch_cuda=torch.version.cuda,packages={p:importlib.metadata.version(p) for p in packages},source_authority='frozen_before_test.json pins sources and input templates',scope='Registered study settings copied unchanged;run receipts record actual execution times.'))
    return dict(copied_protocols=copied,registered_cases=len(cases),case_plan_sha256=digest)
if __name__=='__main__':print(json.dumps(initialize(),indent=2))
