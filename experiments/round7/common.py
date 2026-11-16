"""Paths, immutable inputs and atomic receipts for the sequential round7 study."""
import os,json,time,hashlib,contextlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
TASK=ROOT.parent
OLD=TASK/'PFC/runs/intervention_round6'
LEGACY_OUT=ROOT/'runs/gtrxl_round7'
OUT=ROOT/'runs/gtrxl_round7_replay'
MODEL_SEEDS=[301,302,303,304,305]
LEVELS=[0.,.02,.10,.25,.28,.29,.295,.296,.30]

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,default=serial,allow_nan=False),encoding='utf-8');temp.replace(path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config(**kw):
    result=dict(read(OLD/'phase1_conclusion.json')['config']);result['gate_mode']='learned';result.update(kw);return result
def old_model(seed=201,kind='flashsac'):
    if kind=='bc':return OLD/'bc/gtrxl'/f'seed_{seed}'/'bc.pt'
    return OLD/'phase3'/kind/'gtrxl'/f'seed_{seed}'/'best.pt'
@contextlib.contextmanager
def silent():
    with open(os.devnull,'w') as handle,contextlib.redirect_stdout(handle):yield

def close_milestone(name,conclusion,**evidence):
    save(OUT/(name+'_conclusion.json'),dict(closed=True,finished_epoch=time.time(),conclusion=conclusion,**evidence))
    print('MILESTONE_CLOSED',name,conclusion,flush=True)

def gate_path(seed=301,feedback=False):
    stem='outcome_gate_feedback' if feedback else 'outcome_gate'
    return OUT/(stem+('' if seed==301 else '_seed_'+str(seed))+'.pt')
