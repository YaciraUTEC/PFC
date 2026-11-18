"""Isolated, append-only round8 evidence and immutable round7 references."""
from pathlib import Path
import json,time,hashlib,contextlib,os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
TASK=ROOT.parent
OUT=ROOT/'runs/gtrxl_round8'
R7=TASK/'PFC_round7/runs/gtrxl_round7_replay'
LEVELS=[0.,.02,.10,.25,.28,.29,.295,.296,.30]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 def cast(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,Path):return str(x)
  raise TypeError(type(x).__name__)
 t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(d,indent=2,default=cast,allow_nan=False),encoding='utf-8');t.replace(p)
def close(name,conclusion,**evidence):
 save(OUT/(name+'_conclusion.json'),dict(closed=True,finished_epoch=time.time(),conclusion=conclusion,**evidence));print('MILESTONE_CLOSED',name,conclusion,flush=True)
def verify_round7():
 frozen=read(R7/'frozen_before_final.json');rr=TASK/'PFC_round7'
 for name,d in frozen['sources'].items():assert sha(rr/name)==d,name
 for name,d in frozen['artifacts'].items():assert sha(name)==d,name
 assert sha(R7/'selected_candidate.json')==frozen['selection_sha']
 assert read(R7/'ALL_COMPLETE.json')['complete']
 return dict(source_count=len(frozen['sources']),artifact_count=len(frozen['artifacts']),final_manifest_sha=sha(R7/'FINAL_ARTIFACTS_MANIFEST.json'))
