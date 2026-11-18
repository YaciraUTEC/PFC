"""Resume matched development with validated equivalent execution, retaining provenance."""
from pathlib import Path
import torch
from .common import OUT,read,save,sha,close
from .cache_fast import run as accelerated
from . import active_flow
from ..round7.evaluation import key
from ..round7.power import awake

def execute(spec,base,folder,**kw):
 dest=Path(folder)/'results.json'
 if dest.exists():
  r=read(dest);assert r['spec']==spec;assert {key(x) for x in r['episodes']}=={key(x) for x in base['episodes']}
  for name,d in r['fingerprint'].items():
   p=Path(spec[name]) if name in ['gate','proposer'] else Path(__file__).parent/name
   assert sha(p)==d,'Completed inference inputs changed: '+name
  if 'fast_policy.py' not in r['fingerprint']:
   proof=read(OUT/'h8_fast_execution_equivalence_conclusion.json');assert proof['passed'];save(Path(folder)/'validated_slow_reference_reuse.json',dict(original_result_sha=sha(dest),original_fingerprint=r['fingerprint'],equivalence_receipt_sha=sha(OUT/'h8_fast_execution_equivalence_conclusion.json'),scope='Original slow full-flight result retained unchanged as a mathematically identical control-policy reference. Not a new simulation or independent sample.'))
  return r
 return accelerated(spec,base,folder,**kw)

def main():
 torch.set_num_threads(1);assert read(OUT/'h8_fast_execution_equivalence_conclusion.json')['passed'];active_flow.run=execute
 with awake():active_flow.validation()
if __name__=='__main__':main()
