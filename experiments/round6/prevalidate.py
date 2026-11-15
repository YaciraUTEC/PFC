from .phase1 import run_job,BASE
from .structured import VARIANTS as STRUCT
from .engineering import VARIANTS as ENGINEERING

def main():
 for name,cfg in [('structured_only',STRUCT['structured_only']),('structured_governor',STRUCT['structured_governor']),('reserve_only',ENGINEERING['reserve_only'])]:
  run_job(name,cfg,'validation',range(611000,611024),[0.,.02,.25,.28,.29,.295,.296,.30,.35,.40,.50])
if __name__=='__main__':main()
