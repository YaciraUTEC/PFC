"""Run the complete suite after training,before runtime selection and test freeze."""
import sys,subprocess,time,datetime
from pathlib import Path
from .evaluate import OUT
from ..run import save

def run_tests():
    start=time.time()
    with (OUT/'tests_final.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-m','pytest','experiments/tests','-q','--junitxml='+str(OUT/'tests_final.xml')],cwd=Path.cwd(),stdout=log,stderr=subprocess.STDOUT)
    save(OUT/'tests_final_receipt.json',dict(exit_code=result.returncode,seconds=time.time()-start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='All training complete,before runtime selection and final test freeze'))
    if result.returncode:raise RuntimeError('Final regression suite failed;see tests_final.log')
    print('FINAL_REGRESSION_SUITE_PASSED',round(time.time()-start,2),'seconds',flush=True)
if __name__=='__main__':run_tests()
