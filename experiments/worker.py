"""One independent training worker; only per-architecture outputs are mutated."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,multiprocessing,traceback
from .run import ROOT,ARCHS,train_one,save

def main():
    p=argparse.ArgumentParser();p.add_argument('architecture',choices=ARCHS);args=p.parse_args()
    out=ROOT/'runs/dslpid_comparison_v2';arch=args.architecture
    config=SimpleNamespace(history=32,policy_envs=32,envs=8,seed=42)
    reward=out/'reward.json'
    lr=json.loads((out/'hyperparameter_selection.json').read_text(encoding='utf-8'))['chosen_lr']
    save(out/f'worker_status_{arch}.json',dict(stage='training',architecture=arch,steps=524288))
    try:
        result=train_one(out,reward,config,arch,lr,524288,arch,50)
        save(out/f'worker_status_{arch}.json',dict(stage='complete',architecture=arch,
             steps=result['actual_steps'],training_seconds=result['training_seconds']))
        print('WORKER COMPLETE',arch,flush=True)
    except Exception as error:
        save(out/f'worker_status_{arch}.json',dict(stage='failed',architecture=arch,error=str(error)))
        raise
if __name__=='__main__':multiprocessing.freeze_support();main()