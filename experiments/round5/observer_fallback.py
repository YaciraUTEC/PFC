"""Reserve an estimator-only confirmation arm before opening any final test."""
import json,time,datetime
from .evaluate import OUT,evaluate,close_pool
from ..run import save

def main():
    start=time.time();selection=json.loads((OUT/'core_selection.json').read_text());original=next(r for r in selection['validation'] if r['label']=='original')
    candidates=[r for r in selection['validation'] if r['config']['estimator']!='original' and not r['config']['readiness'] and r['config']['emergency']=='legacy' and r['config']['reserve']==.005 and r['risk_losses']==0 and r['severe_success']>original['severe_success']]
    if not candidates:
        save(OUT/'observer_fallback_selection.json',dict(production_eligible=False,reason='No estimator-only validation improvement',seconds=time.time()-start));return
    chosen=max(candidates,key=lambda r:r['severe_success']);cfg=chosen['config'];losses=0;records=[]
    try:
        for condition in ['exact','estimated']:
            base=json.loads((OUT/'structural_challenger'/f'original_{condition}.json').read_text());base={(r['level'],r['seed']):r for r in base}
            rows=evaluate(condition=condition,seeds=range(528000,528032),levels=[.15],config=cfg)
            losses+=sum(base[(r['level'],r['seed'])]['stable_landing'] and not r['stable_landing'] for r in rows);records.extend([dict(r,condition=condition) for r in rows])
    finally:close_pool()
    save(OUT/'observer_fallback_validation_15percent.json',records)
    save(OUT/'observer_fallback_selection.json',dict(production_eligible=bool(losses==0),label=chosen['label'],config=cfg,validation=chosen,supplemental15percent_losses=int(losses),seconds=time.time()-start,
        rule='Choose estimator-only arm by reserved32route validation;zero protected regressions including supplemental15percent.Estimator-only is confirmed before accepting the more complex selectedcore.',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),final_test_unopened=True))
    print('OBSERVER_ONLY_CONFIRMATION_ARM',cfg,'eligible',losses==0,flush=True)
if __name__=='__main__':main()
