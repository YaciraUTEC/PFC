"""Engineering timing/parity on reserved routes with an already completed model."""
import time,json
import torch
from .evaluate import OUT,evaluate,close_pool
from .inventory import core,cached_evaluation,sha
from .models import load_agent
from ..run import save

def main():
    torch.set_num_threads(1);p=OUT/'models/lstm/seed_101/best.pt'
    assert (p.parent/'complete.json').exists()
    before=sha(p);cfg=core();seeds=range(528024,528028);levels=[0.,.25,.295]
    dest=OUT/'replay_benchmark'/str(int(time.time()));dest.mkdir(parents=True)
    try:
        m=load_agent(p);start=time.perf_counter();full=evaluate(m,condition='estimated',seeds=seeds,levels=levels,config=cfg);full_seconds=time.perf_counter()-start
        start=time.perf_counter();cold=cached_evaluation(p,dest/'cold.json','estimated',seeds,levels,cfg);cold_seconds=time.perf_counter()-start
        start=time.perf_counter();warm=cached_evaluation(p,dest/'warm.json','estimated',seeds,levels,cfg);warm_seconds=time.perf_counter()-start
        start=time.perf_counter();full_warm=evaluate(m,condition='estimated',seeds=seeds,levels=levels,config=cfg);full_warm_seconds=time.perf_counter()-start
    finally:close_pool()
    assert sha(p)==before
    for other in [cold,warm,full_warm]:
        assert all(a['outcome']==b['outcome'] and a['episode_steps']==b['episode_steps'] and abs(a['tracking_rmse']-b['tracking_rmse'])<1e-10 for a,b in zip(full,other))
    meta=json.loads((dest/'warm.meta.json').read_text())
    result=dict(scope='Reserved validation only;actual completed seed101LSTM,not final-test evidence',cases=len(full),checkpoint_sha256=before,full_seconds_including_pool_startup=full_seconds,full_warm_seconds=full_warm_seconds,cold_cache_seconds=cold_seconds,warm_cache_seconds=warm_seconds,warm_speedup=full_warm_seconds/warm_seconds,timing_note='Compare warm physical simulation to warm helper;first full measurement includes process startup.Savings depend on actual zero-action cases;not an architecture throughput claim.',canonical_zero_cases=meta.get('zero_action_replay'),outcomes_steps_rmse_identical=True)
    save(dest/'benchmark.json',result);save(OUT/'exact_replay_benchmark.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
