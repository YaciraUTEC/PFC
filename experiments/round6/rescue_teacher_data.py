"""State-triggered recoveries/returns enrich gate onsets without previous-gate cues."""
import hashlib,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from .evaluate import OUT,save
from .train import selected_config
from .mpc_teacher_data import collect_route
from .data_audit import audit

def main():
    start=time.time();control=json.loads((OUT/'phase1_conclusion.json').read_text());name=control['best_slow_nmpc'];options=control['all_configs'][name]['nmpc_options'];config=selected_config()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(collect_route,i,config,options,'rescue') for i in range(8)]):print('RESCUE_EXPERT_DATA',f.result(),flush=True)
    result=audit();files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'data').glob('route_*.npz'))}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest();save(OUT/'dataset_fingerprint.json',dict(sha256=digest,files=files))
    save(OUT/'rescue_expert_data_complete.json',dict(seconds=time.time()-start,dataset_sha256=digest,passed=True,audit=result,
        cue='estimated_fault and yaw>.8for4ticks;return afteryaw<.35andhorizontal_speed<.20for24ticks;student does not receive the trigger flag.'))
if __name__=='__main__':main()
