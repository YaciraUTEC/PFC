"""Prospective allocation; test is inaccessible until development is closed."""
import time
from .common import OUT,save,read,MODEL_SEEDS,LEVELS

def create():
    path=OUT/'protocol.json'
    if path.exists():return read(path)
    p=dict(created_epoch=time.time(),architecture='gtrxl',base='frozen round6 DSLPID + structured observer + RecoveryDescent',
        source_commit='d6769e2bc5e9c23a8ed8f9a7dbf13e3245d0eb79',model_seeds=MODEL_SEEDS,
        stages=['h0_reproduction','h1_data_and_proposals','h2_counterfactual_gate','h3_filter_ablation','h4_protected_finetuning','h5_frozen_test','h6_report'],
        demonstration_groups=[16,64,256],demonstration_seeds=[700000,700256],development_seeds=[710000,710032],
        gate_training_seeds=[720000,720128],gate_calibration_seeds=[730000,730128],
        final_test_seeds=[740000,740064],confirmation_seeds=[750000,750320],shifted_clock_seeds=[760000,760032],
        levels=LEVELS,critical=[.295,.296],conditions=['exact','estimated'],noise=[1.,2.],fault_clocks=[None,4.,6.,8.],
        bc=dict(curve_seeds=[301,302,303],max_updates=3600,batch=8,length=96,burnin=32,lr=1e-4,warmup=200,min_lr=1e-5,checkpoints=[600,1200,2400,3600],
            description='Common nominal-noise proposal supervision, preserving sequence boundaries; all feasible expert maneuvers may teach proposals; authorization is separately paired.'),
        gate=dict(prefix_ticks=[1,6,12],amplitudes=[.25,.5,1.],terminal='candidate prefix then zero-residual DSLPID until strict terminal',
            supervision=['rescue','harm','both_success','both_fail'],deployable_inputs_only=True,calibration='route-separated; no ensemble disagreement called confidence bound'),
        online=dict(algorithm='FlashSAC only if selected in development',interactions_per_seed=524288,noise_schedule='linear .5..1 to .5..2 across 80% budget',encoder='frozen selected recurrent proposal; limitation disclosed'),
        selection=dict(primary='paired strict-success improvement at .295/.296 estimated, equal noise weights',
            protection='zero observed new losses versus base on validation for protected candidate; otherwise retain base as deployable recommendation',
            secondary=['contact margins','time remaining for 1s settlement','tail-risk','latency'],
            final='freeze one validation-selected candidate, five model seeds; no retuning on final data; report original FlashSAC reference separately',
            acceptance='positive route/model-clustered lower95CI primary and confirmation; no observed losses in protected test strata or shifted-clock checks; full-chain p95 below20.833ms',
            fallback='If no learned candidate passes, conclude retain base; do not change landing rules or assert impossibility of future improvement'),
        physical_filter='Empirically validated actuator-aware predictive screening, not advertised as a formal nonlinear reachability certificate.',
        costs='No Mamba fits, no new FQL benchmark; use previous evidence to focus on GTrXL + BC + FlashSAC and authorization.',
        final_test_locked=True)
    save(path,p);return p
if __name__=='__main__':create();print('PROTOCOL_WRITTEN',OUT/'protocol.json')
