"""One protected refinement plus one feedback-data aggregation, then stop."""
from .common import OUT,gate_path,save
from .runner import stage,parallel_stages
from .power import awake
if __name__=='__main__':
    if not (OUT/'h3_filter_ablation_conclusion.json').exists():raise RuntimeError('Close filter ablation first')
    if not (OUT/'h0_composed_acquisition_pilot_conclusion.json').exists():raise RuntimeError('Verify the composed-student acquisition pilot first')
    with awake():
        for seed in [302,303,304,305]:
            stage('h4_original_gate_'+str(seed),['-m','experiments.round7.gate_fit','--seed',str(seed+6),'--output',gate_path(seed).name,'--milestone','h4_original_gate_'+str(seed)],'h4_original_gate_'+str(seed)+'_conclusion.json')
        stage('h4_head_refinement',['-m','experiments.round7.refine','--phase','train'],'h4_supervised_refinement_conclusion.json')
        jobs=[('h4_feedback_'+split,['-m','experiments.round7.gate_data','--split',split,'--groups','240','--workers','8','--proposers-json',str(OUT/'refinement_proposers.json'),'--behavior-specs-json',str(OUT/'refinement_behavior_specs.json')],'h4_feedback_'+split+'_complete.json') for split in ['train','calibration']]
        parallel_stages(jobs)
        for split in ['train','calibration']:save(OUT/('h4_feedback_'+split+'_complete.json'),dict(complete=True,total_groups=240,new_groups=80,independent_splits_acquired_in_parallel=True))
        for seed in [301,302,303,304,305]:
            stage('h4_feedback_gate_'+str(seed),['-m','experiments.round7.gate_fit','--seed',str(seed+6),'--output',gate_path(seed,True).name,'--milestone','h4_feedback_gate_'+str(seed)],'h4_feedback_gate_'+str(seed)+'_conclusion.json')
        stage('h4_validation',['-m','experiments.round7.refine','--phase','select'],'h4_protected_finetuning_conclusion.json')
        save(OUT/'status.json',dict(stage='h4_closed',status='complete'))
