"""Acquire unbiased natural-history counterfactual labels, fit, evaluate, stop."""
from .common import OUT,read,save,close_milestone
from .runner import stage,parallel_stages
from .power import awake
if __name__=='__main__':
    for marker in ['h1_supervision_repair_conclusion.json','h0_natural_fork_conclusion.json','h0_gate_pilot_conclusion.json','h0_composed_method_conclusion.json','h0_contracts_conclusion.json']:
        if not (OUT/marker).exists():raise RuntimeError('Close prerequisite '+marker)
    with awake():
        jobs=[('h2_labels_'+split,['-m','experiments.round7.gate_data','--split',split,'--groups','160','--workers','8'],'gate_data_'+split+'_complete.json') for split in ['train','calibration']]
        parallel_stages(jobs)
        audits={k:read(OUT/f'gate_data_{k}_complete.json') for k in ['train','calibration']}
        assert all(r['fork_method']=='natural_prefix_replay' and r['maximum_prefix_error']==0 for r in audits.values())
        close_milestone('h2_terminal_labels','Fresh complete option and baseline continuations start from exactly reproduced natural states; all biased save/restore labels excluded.',audits=audits)
        close_milestone('h2_natural_labels','All160 train and160 calibration groups regenerated with independent natural prefix replay and all four options.',audits=audits)
        stage('h2_gate_training',['-m','experiments.round7.gate_fit'],'h2_gate_model_conclusion.json')
        stage('h2_gate_flights',['-m','experiments.round7.study','--phase','gate'],'h2_counterfactual_gate_conclusion.json')
        save(OUT/'status.json',dict(stage='h2_closed',status='complete'))
