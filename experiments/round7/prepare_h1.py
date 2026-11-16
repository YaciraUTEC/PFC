"""Close proposal selection and complete the selected scale's five seeds."""
from .common import OUT,read,close_milestone
from .proposal_validation import select
from .bc import fit
from .power import awake
if __name__=='__main__':
    if not (OUT/'h1_data_and_proposals_conclusion.json').exists() or not (OUT/'h1_matched_original_data_conclusion.json').exists():raise RuntimeError('Complete the matched scale fits first')
    with awake():
        if not (OUT/'h1_proposal_selection_conclusion.json').exists():select()
        groups=read(OUT/'h1_proposal_selection_conclusion.json')['selected_groups'];rows=[fit(groups,s) for s in [304,305]]
        close_milestone('h1_five_models','The selected proposal scale now has five independent GTrXL training seeds for the subsequent authorizer and final evaluation.',selected_groups=groups,additional_fits=rows)
