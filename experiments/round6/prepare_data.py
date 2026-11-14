import torch
from .data import collect
from .train import selected_config
if __name__=='__main__':
    import time
    from .evaluate import OUT
    while not (OUT/'phase1_physical_closed.json').exists() or not (OUT/'phase1_corrections_closed.json').exists():time.sleep(5)
    torch.set_num_threads(1);collect(selected_config(),count=16,workers=8)
    from .data_audit import audit
    audit()
