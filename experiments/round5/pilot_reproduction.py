"""Check the historical pilot's exact initialization/loss reproduction."""
import time
import torch
from .evaluate import OUT
from .models import Agent,clone_loss as historical_loss
from .bc import setup,pretrain,starting_config
from .data import load,SequenceData
from ..run import save

def main():
    torch.set_num_threads(1);start=time.time();directory=OUT/'initialization_audit/lstm_historical_pilot';directory.mkdir(parents=True,exist_ok=True)
    data=load(include_dagger=False);dataset=SequenceData(data,'residual');setup(97);m=Agent(starting_config('lstm')).cuda()
    pretrain(m,dataset,97,directory,updates=400,loss_function=historical_loss)
    before=Agent.load(OUT/'pilots/residual/lstm/bc.pt',device='cpu').state_dict();after=m.cpu().state_dict()
    differences={k:float((before[k].double()-after[k].double()).abs().max()) for k in before}
    exact=all(torch.equal(before[k],after[k]) for k in before)
    result=dict(architecture='lstm',seed=97,updates=400,historical_loss=True,bit_identical=exact,max_parameter_error=max(differences.values()),seconds=time.time()-start,scope='Auxiliary software reproducibility only;does not change or replace any training pilot or principal policy.')
    save(OUT/'pilot_reproduction_audit.json',result);print('PILOT_REPRODUCTION',result,flush=True)
    assert exact,'Historical pilot reproduction differs;inspect before claiming exact reproduction'
if __name__=='__main__':main()
