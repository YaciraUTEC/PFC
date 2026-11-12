"""Runtime proposal evaluation, including actions rejected by the internal gate."""
from torch import nn

class ProposalPolicy(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def predict_actions(self,obs,state=None):
        _,proposal,state=self.base.propose_actions(obs,state);return proposal,state
