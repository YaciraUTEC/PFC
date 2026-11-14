"""Physics-only decision inputs prevent autoregressive intervention-label leakage."""
import torch
from ..round5.models import Agent as PreviousAgent
from ..round4.models import StreamingCore,tree_map,detach,select,concatenate
from .env import DIM
ARCHS=['mamba3','gtrxl']
INPUT_VERSION='physics_only_onset_balanced_v1'
def physics_observation(obs):
    clean=obs.clone()
    clean[...,35:39]=0.;clean[...,39]=-1.
    clean[...,59]=0.;clean[...,77:79]=-1.
    return clean
class PhysicsMemory(StreamingCore):
    def forward(self,obs,state=None):return super().forward(physics_observation(obs),state)
class Agent(PreviousAgent):
    def __init__(self,cfg,dim=DIM):
        cfg=dict(cfg,decision_inputs=INPUT_VERSION);super().__init__(cfg,dim)
        self.encoder=PhysicsMemory(dim,cfg)
    def forward(self,obs,state=None):return super().forward(physics_observation(obs),state)
def configuration(arch):
    from ..round5.bc import starting_config
    cfg=starting_config(arch);cfg['lr']=5e-5 if arch=='mamba3' else 2.5e-5
    return cfg
