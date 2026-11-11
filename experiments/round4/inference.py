"""Stable reference inference:float64 network,one float32 action boundary.
Checkpoints remain their original float32 learned weights,unchanged on disk.
"""
import numpy as np
import torch
from stable_baselines3 import PPO
from .models import Agent
def load_agent(path,device='cpu'):
    return Agent.load(path,device=device).double().eval()

class HistoricalActor:
    def __init__(self,model,dtype=torch.float64):
        self.model=model;self.policy=model.policy.to(dtype=dtype).eval()
        for module in self.policy.modules():
            if hasattr(module,'pos_embedding_dict'):module.pos_embedding_dict.clear()
            if hasattr(module,'att_mask'):module.att_mask.clear()
        self.device=model.device;self.dtype=dtype
        self.policy_kwargs=model.policy_kwargs;self.observation_space=model.observation_space
        self.action_space=model.action_space
    @torch.no_grad()
    def predict(self,obs,deterministic=True):
        if not deterministic:raise ValueError('Reference adapter is deterministic only')
        x=torch.as_tensor(obs,device=self.device,dtype=self.dtype)
        z=self.policy.pi_features_extractor(x)
        mu=self.policy.action_net(self.policy.mlp_extractor.forward_actor(z))
        # Historical round3 custom bounded policy has5actions;v2 normal policy4.
        action=mu.tanh() if self.action_space.shape[0]==5 else mu.clamp(-1,1)
        return action.cpu().numpy(),None
def load_historical(path,device='cpu',dtype=torch.float64):
    return HistoricalActor(PPO.load(path,device=device),dtype=dtype)
