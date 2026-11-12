"""Physical proposal regression with class-balanced binary benefit supervision.
The gate is a decision score,not a calibrated probability under balanced loss.
"""
import torch
from torch.nn import functional as F

def clone_loss(mu,target,mask,gate_valid=None):
    a=mu[...,:5].tanh();amount=(a[...,4]+1)/2;target_amount=(target[...,4]+1)/2;on=(target[...,5]+1)/2
    effective=(amount[...,None]*a[...,:4]-target_amount[...,None]*target[...,:4]).square().mean(-1)
    direction=(a[...,:4]-target[...,:4]).square().mean(-1)*target_amount
    regression=(3*effective+.1*(amount-target_amount).square()+.05*direction)*mask
    valid=mask if gate_valid is None else mask*gate_valid
    terms=F.binary_cross_entropy_with_logits(mu[...,5],on,reduction='none')
    positive=valid*on;negative=valid*(1-on);np_=positive.sum();nn_=negative.sum()
    if bool(np_>0) and bool(nn_>0):gate=(terms*positive).sum()/np_/3+2*(terms*negative).sum()/nn_/3
    else:gate=(terms*valid).sum()/valid.sum().clamp_min(1)
    return regression.sum()/mask.sum().clamp_min(1)+.3*gate
