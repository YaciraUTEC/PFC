"""Parallel affine prefix scan of the exact Mamba-1 recurrence."""
import torch
def affine_scan(a,b):
    length=a.shape[1];offset=1
    while offset<length:
        old_a=a
        b=torch.cat((b[:,:offset],b[:,offset:]+old_a[:,offset:]*b[:,:-offset]),dim=1)
        if offset*2<length:
            a=torch.cat((old_a[:,:offset],old_a[:,offset:]*old_a[:,:-offset]),dim=1)
        offset*=2
    return b
