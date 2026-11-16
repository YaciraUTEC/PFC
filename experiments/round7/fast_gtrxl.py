"""Frozen CPU GTrXL inference: reuse K/V projections of unchanged memory.
The public raw-memory state and all learned parameters stay unchanged. Full
sequences, training and non-CPU/f64 calls use the original implementation. A
cache hit requires exact equality of the supplied raw memory and the previous
output memory, so episode resets and counterfactual branch switches invalidate it.
"""
from types import MethodType
import torch
from ..vendor_gtrxl import GTrXL


def clear(core):
    core._kv_expected=None;core._kv_projected=None;core._kv_positions=None


def step_forward(self,x,batch_first=False,return_mem=True):
    sequence=x.transpose(0,1) if batch_first else x
    cur,batch=sequence.shape[:2]
    if self.training or cur!=1 or x.is_cuda or x.dtype!=torch.float64 or self.use_embedding_layer:
        clear(self)
        if any(v.device!=x.device or v.dtype!=x.dtype for v in self.pos_embedding_dict.values()):self.pos_embedding_dict.clear()
        return self._reference_forward(x,batch_first=batch_first,return_mem=return_mem)
    memory=None if self.memory is None else self.memory.get()
    mask=self.att_mask.get(1)
    if memory is None or memory.shape[1]!=self.memory_len or memory.shape[-2:]!=(batch,self.embedding_dim) or (mask is not None and bool(mask.any())):
        clear(self);return self._reference_forward(x,batch_first=batch_first,return_mem=return_mem)
    hit=self._kv_expected is not None and self._kv_expected.shape==memory.shape and torch.equal(memory,self._kv_expected)
    if hit:self._kv_hits+=1
    else:self._kv_misses+=1
    pos=self.pos_embedding_dict.get(1)
    if pos is None or pos.device!=x.device or pos.dtype!=x.dtype:
        pos_ips=torch.arange(self.memory_len,-1,-1.,dtype=torch.float,device=x.device);pos=self.pos_embedding(pos_ips);self.pos_embedding_dict[1]=pos
    if self._kv_positions is None:self._kv_positions=[layer.attention.project_pos(pos).view(self.memory_len+1,self.head_num,self.head_dim) for layer in self.layers]
    out=sequence;hidden=[out];updated=[]
    for i,layer in enumerate(self.layers):
        att=layer.attention
        old=self._kv_projected[i] if hit else att.attention_kv(layer.layernorm1(memory[i]))
        current=att.attention_kv(layer.layernorm1(out));kv=torch.cat([old,current],0)
        keys,values=torch.chunk(kv,2,-1);keys=keys.view(self.memory_len+1,batch,self.head_num,self.head_dim);values=values.view_as(keys)
        query=att.attention_q(out).view(1,batch,self.head_num,self.head_dim);r=self._kv_positions[i]
        content=(query+self.u).permute(1,2,0,3)@keys.permute(1,2,3,0)
        position=(query+self.v).permute(1,2,0,3)@r.permute(1,2,0);position=att._rel_shift(position)
        scores=content+position;scores.mul_(att.scale);weights=att.dropout(scores.softmax(-1))
        vector=weights@values.permute(1,2,0,3);vector=vector.permute(2,0,1,3).contiguous().view(1,batch,self.head_num*self.head_dim)
        a=layer.activation(layer.dropout(att.dropout(att.project(vector))))
        first=layer.gate1(out,a) if layer.gating else out+a
        second=layer.dropout(layer.mlp(layer.layernorm2(first)))
        out=layer.gate2(first,second) if layer.gating else first+second
        hidden.append(out.clone());updated.append(kv[1:].detach())
    out=self.dropout(out);self.memory.update(hidden)
    self._kv_expected=self.memory.get().detach();self._kv_projected=updated
    if batch_first:out=out.transpose(0,1)
    result=dict(logit=out)
    if return_mem:result['memory']=memory
    return result


def install(model):
    for core in model.modules():
        if isinstance(core,GTrXL) and not hasattr(core,'_reference_forward'):
            if any(p.requires_grad for p in core.parameters()):raise RuntimeError('K/V inference cache requires frozen encoder weights')
            core._reference_forward=core.forward;core._kv_hits=core._kv_misses=0;clear(core);core.forward=MethodType(step_forward,core)
    return model
