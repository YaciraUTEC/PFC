"""Explicit persistent state and full contiguous-sequence training.
SSMs are portable mathematical references, not fused-kernel speed benchmarks.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..models import Mamba1,Mamba2,Mamba3,RMSNorm,rotary,decay_mask
from ..scan import affine_scan

ARCHS=['lstm','mamba','mamba2','mamba3','gtrxl','ltc']
def tree_map(fn,x):
    if x is None:return None
    if isinstance(x,(list,tuple)):return type(x)(tree_map(fn,v) for v in x)
    return fn(x)
def detach(x):return tree_map(lambda t:t.detach(),x)
def select(x,ids):return tree_map(lambda t:t[ids],x)
def concatenate(xs):
    if isinstance(xs[0],(list,tuple)):return type(xs[0])(concatenate([x[j] for x in xs]) for j in range(len(xs[0])))
    return torch.cat(xs,0)

def configurations(arch):
    common=dict(arch=arch,width=64,layers=1,state=32,head=16,memory=64,ode=3,gate_bias=2.,lr=1e-4)
    a=common.copy();b=dict(common,width=96)
    if arch=='lstm':b.update(layers=2,lr=7e-5)
    elif arch=='mamba':a['state']=16;b.update(state=32,lr=7e-5)
    elif arch in ['mamba2','mamba3']:b.update(state=64,head=32,lr=7e-5)
    elif arch=='gtrxl':a.update(layers=2);b.update(layers=2,memory=128,gate_bias=1.,lr=5e-5)
    elif arch=='ltc':b.update(ode=6,width=64,lr=7e-5)
    return [a,b]

class StreamingCore(nn.Module):
    def __init__(self,dim,cfg):
        super().__init__();self.cfg=cfg;self.arch=cfg['arch'];w=cfg['width'];layers=cfg['layers']
        self.input=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,w),nn.Tanh())
        self.norm=nn.LayerNorm(w)
        if self.arch=='lstm':
            self.core=nn.LSTM(w,w,layers,batch_first=True)
            for name,p in self.core.named_parameters():
                if 'weight_hh' in name:nn.init.orthogonal_(p)
                if 'bias' in name:
                    nn.init.zeros_(p)
                    if 'bias_ih' in name:
                        with torch.no_grad():p[w:2*w].fill_(1.)
        elif self.arch=='ltc':
            from ncps.torch import LTC
            self.core=LTC(w,w,batch_first=True,ode_unfolds=cfg['ode'],mixed_memory=False,implicit_param_constraints=True)
        elif self.arch=='gtrxl':
            from ..vendor_gtrxl import GTrXL
            self.core=GTrXL(w,head_dim=w//4,embedding_dim=w,head_num=4,layer_num=layers,
                memory_len=cfg['memory'],dropout_ratio=0,gru_bias=cfg['gate_bias'],use_embedding_layer=False)
        else:
            cls={'mamba':Mamba1,'mamba2':Mamba2,'mamba3':Mamba3}[self.arch]
            self.blocks=nn.ModuleList([cls(w,cfg['state']) if self.arch=='mamba' else cls(w,cfg['state'],cfg['head']) for _ in range(layers)])
            self.norms=nn.ModuleList([RMSNorm(w) for _ in range(layers)])
    def initial(self,batch,device=None):
        p=next(self.parameters());device=p.device if device is None else device
        z=lambda *shape:torch.zeros(batch,*shape,device=device,dtype=p.dtype)
        c=self.cfg;w=c['width']
        if self.arch=='lstm':return (z(c['layers'],w),z(c['layers'],w))
        if self.arch=='ltc':return z(w)
        if self.arch=='gtrxl':return z(c['layers']+1,c['memory'],w)
        states=[]
        for b in self.blocks:
            if self.arch=='mamba':states.append((z(b.inner,3),z(b.inner,b.n)))
            elif self.arch=='mamba2':states.append((z(b.inner+2*b.n,3),z(b.h,b.p,b.n)))
            else:states.append((z(b.h,b.angles),z(b.h,b.p,b.n),z(b.h,b.n),z(b.h,b.p)))
        return states
    def forward(self,obs,state=None):
        # All rows in a training sample are one episode; padding is loss-masked.
        y=self.input(obs);B,T,_=y.shape
        state=self.initial(B) if state is None else state
        if self.arch=='lstm':
            y,(h,c)=self.core(y,(state[0].transpose(0,1).contiguous(),state[1].transpose(0,1).contiguous()))
            new=(h.transpose(0,1),c.transpose(0,1))
        elif self.arch=='ltc':
            outputs=[];new=state
            for t in range(T):
                out,new=self.core.rnn_cell(y[:,t],new,1/48)
                outputs.append(out)
            y=torch.stack(outputs,1)
        elif self.arch=='gtrxl':
            self.core.reset_memory(state=state.permute(1,2,0,3))
            rows=torch.arange(T,device=y.device)[:,None]
            cols=torch.arange(self.cfg['memory']+T,device=y.device)[None,:]
            self.core.att_mask[T]=((cols>self.cfg['memory']+rows)|(cols<rows)).unsqueeze(-1)
            y=self.core(y,batch_first=True)['logit']
            new=self.core.get_memory().permute(2,0,1,3)
        else:
            new=[]
            for block,norm,st in zip(self.blocks,self.norms,state):
                out,nst=self._ssm(block,y,st);new.append(nst);y=norm(y+out)
        return self.norm(y),new
    def _ssm(self,b,u,st):
        B,T,_=u.shape
        if self.arch=='mamba':
            cache,ssm=st;x,z=b.in_proj(u).chunk(2,-1)
            raw=x.transpose(1,2);full=torch.cat([cache,raw],-1)
            x=F.silu(F.conv1d(full,b.conv.weight,b.conv.bias,groups=b.inner).transpose(1,2))
            dt,K,Q=torch.split(b.x_proj(x),[b.rank,b.n,b.n],-1)
            dt=F.softplus(b.dt_proj(dt));a=torch.exp(dt[...,None]*(-b.A_log.exp()))
            v=dt[...,None]*x[...,None]*K[:,:,None,:]
            h=affine_scan(a,v)+a.cumprod(1)*ssm[:,None]
            out=(h*Q[:,:,None,:]).sum(-1)+x*b.D
            return b.out_proj(out*F.silu(z)),(full[:,:,-3:],h[:,-1])
        if self.arch=='mamba2':
            cache,ssm=st
            z,raw,dt=torch.split(b.in_proj(u),[b.inner,b.inner+2*b.n,b.h],-1)
            full=torch.cat([cache,raw.transpose(1,2)],-1)
            xbc=F.silu(F.conv1d(full,b.conv.weight,b.conv.bias,groups=b.inner+2*b.n).transpose(1,2))
            x,K,Q=torch.split(xbc,[b.inner,b.n,b.n],-1);x=x.reshape(B,T,b.h,b.p)
            dt=F.softplus(dt+b.dt_bias);a=torch.exp(-b.A_log.exp()*dt)
            add=dt[:,:,:,None,None]*x[...,None]*K[:,:,None,None,:]
            aa=a[:,:,:,None,None].expand_as(add)
            h=affine_scan(aa,add)+a.cumprod(1)[:,:,:,None,None]*ssm[:,None]
            out=(h*Q[:,:,None,None,:]).sum(-1)+x*b.D[None,None,:,None]
            return b.out_proj(b.norm(out.flatten(-2)*F.silu(z))),(full[:,:,-3:],h[:,-1])
        phase,ssm,prevk,prevv=st
        z,x,K,Q,ddt,ddA,trap,ang=torch.split(b.in_proj(u),[b.inner,b.inner,b.n,b.n,b.h,b.h,b.h,b.angles],-1)
        dt=F.softplus(ddt+b.dt_bias)
        A=-(ddA.clamp_min(0)+1/(1-ddA.clamp_max(0))).clamp_min(1e-4)
        a=(A*dt).exp();lam=trap.sigmoid()
        phases=phase[:,None]+(ang.tanh()[:,:,None,:]*math.pi*dt[...,None]).cumsum(1)
        K=rotary(b.B_norm(K)[:,:,None]+b.B_bias[None,None],phases)
        Q=rotary(b.C_norm(Q)[:,:,None]+b.C_bias[None,None],phases)
        x=x.reshape(B,T,b.h,b.p)
        kv=x[...,None]*K[:,:,:,None,:]
        previous=torch.cat([(prevv[...,None]*prevk[:,:,None,:])[:,None],kv[:,:-1]],1)
        add=dt[:,:,:,None,None]*(lam[:,:,:,None,None]*kv+
            a[:,:,:,None,None]*(1-lam[:,:,:,None,None])*previous)
        aa=a[:,:,:,None,None].expand_as(add)
        h=affine_scan(aa,add)+a.cumprod(1)[:,:,:,None,None]*ssm[:,None]
        out=(h*Q[:,:,:,None,:]).sum(-1)+x*b.D[None,None,:,None]
        return b.out_proj((out*F.silu(z.reshape(B,T,b.h,b.p))).flatten(-2)),(phases[:,-1],h[:,-1],K[:,-1],x[:,-1])

class Agent(nn.Module):
    def __init__(self,cfg,dim=64):
        super().__init__();self.cfg=cfg;self.dim=dim
        self.encoder=StreamingCore(dim,cfg);w=cfg['width']
        self.actor=nn.Sequential(nn.Linear(w,64),nn.Tanh(),nn.Linear(64,5))
        # Critic is deliberately separate from the policy/memory representation.
        self.critic=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.log_std=nn.Parameter(torch.full((5,),-2.8))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():self.actor[-1].bias[-1]=-4.
    def forward(self,obs,state=None):
        z,st=self.encoder(obs,state)
        return self.actor(z),self.critic(obs).squeeze(-1),st
    def act(self,obs,state=None,deterministic=False):
        mu,v,st=self(obs[:,None],state);mu=mu[:,0];v=v[:,0]
        dist=torch.distributions.Normal(mu,self.log_std.exp())
        raw=mu if deterministic else dist.sample()
        a=raw.tanh();lp=(dist.log_prob(raw)-torch.log(1-a.square()+1e-6)).sum(-1)
        return a,lp,v,st
    def predict_actions(self,obs,state=None):
        z,st=self.encoder(obs[:,None],state)
        return self.actor(z[:,0]).tanh(),st
    def optimizer(self,lr):
        decay=[];nodecay=[]
        for n,p in self.named_parameters():
            if not p.requires_grad:continue
            (nodecay if p.ndim<2 or any(k in n for k in ['A_log','dt_bias','dt_proj.bias','_params','norm']) else decay).append(p)
        return torch.optim.AdamW([dict(params=decay,weight_decay=1e-4),dict(params=nodecay,weight_decay=0)],lr=lr)
    def save(self,path,**metadata):
        torch.save(dict(config=self.cfg,dim=self.dim,state=self.state_dict(),metadata=metadata),path)
    @classmethod
    def load(cls,path,device='cuda'):
        d=torch.load(path,map_location=device,weights_only=False)
        m=cls(d['config'],d['dim']).to(device);m.load_state_dict(d['state']);m.eval();return m
