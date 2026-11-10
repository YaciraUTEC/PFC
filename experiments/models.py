"""Portable PyTorch reference blocks. Same finite history for every architecture.
Mamba/2: state-spaces/mamba modules; Mamba-3: official SISO recurrence.
GTrXL: vendored DI-engine implementation; LTC: original ncps.torch.LTC.
No CUDA-kernel speed comparison is claimed for these reference backends.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class RMSNorm(nn.Module):
    def __init__(self,n):super().__init__();self.weight=nn.Parameter(torch.ones(n))
    def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*self.weight

def decay_mask(a):
    # a=(batch,heads,length), exp(sum_{s+1..t} a), only causal entries.
    cs=a.cumsum(-1);diff=cs.unsqueeze(-1)-cs.unsqueeze(-2)
    mask=torch.ones(diff.shape[-2:],device=a.device,dtype=torch.bool).tril()
    return torch.exp(diff.masked_fill(~mask,-torch.inf))

class Mamba1(nn.Module):
    def __init__(self,d,state=16):
        super().__init__();self.inner=2*d;self.n=state;self.rank=math.ceil(d/16)
        self.in_proj=nn.Linear(d,4*d,bias=False)
        self.conv=nn.Conv1d(2*d,2*d,4,padding=3,groups=2*d)
        self.x_proj=nn.Linear(2*d,self.rank+2*state,bias=False)
        self.dt_proj=nn.Linear(self.rank,2*d)
        nn.init.uniform_(self.dt_proj.weight,-self.rank**-.5,self.rank**-.5)
        dt=torch.exp(torch.rand(2*d)*(math.log(.1)-math.log(.001))+math.log(.001))
        with torch.no_grad():self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.arange(1,state+1).float().log().repeat(2*d,1))
        self.D=nn.Parameter(torch.ones(2*d));self.out_proj=nn.Linear(2*d,d,bias=False)
    def forward(self,u):
        x,z=self.in_proj(u).chunk(2,-1)
        x=F.silu(self.conv(x.transpose(1,2))[...,:u.shape[1]].transpose(1,2))
        dt,B,C=torch.split(self.x_proj(x),[self.rank,self.n,self.n],-1)
        dt=F.softplus(self.dt_proj(dt));A=-self.A_log.exp()
        da=torch.exp(dt.unsqueeze(-1)*A)
        bx=dt.unsqueeze(-1)*x.unsqueeze(-1)*B.unsqueeze(-2)
        from .scan import affine_scan
        states=affine_scan(da,bx)
        y=(states*C.unsqueeze(-2)).sum(-1)+x*self.D
        return self.out_proj(y*F.silu(z))

class Mamba2(nn.Module):
    def __init__(self,d,state=32,headdim=32):
        super().__init__();self.inner=2*d;self.n=state;self.p=headdim;self.h=self.inner//headdim
        self.in_proj=nn.Linear(d,2*self.inner+2*state+self.h,bias=False)
        c=self.inner+2*state;self.conv=nn.Conv1d(c,c,4,padding=3,groups=c)
        dt=torch.exp(torch.rand(self.h)*(math.log(.1)-math.log(.001))+math.log(.001))
        self.dt_bias=nn.Parameter(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.empty(self.h).uniform_(1,16).log())
        self.D=nn.Parameter(torch.ones(self.h));self.norm=RMSNorm(self.inner)
        self.out_proj=nn.Linear(self.inner,d,bias=False)
    def forward(self,u):
        z,xbc,dt=torch.split(self.in_proj(u),[self.inner,self.inner+2*self.n,self.h],-1)
        dt=F.softplus(dt+self.dt_bias)
        xbc=F.silu(self.conv(xbc.transpose(1,2))[...,:u.shape[1]].transpose(1,2))
        x,B,C=torch.split(xbc,[self.inner,self.n,self.n],-1)
        x=x.reshape(*x.shape[:2],self.h,self.p)
        a=(-self.A_log.exp()*dt).transpose(1,2)
        qk=torch.einsum('btn,bsn->bts',C,B).unsqueeze(1)
        att=qk*decay_mask(a)*dt.transpose(1,2).unsqueeze(-2)
        y=torch.einsum('bhts,bshp->bthp',att,x)+x*self.D[None,None,:,None]
        return self.out_proj(self.norm(y.flatten(-2)*F.silu(z)))

def rotary(x,angles):
    pairs=x.reshape(*x.shape[:-1],-1,2)
    cs,ss=angles.cos(),angles.sin()
    pad=pairs.shape[-2]-angles.shape[-1]
    cs=F.pad(cs,(0,pad),value=1);ss=F.pad(ss,(0,pad))
    a,b=pairs.unbind(-1)
    return torch.stack((a*cs-b*ss,a*ss+b*cs),-1).flatten(-2)

def mamba3_scan(Q,K,V,ADT,DT,Trap,Q_bias,K_bias,Angles,D=None,Z=None):
    # Vectorized exact finite-context form of the official SISO step recurrence.
    h=V.shape[2]
    Q=Q.expand(-1,-1,h,-1)+Q_bias[None,None]
    K=K.expand(-1,-1,h,-1)+K_bias[None,None]
    phase=(torch.tanh(Angles)*math.pi*DT.transpose(1,2).unsqueeze(-1)).cumsum(1)
    Q,K=rotary(Q,phase),rotary(K,phase)
    trap=torch.sigmoid(Trap)
    shifted=F.pad((DT*(1-trap))[:,:,1:],(0,1))
    scale=DT*trap+shifted
    att=torch.einsum('bthn,bshn->bhts',Q,K*scale.transpose(1,2).unsqueeze(-1))*decay_mask(ADT)
    y=torch.einsum('bhts,bshp->bthp',att,V)
    y=y-V*(Q*K).sum(-1).mul(shifted.transpose(1,2)).unsqueeze(-1)
    if D is not None:y=y+D[None,None,:,None]*V
    if Z is not None:y=y*F.silu(Z)
    return y

class Mamba3(nn.Module):
    def __init__(self,d,state=32,headdim=32):
        super().__init__();self.inner=2*d;self.n=state;self.p=headdim;self.h=self.inner//headdim
        self.angles=state//4
        self.in_proj=nn.Linear(d,2*self.inner+2*state+3*self.h+self.angles,bias=False)
        dt=torch.exp(torch.rand(self.h)*(math.log(.1)-math.log(.001))+math.log(.001))
        self.dt_bias=nn.Parameter(dt+torch.log(-torch.expm1(-dt)))
        self.B_bias=nn.Parameter(torch.ones(self.h,state));self.C_bias=nn.Parameter(torch.ones(self.h,state))
        self.B_norm=RMSNorm(state);self.C_norm=RMSNorm(state)
        self.D=nn.Parameter(torch.ones(self.h));self.out_proj=nn.Linear(self.inner,d,bias=False)
    def forward(self,u):
        z,x,B,C,dd_dt,dd_A,trap,angles=torch.split(self.in_proj(u),
          [self.inner,self.inner,self.n,self.n,self.h,self.h,self.h,self.angles],-1)
        dt=F.softplus(dd_dt+self.dt_bias)
        A=-(dd_A.clamp_min(0)+1/(1-dd_A.clamp_max(0))).clamp_min(1e-4)
        shape=(*u.shape[:2],self.h,self.p)
        y=mamba3_scan(self.C_norm(C).unsqueeze(2),self.B_norm(B).unsqueeze(2),x.reshape(shape),
          (A*dt).transpose(1,2),dt.transpose(1,2),trap.transpose(1,2),
          self.C_bias,self.B_bias,angles.unsqueeze(2).expand(-1,-1,self.h,-1),
          self.D,z.reshape(shape))
        return self.out_proj(y.flatten(-2))

class TemporalFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,architecture='lstm',width=96,layers=2,features_dim=64):
        super().__init__(observation_space,features_dim)
        self.architecture=architecture;self.width=width;self._ltc_compiled=False
        inp=observation_space.shape[-1]
        if architecture=='lstm':
            self.core=nn.LSTM(inp,width,layers,batch_first=True)
        elif architecture=='ltc':
            from ncps.torch import LTC
            self.core=LTC(inp,width,batch_first=True,ode_unfolds=3,mixed_memory=False)
        elif architecture=='gtrxl':
            from .vendor_gtrxl import GTrXL
            self.core=GTrXL(inp,head_dim=width//4,embedding_dim=width,head_num=4,
                     layer_num=layers,memory_len=observation_space.shape[0]//2,dropout_ratio=0)
        else:
            self.embed=nn.Linear(inp,width)
            cls={'mamba':Mamba1,'mamba2':Mamba2,'mamba3':Mamba3}[architecture]
            self.blocks=nn.ModuleList([cls(width) for _ in range(layers)])
            self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.out=nn.Sequential(nn.Linear(width,features_dim),nn.Tanh())
    def forward(self,obs):
        if self.architecture in ('lstm','ltc'):
            if self.architecture=='ltc' and obs.is_cuda and not self._ltc_compiled:
                from .compile_support import prepare_compiler
                prepare_compiler()
                self.core.rnn_cell.forward=torch.compile(self.core.rnn_cell.forward,
                   fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
                self._ltc_compiled=True
            y,_=self.core(obs)
        elif self.architecture=='gtrxl':
            # Explicit window-owned memory, never shared across unrelated PPO samples.
            self.core.reset_memory(batch_size=obs.shape[0])
            half=obs.shape[1]//2
            self.core(obs[:,:half],batch_first=True)
            y=self.core(obs[:,half:],batch_first=True)['logit']
        else:
            y=self.embed(obs)
            for block,norm in zip(self.blocks,self.norms):y=norm(y+block(y))
        return self.out(y[:,-1])
