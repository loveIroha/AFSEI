"""Original ORd 2011 for fixed-tissue comparison; native GPB/FSI stays separate."""
from dataclasses import dataclass,replace
import json
from pathlib import Path
import threading
import torch

_lock=threading.Lock()
_loaded=False


def build():
    global _loaded
    if _loaded: return
    with _lock:
        if _loaded: return
        from torch.utils.cpp_extension import CUDA_HOME,load
        from ..mac.cuda_ib import _validate_toolkit
        if not torch.cuda.is_available() or CUDA_HOME is None:
            raise RuntimeError('ORd requires CUDA PyTorch and a matching CUDA toolkit')
        _validate_toolkit(CUDA_HOME,torch.version.cuda)
        root=Path(__file__).parent/'csrc'
        load(name='afsei_ord_cuda',sources=[str(root/f) for f in ('ord.cpp','ord.cu')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],is_python_module=False)
        _loaded=True


@dataclass(frozen=True)
class ORdState:
    voltage_v: torch.Tensor
    ionic: torch.Tensor


class ORdCell:
    def __init__(self,dt_ep_s=5e-5,dt_ode_ms=.002,*,celltype=1):
        from .tissue import GPBLand
        settings=GPBLand(dt_ep_s,dt_ode_ms)  # one common substep/time validation contract
        self.dt_ep_s,self.dt_ode_ms,self.substeps=settings.dt_ep_s,settings.dt_ode_ms,settings.substeps
        self.celltype=celltype
        self.constants=None

    def initial(self,like):
        build()
        constants,state=torch.ops.afsei_ord_cuda.initialize(self.celltype)
        self.constants=constants.to(device=like.device)
        return ORdState(torch.full_like(like,state[0].item()/1000),
                        state[1:].to(device=like.device).expand(len(like),-1).clone())

    def ionic(self,state): return state.ionic
    def calcium(self,state): return state.ionic[:,8]
    def voltage(self,state,value): return replace(state,voltage_v=value)

    @torch.no_grad()
    def advance_ode(self,state,time_s):
        y,v=state.ionic,state.voltage_v
        if (not v.is_cuda or v.dtype!=torch.float64 or y.shape!=(len(v),40) or
                not torch.isfinite(v).all() or not torch.isfinite(y).all() or (y[:,1:9]<=0).any()):
            raise ValueError('finite ORd state with positive concentrations required')
        output,current,guards=torch.ops.afsei_ord_cuda.advance(y.contiguous(),(v*1000).contiguous(),
            self.constants,self.dt_ode_ms,self.substeps)
        if guards.any() or not torch.isfinite(output).all() or not torch.isfinite(current).all() or (output[:,1:9]<=0).any():
            raise FloatingPointError('ORd failed; no nonfinite replacement or silent fallback')
        return ORdState(v,output),current

    def step(self,ep,state,time_s,stimulus):
        ode,current=self.advance_ode(state,time_s)
        voltage,info=ep.step(state.voltage_v,current,stimulus,self.dt_ep_s)
        return self.voltage(ode,voltage),current,info

    def provenance(self):
        data=json.loads((Path(__file__).parent/'ord_provenance.json').read_text())
        data['celltype']='epicardial' if self.celltype==1 else 'endocardial'
        return data
