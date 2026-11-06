from pathlib import Path
import json,torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
_module=None
def module():
    global _module
    if _module is None:
        _module=load(name='afsei_node_staged_gpb',sources=[str(HERE/'staged.cpp'),str(HERE/'staged.cu')],
            extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false','-lineinfo'],verbose=False)
    return _module

class NodeSDIRK:
    """Persistent SoA workspace; each node retains its own convergence flag.

    Graphs reference owned buffers, never a caller's tensors. Returned states
    own their memory so a later step/rejected trial cannot overwrite them.
    """
    def __init__(self,n,*,native,dt_ms,substeps,device='cuda',advance_land=False):
        self.n=n;self.native=native;self.dt=dt_ms;self.steps=substeps;self.land_enabled=advance_land
        self.mod=module();self.ns=54 if native else 57
        meta=json.loads((HERE/'staged_metadata.json').read_text());c=meta['native_cache_fields' if native else 'production_cache_fields']
        opts=dict(device=device,dtype=torch.float64)
        self.input=torch.empty((self.ns,n) if native else (n,self.ns),**opts)
        self.state=torch.empty((self.ns,n),**opts);self.k1=torch.empty_like(self.state)
        self.k2=torch.empty_like(self.state);self.temp=torch.empty_like(self.state)
        self.cache=torch.empty((c,n),**opts);self.V=torch.empty(n,**opts)
        self.active=torch.empty(n,device=device,dtype=torch.bool)
        self.land_input=torch.zeros((n,4),**opts);self.land=torch.empty_like(self.land_input)
        self.stretch=torch.ones(n,**opts);self.zs=torch.zeros(n,**opts);self.zw=torch.zeros(n,**opts)
        self.current=torch.empty(n,**opts);self.guards=torch.empty(n,device=device,dtype=torch.int64)
        self.graph=None
    def _run(self):
        self.state.copy_(self.input if self.native else self.input.T)
        self.land.copy_(self.land_input);self.current.zero_();self.guards.zero_()
        self.mod.advance(self.state,self.V,self.cache,self.k1,self.k2,self.temp,self.active,
            self.land,self.stretch,self.zs,self.zw,self.current,self.guards,0.,self.dt,self.steps,self.land_enabled,self.native)
        self.current.div_(self.steps)
    @torch.no_grad()
    def advance(self,y,V,land=None,stretch=None,zs=None,zw=None):
        expected=(self.ns,self.n) if self.native else (self.n,self.ns)
        if tuple(y.shape)!=expected or y.dtype!=torch.float64 or y.device!=self.input.device or V.shape!=(self.n,) or V.device!=y.device or V.dtype!=y.dtype:
            raise ValueError('matching CUDA float64 nodal state/voltage required')
        self.input.copy_(y);self.V.copy_(V)
        if land is not None:self.land_input.copy_(land)
        for dst,src in [(self.stretch,stretch),(self.zs,zs),(self.zw,zw)]:
            if src is not None:dst.copy_(src)
        if self.graph is None:
            stream=torch.cuda.Stream(device=y.device);stream.wait_stream(torch.cuda.current_stream(y.device))
            with torch.cuda.stream(stream):self._run()
            torch.cuda.current_stream(y.device).wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self._run()
        self.graph.replay()
        state=self.state.clone() if self.native else self.state.T.contiguous()
        return state,self.land.clone(),self.current.clone(),self.guards.clone()
