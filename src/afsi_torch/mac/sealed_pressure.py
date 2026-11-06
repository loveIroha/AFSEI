"""MAC pressure with stationary impermeable face boundaries.

Only boundary fluxes change. D is the original finite-volume divergence,
G = face_mask * G_open. Consequently A=-DG is SPD, with half-cell physical
Dirichlet faces and zero gradient/flux at closed faces. No cell is removed.
Piecewise-constant aggregation uses an exact Galerkin hierarchy R A P;
the symmetric V cycle preconditions the existing GPU PCG implementation.
"""
from math import prod, isfinite
from dataclasses import replace
import torch
import torch.nn.functional as F
from .execution import tensor_kernel
from ..fluid.solvers import pcg, SolverOptions


def tightened_pressure_options(options,tolerances):
    """Optional stricter inner solve; preserve the operator and original options."""
    if tolerances is None:return options
    if not isinstance(tolerances,dict) or set(tolerances)!={'rtol','atol'}:
        raise ValueError('pressure tolerances require rtol and atol')
    for key,value in tolerances.items():
        if (type(value) not in (int,float) or not isfinite(value) or value<0
                or value>getattr(options,key)):
            raise ValueError('pressure tolerances can only tighten finite inherited tolerances')
    if tolerances['rtol']+tolerances['atol']<=0:
        raise ValueError('at least one positive pressure tolerance required')
    return replace(options,**tolerances)


def slab(axis, lo=None, hi=None):
    return tuple(slice(lo,hi) if a == axis else slice(None) for a in range(3))


def divergence(u, spacing):
    return sum(torch.diff(v,dim=a)/h for a,(v,h) in enumerate(zip(u,spacing)))


def gradient(p, spacing, masks):
    g=[]
    for a,(h,m) in enumerate(zip(spacing,masks)):
        low=p.narrow(a,0,1); high=p.narrow(a,p.shape[a]-1,1)
        g.append(m*torch.cat((2*low/h,torch.diff(p,dim=a)/h,-2*high/h),dim=a))
    return tuple(g)


def action(p,spacing,masks):
    return -divergence(gradient(p,spacing,masks),spacing)


def diagonal(shape,spacing,masks):
    result=masks[0].new_zeros(shape)
    for a,(h,m) in enumerate(zip(spacing,masks)):
        result += (m.narrow(a,0,shape[a])+m.narrow(a,1,shape[a]))/h**2
        result.narrow(a,0,1).add_(m.narrow(a,0,1)/h**2)
        result.narrow(a,shape[a]-1,1).add_(m.narrow(a,shape[a],1)/h**2)
    return result


def coarsen(masks):
    result=[]
    for normal,m in enumerate(masks):
        ids=torch.arange(0,m.shape[normal],2,device=m.device)
        coarse=m.index_select(normal,ids)
        for a in range(m.ndim):
            if a != normal:
                coarse=coarse.unfold(a,2,2).mean(-1)
        # Constant P and volume-average R: coefficient 2*mean is R A P,
        # including physical boundary half-cell terms, not rediscretization.
        result.append(2*coarse)
    return tuple(result)


class SealedPressure:
    def __init__(self, grid, masks, plane_indices, *, options, fused=False,
                 graphs=False, max_iterations=200, compact_geometry=False):
        if type(compact_geometry) is not bool:raise ValueError('compact_geometry must be boolean')
        self.options=options;self.grid=grid
        self.shapes=[];self.spacings=[];self.masks=[];self.diagonals=[]
        shape=tuple(grid.shape);spacing=tuple(grid.spacing);planes=tuple(plane_indices)
        while True:
            self.shapes.append(shape);self.spacings.append(spacing);self.masks.append(masks)
            self.diagonals.append(diagonal(shape,spacing,masks))
            if min(shape)<=4 or any(n%2 for n in shape) or any(i%2 for i in planes): break
            shape=tuple(n//2 for n in shape);spacing=tuple(2*h for h in spacing)
            planes=tuple(i//2 for i in planes);masks=coarsen(masks)
        n=prod(shape)
        if n>2048: raise ValueError('sealed pressure coarse grid exceeds 2048 cells')
        like=masks[0];eye=torch.eye(n,device=like.device,dtype=like.dtype)
        matrix=torch.vmap(lambda p:action(p.reshape(shape),spacing,masks).reshape(-1))(eye).T
        self.inverse=torch.linalg.inv(matrix)
        self.compact_geometry=compact_geometry
        self.execution_masks=[]
        for masks_at_level in self.masks:
            compact=[]
            for value in masks_at_level:
                small=value
                if compact_geometry:
                    small=value.bool()
                    if not torch.equal(small.to(value.dtype),value):
                        small=value.float()
                        if not torch.equal(small.to(value.dtype),value):small=value
                compact.append(small)
            self.execution_masks.append(tuple(compact))
        # Diagonals/coarse inverse and every pressure/vector operation stay
        # FP64. The immutable coefficients are reduced only if exact casting
        # back to their canonical values has been verified once at setup.
        self.actions=[];self.smoothers=[]
        for h,m,d in zip(self.spacings,self.execution_masks,self.diagonals):
            op=lambda p,h=h,m=m:action(p,h,m)
            def smooth(p,b,h=h,m=m,d=d):
                for _ in range(options.smooth): p=p+(2/3)*(b-action(p,h,m))/d
                return p
            self.actions.append(tensor_kernel(op,like.device) if fused else op)
            self.smoothers.append(tensor_kernel(smooth,like.device) if fused else smooth)
        self.graphs=graphs and like.is_cuda;self.graph=None
        self.prec_rhs=like.new_zeros(grid.shape);self.prec_result=like.new_zeros(grid.shape)
        self.cg_options=SolverOptions(rtol=options.rtol,atol=options.atol,
            max_iterations=max_iterations,check_every=options.check_every,recompute_every=40)
        self.backend='MAC sealed-face Galerkin-MG-PCG'+(' / CUDA graph' if self.graphs else '')
        if compact_geometry:self.backend+=' / lossless geometry storage'
        self.cycle_count=0

    def cycle(self,level,p,b):
        if level==len(self.shapes)-1:
            return (self.inverse@b.reshape(-1)).reshape(b.shape)
        p=self.smoothers[level](p,b)
        residual=b-self.actions[level](p)
        pool=F.avg_pool2d if b.ndim==2 else F.avg_pool3d
        coarse=pool(residual[None,None],2,2)[0,0]
        correction=self.cycle(level+1,torch.zeros_like(coarse),coarse)
        for a in range(b.ndim): correction=correction.repeat_interleave(2,dim=a)
        return self.smoothers[level](p+correction,b)

    def _precondition_body(self):
        self.prec_result.copy_(self.cycle(0,torch.zeros_like(self.prec_rhs),self.prec_rhs))

    def precondition(self,b):
        self.cycle_count+=1
        if not self.graphs: return self.cycle(0,torch.zeros_like(b),b)
        if self.graph is None:
            from .graph_capture import capture_initialization
            with capture_initialization():
                self.prec_rhs.copy_(b)
                stream=torch.cuda.Stream(device=b.device)
                stream.wait_stream(torch.cuda.current_stream(b.device))
                with torch.cuda.stream(stream):
                    self._precondition_body();self._precondition_body()
                torch.cuda.current_stream(b.device).wait_stream(stream)
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=stream): self._precondition_body()
        self.prec_rhs.copy_(b);self.graph.replay()
        return self.prec_result.clone()

    @torch.no_grad()
    def solve(self,b,initial=None):
        before=self.cycle_count
        p,info=pcg(self.actions[0],b,self.diagonals[0],initial=initial,
            options=self.cg_options,preconditioner=self.precondition)
        return p,dict(cycles=self.cycle_count-before,iterations=info.iterations,
            residual_norm=info.residual_norm,tolerance=info.tolerance,backend=self.backend)
