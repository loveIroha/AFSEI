"""GPU sine-transform solve for the ORIGINAL cell-centred MAC pressure matrix.

Odd reflection at physical box faces gives a DST-II in each Cartesian axis.
This is the same -DG as open_boundary.negative_laplacian, not a periodic FFT
approximation. Every solve checks the original true residual and tolerances.
"""
from math import isfinite
import torch
from .multigrid import MGOptions
from .open_boundary import negative_laplacian
from .execution import tensor_kernel


class SpectralDirichletPressure:
    def __init__(self,grid,*,device='cpu',dtype=torch.float64,options=None,workspace=True):
        if dtype != torch.float64: raise ValueError('float64 spectral pressure required')
        self.grid = grid
        self.options = options or MGOptions()
        self.backend = 'torch-dst2-dirichlet'
        self.shapes,self.spacings = [grid.shape],[grid.spacing]
        like = torch.empty((),device=device,dtype=dtype)
        self.rhs = like.new_empty(grid.shape)
        self.direct_solves = 0
        self.phases = []
        self.eigenvalue = like.new_zeros(grid.shape)
        for axis,(n,h) in enumerate(zip(grid.shape,grid.spacing)):
            k = torch.arange(1,n+1,device=device,dtype=dtype)
            shape = [1,1,1]; shape[axis] = n
            theta = torch.pi*k/(2*n)
            self.phases.append((theta.sin().reshape(shape),theta.cos().reshape(shape)))
            self.eigenvalue += (4*theta.sin().square()/h**2).reshape(shape)
        self._measure = tensor_kernel(lambda rhs,p: torch.stack((
            torch.linalg.vector_norm(rhs),torch.linalg.vector_norm(rhs-negative_laplacian(p,grid.spacing)))),device)
        from .spectral_workspace import SpectralWorkspace
        self.workspace=SpectralWorkspace(self) if workspace else None

    def forward(self,x,axis):
        n = x.shape[axis]
        extended = torch.cat((x,-x.flip((axis,))),axis)
        f = torch.fft.rfft(extended,dim=axis).narrow(axis,1,n)
        sine,cosine = self.phases[axis]
        return f.real*sine-f.imag*cosine

    def inverse(self,x,axis):
        n = x.shape[axis]
        sine,cosine = self.phases[axis]
        f = torch.complex(x*sine,-x*cosine)
        shape = list(f.shape); shape[axis] = 1
        f = torch.cat((f.new_zeros(shape),f),axis)
        return torch.fft.irfft(f,n=2*n,dim=axis).narrow(axis,0,n)

    def inverse_operator(self,rhs):
        if self.workspace is not None:
            return self.workspace.inverse(rhs).clone()
        value = rhs
        for axis in (2,1,0): value = self.forward(value,axis)
        value = value/self.eigenvalue
        for axis in (0,1,2): value = self.inverse(value,axis)
        return value.contiguous()

    @torch.no_grad()
    def solve(self,rhs,initial=None):
        if rhs.shape != self.rhs.shape or rhs.dtype != self.rhs.dtype or rhs.device != self.rhs.device:
            raise ValueError('spectral pressure RHS shape/dtype/device mismatch')
        if initial is not None and (initial.shape != rhs.shape or initial.dtype != rhs.dtype or initial.device != rhs.device):
            raise ValueError('invalid initial spectral pressure')
        if initial is not None and not torch.isfinite(initial).all():
            raise ValueError('nonfinite initial spectral pressure')
        p = self.inverse_operator(rhs)
        norm,residual = self._measure(rhs,p).tolist()
        if not isfinite(norm) or not isfinite(residual): raise ValueError('nonfinite spectral pressure input/output')
        tolerance = max(self.options.atol,self.options.rtol*norm)
        refinements = 0
        # Roundoff correction if an unusually strict configured target demands it.
        while residual > tolerance and refinements < 2:
            p += self.inverse_operator(rhs-negative_laplacian(p,self.grid.spacing))
            _,residual = self._measure(rhs,p).tolist()
            refinements += 1
        if not isfinite(residual) or residual > tolerance:
            raise RuntimeError(f'spectral pressure true residual failed: {residual:g} > {tolerance:g}')
        self.direct_solves += 1+refinements
        return p,dict(cycles=0,residual_norm=residual,tolerance=tolerance,backend=self.backend,
                      direct_solves=1+refinements)
