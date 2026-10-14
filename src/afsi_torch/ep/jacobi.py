"""EP-only Jacobi-PCG: insulating CSR system, no Dirichlet lift or AMG.

The scripted loop implements textbook PCG independently of TorchCor. A true
residual gates acceptance; recurrence replacement restarts PCG when necessary.
"""
import math
import torch
from ..fluid.solvers import SolveInfo


@torch.jit.script
def _solve(A: torch.Tensor, b: torch.Tensor, diagonal: torch.Tensor,
           initial: torch.Tensor, rtol: float, atol: float,
           max_iterations: int, recompute_every: int):
    x=initial.clone()
    rhs_norm=float(torch.linalg.vector_norm(b))
    if not math.isfinite(rhs_norm):
        raise RuntimeError('nonfinite EP Jacobi-PCG right-hand side norm')
    tolerance=max(atol,rtol*rhs_norm)
    r=b-torch.sparse.mm(A,x[:,None]).flatten()
    residual_norm=float(torch.linalg.vector_norm(r))
    if not math.isfinite(residual_norm):
        raise RuntimeError('nonfinite initial EP Jacobi-PCG residual')
    if residual_norm<=tolerance:
        return x,0,residual_norm,rhs_norm,tolerance
    z=r/diagonal
    direction=z.clone()
    rz=torch.dot(r,z)
    for iteration in range(1,max_iterations+1):
        Ad=torch.sparse.mm(A,direction[:,None]).flatten()
        curvature=torch.dot(direction,Ad)
        if not bool(torch.isfinite(curvature)&(curvature>0)&torch.isfinite(rz)&(rz>0)):
            raise RuntimeError('EP Jacobi-PCG lost positive finite curvature')
        alpha=rz/curvature
        x=x+alpha*direction
        r=r-alpha*Ad
        residual_norm=float(torch.linalg.vector_norm(r))
        if not math.isfinite(residual_norm):
            raise RuntimeError('nonfinite EP Jacobi-PCG residual')
        restart=iteration%recompute_every==0
        if residual_norm<=tolerance or restart or iteration==max_iterations:
            r=b-torch.sparse.mm(A,x[:,None]).flatten()
            residual_norm=float(torch.linalg.vector_norm(r))
            if not math.isfinite(residual_norm):
                raise RuntimeError('nonfinite true EP Jacobi-PCG residual')
            if residual_norm<=tolerance:
                return x,iteration,residual_norm,rhs_norm,tolerance
            restart=True
        z=r/diagonal
        next_rz=torch.dot(r,z)
        direction=z if restart else z+(next_rz/rz)*direction
        rz=next_rz
    raise RuntimeError('EP Jacobi-PCG iteration limit reached')


class JacobiPCG:
    def __init__(self,matrix,diagonal,options):
        if (matrix.layout!=torch.sparse_csr or matrix.shape[0]!=matrix.shape[1]
                or diagonal.shape!=(matrix.shape[0],) or diagonal.device!=matrix.device
                or diagonal.dtype!=matrix.dtype or matrix.dtype not in (torch.float32,torch.float64)):
            raise ValueError('matching floating CSR matrix and Jacobi diagonal required')
        if not torch.isfinite(matrix.values()).all() or not torch.isfinite(diagonal).all() or (diagonal<=0).any():
            raise ValueError('finite matrix and positive Jacobi diagonal required')
        self.matrix,self.diagonal,self.options=matrix,diagonal,options

    @torch.no_grad()
    def solve(self,rhs,*,initial):
        for value in (rhs,initial):
            if value.shape!=self.diagonal.shape or value.dtype!=self.diagonal.dtype or value.device!=self.diagonal.device:
                raise ValueError('EP PCG vectors must match the cached matrix')
        x,iterations,residual,rhs_norm,tolerance=_solve(self.matrix,rhs,self.diagonal,initial,
            self.options.rtol,self.options.atol,self.options.max_iterations,self.options.recompute_every)
        if not math.isfinite(residual) or residual>tolerance:
            raise RuntimeError('EP PCG true residual acceptance failed')
        return x,SolveInfo(iterations,residual,rhs_norm,tolerance)

    def close(self):
        pass
