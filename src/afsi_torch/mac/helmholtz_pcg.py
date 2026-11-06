"""Optional linear solver for the unchanged sealed BE viscosity matrix.

Normal components at exterior MAC faces carry half-volume weights. Their
Neumann Laplacian is self-adjoint in those weights, not in the Euclidean norm.
The square-root transform makes PCG legitimate. Fixed masked faces are removed.
One block-diagonal solve handles all three components, normalized by their
original individual residual tolerances. Pressure and IB are untouched.
"""
from math import prod
import torch
from .open_boundary import velocity_laplacian
from .execution import tensor_kernel
from ..fluid.solvers import pcg,SolverOptions


class BEViscousPCG:
    def __init__(self,grid,dt,rho,mu,masks,diagonals,*,fused=False,max_iterations=2000,check_every=16):
        self.grid,self.alpha,self.masks=grid,dt*mu/rho,masks
        self.sizes=[prod(grid.face_shape(c)) for c in range(3)]
        self.weights=[]
        for c,m in enumerate(masks):
            w=torch.ones_like(m);w.select(c,0).fill_(.5);w.select(c,-1).fill_(.5)
            self.weights.append(w.sqrt())
        self.fixed=torch.cat([(m==0).reshape(-1) for m in masks])
        self.diagonal=torch.cat([d.reshape(-1) for d in diagonals])
        self.options=SolverOptions(rtol=0.,atol=.5,max_iterations=max_iterations,
            check_every=check_every,recompute_every=200)
        self.action=tensor_kernel(self._action,masks[0].device) if fused else self._action
        self.iterations=0

    def unpack(self,v):
        return tuple(part.reshape(self.grid.face_shape(c)) for c,part in enumerate(v.split(self.sizes)))

    def _action(self,y):
        result=[]
        for c,(part,w,m) in enumerate(zip(self.unpack(y),self.weights,self.masks)):
            x=m*part/w
            result.append((m*w*(x-self.alpha*velocity_laplacian(x,c,self.grid.spacing))).reshape(-1))
        return torch.cat(result)

    @torch.no_grad()
    def solve(self,rhs,targets):
        # H is block diagonal, so a scalar scale per component commutes with H.
        # ||scaled residual|| <= .5 guarantees EACH physical component's norm
        # <= sqrt(2)*.5*target, leaving room for roundoff in the original check.
        scales=[t if t>0 else 1. for t in targets]
        b=torch.cat([(m*w*v/s).reshape(-1) for m,w,v,s in zip(self.masks,self.weights,rhs,scales)])
        y,info=pcg(self.action,b,self.diagonal,fixed=self.fixed,options=self.options)
        self.iterations+=info.iterations
        u=tuple(m*part*s/w for part,m,w,s in zip(self.unpack(y),self.masks,self.weights,scales))
        return u,info
