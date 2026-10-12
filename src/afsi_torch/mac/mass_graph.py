"""Experimental PCG blocks captured between the existing host check points.

CSR/Jacobi recurrence, recomputation schedule and true residual acceptance are
the same as MassSolver. Each result is cloned out of the graph-owned workspace.
CPU executes the same blocks eagerly for recurrence/ownership tests.
"""
from math import isfinite
import torch
from .mass_solver import MassSolver,_advance,_direction
from .execution import tensor_kernel
from .graph_capture import capture_initialization
from ..fluid.solvers import SolveInfo


def _advance_into(x,r,d,Ad,rz,active,broken,tol,rn):
    next_active,next_broken,next_rn=_advance(x,r,d,Ad,rz,active,broken,tol)
    active.copy_(next_active)
    broken.copy_(next_broken)
    rn.copy_(next_rn)


def _direction_into(r,diag,d,rz,active):
    rz.copy_(_direction(r,diag,d,rz,active))


def _initial_metrics(rhs,residual,initial):
    # One device reduction/host transfer instead of three serial barriers.
    finite=torch.isfinite(rhs).all() & torch.isfinite(initial).all()
    return torch.stack((finite.to(rhs.dtype),torch.linalg.vector_norm(rhs),
                        torch.linalg.vector_norm(residual)))


class GraphMassSolver(MassSolver):
    def __init__(self,mass,diagonal,options=None):
        super().__init__(mass,diagonal,options)
        # The consistent mass operator is immutable. Check it once before
        # using the exact M*0=0 shortcut, so NaN/Inf entries cannot be hidden.
        if not torch.isfinite(mass.values()).all():
            raise ValueError('finite constant mass operator required')
        if not 1<=self.options.check_every<=16 or self.options.recompute_every<1:
            raise ValueError('graph PCG requires check_every in [1,16] and positive recompute_every')
        self.rz=diagonal.new_ones(())
        self.active=torch.zeros((),device=diagonal.device,dtype=torch.bool)
        self.broken=torch.zeros_like(self.active)
        self.rn=diagonal.new_zeros(())
        # Compile scalar writeback with the vector work instead of submitting
        # separate tiny copy kernels on every iteration in the captured block.
        self.advance_into=tensor_kernel(_advance_into,diagonal.device)
        self.direction_into=tensor_kernel(_direction_into,diagonal.device)
        self.initial_metrics=tensor_kernel(_initial_metrics,diagonal.device)
        self.graphs={}
        if diagonal.is_cuda:
            self._capture()

    def _block(self,count):
        for i in range(count):
            self.action(self.d,self.Ad)
            self.advance_into(self.x,self.r,self.d,self.Ad,
                              self.rz,self.active,self.broken,self.tol,self.rn)
            if i+1<count:
                self.direction_into(self.r,self.diagonal,self.d,self.rz,self.active)

    def _capture(self):
        # Empty inactive work warms library/compiled kernels without a host
        # check in the captured region. All graph inputs have stable addresses.
        with torch.cuda.device(self.x.device), capture_initialization():
            for v in (self.x,self.r,self.d,self.Ad):
                v.zero_()
            self.tol.fill_(1.)
            stream=torch.cuda.Stream(device=self.x.device)
            stream.wait_stream(torch.cuda.current_stream(self.x.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._block(self.options.check_every)
            torch.cuda.current_stream(self.x.device).wait_stream(stream)
            for count in range(1,self.options.check_every+1):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    self._block(count)
                self.graphs[count]=graph
            torch.cuda.current_stream(self.x.device).wait_stream(stream)

    @torch.no_grad()
    def solve(self,rhs,initial=None):
        opt=self.options
        for v in (rhs,) if initial is None else (rhs,initial):
            if v.shape!=self.x.shape or v.dtype!=self.x.dtype or v.device!=self.x.device:
                raise ValueError('mass solve shape, dtype or device mismatch')
        def true_residual():
            self.action(self.x,self.Ad)
            torch.sub(rhs,self.Ad,out=self.r)
            return torch.linalg.vector_norm(self.r).item()
        if initial is None:
            self.x.zero_()
            # M*0 is exactly zero; skip the redundant sparse multiplication.
            self.r.copy_(rhs)
        else:
            self.x.copy_(initial)
            self.action(self.x,self.Ad)
            torch.sub(rhs,self.Ad,out=self.r)
        finite,rhs_norm,true_norm=self.initial_metrics(rhs,self.r,self.x).tolist()
        if not finite:
            raise ValueError('finite mass RHS and initial guess required')
        if not isfinite(rhs_norm) or not isfinite(true_norm):
            raise RuntimeError('nonfinite initial PCG residual')
        tolerance=max(opt.atol,opt.rtol*rhs_norm)
        self.tol.fill_(tolerance)
        if true_norm<=tolerance:
            return self.x.clone(),SolveInfo(0,true_norm,rhs_norm,tolerance)
        self.rz.copy_(self.restart(self.r,self.diagonal,self.d))
        self.active.fill_(True)
        self.broken.fill_(False)
        iteration=0
        while iteration<opt.max_iterations:
            count=min(opt.check_every-iteration%opt.check_every,
                      opt.recompute_every-iteration%opt.recompute_every,
                      opt.max_iterations-iteration)
            if self.graphs:
                self.graphs[count].replay()
            else:
                self._block(count)
            iteration+=count
            failed,value=torch.stack((self.broken.to(rhs.dtype),self.rn)).tolist()
            if failed:
                raise RuntimeError('PCG breakdown: operator/preconditioner must be positive definite and finite')
            restart=iteration%opt.recompute_every==0
            if restart or value<=tolerance or iteration==opt.max_iterations:
                true_norm=true_residual()
                if not isfinite(true_norm):
                    raise RuntimeError('nonfinite true PCG residual')
                if true_norm<=tolerance:
                    return self.x.clone(),SolveInfo(iteration,true_norm,rhs_norm,tolerance)
                restart=True
                self.active.fill_(True)
            self.rz.copy_(self.restart(self.r,self.diagonal,self.d) if restart else
                          self.direction(self.r,self.diagonal,self.d,self.rz,self.active))
        raise RuntimeError(f'PCG did not converge: {true_norm:g} > {tolerance:g}')
