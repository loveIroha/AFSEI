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


def _stage_kernel(function,device):
    if torch.device(device).type=='cuda':
        return torch.compile(function,fullgraph=True,dynamic=True,
                             options={'triton.cudagraphs':False})
    return tensor_kernel(function,device)


def _initialize_into(rhs,x,Ad,r,d,diag,rz,tol,active,broken,metrics,rtol,atol,zero):
    if zero:
        x.zero_(); r.copy_(rhs)
    else:
        torch.sub(rhs,Ad,out=r)
    finite=torch.isfinite(rhs).all() & torch.isfinite(x).all()
    rhs_norm=torch.linalg.vector_norm(rhs)
    residual_norm=rhs_norm if zero else torch.linalg.vector_norm(r)
    tol.copy_(torch.maximum(rhs_norm*rtol,atol))
    d.copy_(r/diag); rz.copy_((r*d).sum())
    active.copy_(finite & (residual_norm>tol)); broken.zero_()
    metrics.copy_(torch.stack((finite.to(rhs.dtype),rhs_norm,residual_norm)))


def _true_into(rhs,Ad,r,d,diag,rz,active,metrics):
    torch.sub(rhs,Ad,out=r)
    metrics[2].copy_(torch.linalg.vector_norm(r))
    # Preparing a restart before its host acceptance decision is harmless:
    # an accepted solve returns an owned x, otherwise these buffers are used.
    d.copy_(r/diag); rz.copy_((r*d).sum()); active.fill_(True)


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
        self.rhs_buffer=torch.empty_like(diagonal)
        self.guess_buffer=torch.empty_like(diagonal)
        self.metrics=diagonal.new_empty(3)
        self.relative_tolerance=diagonal.new_tensor(self.options.rtol)
        self.absolute_tolerance=diagonal.new_tensor(self.options.atol)
        self.initialize_into=_stage_kernel(_initialize_into,diagonal.device)
        self.true_into=_stage_kernel(_true_into,diagonal.device)
        self.stage_graphs={}
        self.graphs={}
        if diagonal.is_cuda:
            self._capture()
            self._capture_stages()

    def _initial_stage(self,zero):
        if not zero:
            self.x.copy_(self.guess_buffer)
            self.action(self.x,self.Ad)
        self.initialize_into(self.rhs_buffer,self.x,self.Ad,self.r,self.d,self.diagonal,
            self.rz,self.tol,self.active,self.broken,self.metrics,
            self.relative_tolerance,self.absolute_tolerance,zero)

    def _true_stage(self):
        self.action(self.x,self.Ad)
        self.true_into(self.rhs_buffer,self.Ad,self.r,self.d,self.diagonal,
                       self.rz,self.active,self.metrics)

    def _stage(self,name):
        if self.stage_graphs:
            self.stage_graphs[name].replay()
        elif name=='true':
            self._true_stage()
        else:
            self._initial_stage(name=='zero')

    @torch.no_grad()
    def _capture_stages(self):
        # Capture initialization and true residual/restart separately from the
        # PCG blocks: the ORIGINAL host acceptance/check schedule stays intact.
        with torch.cuda.device(self.x.device),capture_initialization():
            self.rhs_buffer.zero_(); self.guess_buffer.zero_()
            stream=torch.cuda.Stream(device=self.x.device)
            stream.wait_stream(torch.cuda.current_stream(self.x.device))
            operations={'zero':lambda:self._initial_stage(True),
                        'guess':lambda:self._initial_stage(False),'true':self._true_stage}
            with torch.cuda.stream(stream):
                for operation in operations.values():
                    for _ in range(3): operation()
            torch.cuda.current_stream(self.x.device).wait_stream(stream)
            for name,operation in operations.items():
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream): operation()
                self.stage_graphs[name]=graph
            torch.cuda.current_stream(self.x.device).wait_stream(stream)

    def _block(self,count):
        for i in range(count):
            self.action(self.d,self.Ad)
            self.advance_into(self.x,self.r,self.d,self.Ad,
                              self.rz,self.active,self.broken,self.tol,self.rn)
            if i+1<count:
                self.direction_into(self.r,self.diagonal,self.d,self.rz,self.active)

    @torch.no_grad()
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
        self.rhs_buffer.copy_(rhs)
        if initial is not None: self.guess_buffer.copy_(initial)
        self._stage('zero' if initial is None else 'guess')
        finite,rhs_norm,true_norm=self.metrics.tolist()
        if not finite:
            raise ValueError('finite mass RHS and initial guess required')
        if not isfinite(rhs_norm) or not isfinite(true_norm):
            raise RuntimeError('nonfinite initial PCG residual')
        tolerance=max(opt.atol,opt.rtol*rhs_norm)
        if true_norm<=tolerance:
            return self.x.clone(),SolveInfo(0,true_norm,rhs_norm,tolerance)
        if rhs.dtype==torch.float32:
            # Preserve the legacy Python-double initial comparison for f32.
            self.tol.fill_(tolerance); self.active.fill_(True)
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
                self._stage('true')
                true_norm=self.metrics[2].item()
                if not isfinite(true_norm):
                    raise RuntimeError('nonfinite true PCG residual')
                if true_norm<=tolerance:
                    return self.x.clone(),SolveInfo(iteration,true_norm,rhs_norm,tolerance)
                restart=True
            if not restart:
                self.direction_into(self.r,self.diagonal,self.d,self.rz,self.active)
        raise RuntimeError(f'PCG did not converge: {true_norm:g} > {tolerance:g}')
