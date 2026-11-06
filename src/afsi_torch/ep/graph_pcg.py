"""Reusable CUDA-Graph Jacobi-PCG, with bounded blocks and true-residual acceptance."""
import math
import torch
from ..fluid.solvers import SolveInfo
from .gpu_kernels import CSRAction, update_stats, direction


class GraphJacobiPCG:
    def __init__(self, matrix, diagonal, options, *, ordering='natural'):
        self.matrix,self.options=matrix,options
        self.permutation=None
        if ordering=='rcm':
            # One-time CPU graph ordering only; all PDE solves remain on GPU.
            from collections import deque
            import numpy as np
            rows=matrix.crow_indices().cpu().numpy();cols=matrix.col_indices().cpu().numpy()
            degree=np.diff(rows);seen=np.zeros(len(degree),dtype=bool);order=[]
            for seed in np.argsort(degree,kind='stable'):
                if seen[seed]:continue
                queue=deque([int(seed)]);seen[seed]=True
                while queue:
                    node=queue.popleft();order.append(node)
                    neighbors=[int(j) for j in cols[rows[node]:rows[node+1]] if not seen[j]]
                    neighbors.sort(key=lambda j:(degree[j],j))
                    for j in neighbors:
                        if not seen[j]:seen[j]=True;queue.append(j)
            self.permutation=torch.tensor(order[::-1],device=matrix.device,dtype=torch.long)
            inverse=torch.empty_like(self.permutation)
            inverse[self.permutation]=torch.arange(len(degree),device=matrix.device)
            coo=matrix.to_sparse_coo()
            matrix=torch.sparse_coo_tensor(inverse[coo.indices()],coo.values(),matrix.shape,
                device=matrix.device).coalesce().to_sparse_csr()
            diagonal=diagonal[self.permutation]
        elif ordering!='natural':
            raise ValueError('unknown CSR ordering')
        self.diagonal=diagonal
        self.action=CSRAction(matrix)
        self.x,self.r,self.z,self.p,self.ap,self.b,self.true=[torch.empty_like(diagonal) for _ in range(7)]
        scalar=lambda:diagonal.new_zeros(())
        self.rho,self.next_rho,self.curvature,self.rnorm,self.tol=[scalar() for _ in range(5)]
        self.active=torch.zeros((),device=matrix.device,dtype=torch.bool)
        self.bad=self.active.clone()
        blocks=(diagonal.numel()+255)//256
        self.partial_rr=diagonal.new_empty(blocks)
        self.partial_rz=diagonal.new_empty(blocks)
        self.graphs={}
        self._prepared=False

    def _block(self,count):
        for _ in range(count):
            self.action(self.p,self.ap)
            torch.dot(self.p,self.ap,out=self.curvature)
            valid=torch.isfinite(self.curvature)&(self.curvature>0)&torch.isfinite(self.rho)&(self.rho>0)
            self.bad.logical_or_(self.active&~valid)
            enabled=self.active&valid
            a=torch.where(enabled,self.rho/torch.where(valid,self.curvature,1.),0.)
            update_stats(self.x,self.r,self.z,self.p,self.ap,self.diagonal,a,
                         self.partial_rr,self.partial_rz)
            torch.sum(self.partial_rr,dim=0,out=self.rnorm);self.rnorm.sqrt_()
            self.active.logical_and_(torch.isfinite(self.rnorm)&(self.rnorm>self.tol)&valid)
            torch.sum(self.partial_rz,dim=0,out=self.next_rho)
            beta=torch.where(self.active,self.next_rho/torch.where(self.rho!=0,self.rho,1.),0.)
            direction(self.p,self.z,beta,self.active)
            self.rho.copy_(self.next_rho)

    def _graph(self,count):
        if count not in self.graphs:
            # Warmup and capture use disposable state; caller initializes after this.
            stream=torch.cuda.Stream(device=self.x.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.p.zero_();self.r.zero_();self.x.zero_();self.rho.fill_(1);self.active.fill_(False)
                self._block(count)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):self._block(count)
            self.graphs[count]=graph
        return self.graphs[count]

    def _restart(self):
        self.r.copy_(self.true)
        torch.div(self.r,self.diagonal,out=self.z)
        self.p.copy_(self.z)
        torch.dot(self.r,self.z,out=self.rho)
        self.active.fill_(True)

    @torch.no_grad()
    def solve(self,rhs,*,initial,initial_residual=None):
        o=self.options
        block=min(o.check_every,o.recompute_every,o.max_iterations)
        # Capture all possible tail block sizes before copying external input.
        if not self._prepared:
            sizes=set();cursor=0
            while cursor<o.max_iterations:
                count=min(block,o.max_iterations-cursor,o.recompute_every-cursor%o.recompute_every)
                sizes.add(count);cursor+=count
            for count in sizes:self._graph(count)
            self._prepared=True
        perm=self.permutation
        self.b.copy_(rhs if perm is None else rhs[perm])
        self.x.copy_(initial if perm is None else initial[perm])
        norm=float(torch.linalg.vector_norm(self.b))
        if not math.isfinite(norm):raise RuntimeError('nonfinite EP PCG RHS')
        tolerance=max(o.atol,o.rtol*norm);self.tol.fill_(tolerance);self.bad.fill_(False)
        matvecs=0;true_checks=0
        if initial_residual is None:
            self.action(self.x,self.ap);torch.sub(self.b,self.ap,out=self.true)
            matvecs+=1;true_checks+=1
        else:
            self.true.copy_(initial_residual if perm is None else initial_residual[perm])
        residual=float(torch.linalg.vector_norm(self.true))
        if not math.isfinite(residual):raise RuntimeError('nonfinite initial EP residual')
        iterations=0
        # Even a supplied algebraic residual cannot bypass true acceptance.
        if residual<=tolerance and initial_residual is not None:
            self.action(self.x,self.ap);torch.sub(self.b,self.ap,out=self.true)
            matvecs+=1;true_checks+=1
            residual=float(torch.linalg.vector_norm(self.true))
            if not math.isfinite(residual):raise RuntimeError('nonfinite initial EP true residual')
        if residual>tolerance:self._restart()
        while residual>tolerance and iterations<o.max_iterations:
            count=min(block,o.max_iterations-iterations,o.recompute_every-iterations%o.recompute_every)
            self.graphs[count].replay();iterations+=count
            matvecs+=count
            diagnostics=torch.stack((self.rnorm,self.bad.to(self.x.dtype),
                                     self.active.to(self.x.dtype))).tolist()
            residual,bad,active=diagnostics
            if bad or not math.isfinite(residual):raise RuntimeError('EP PCG nonfinite/negative curvature')
            if residual<=tolerance or not active or iterations%o.recompute_every==0 or iterations==o.max_iterations:
                self.action(self.x,self.ap);torch.sub(self.b,self.ap,out=self.true)
                matvecs+=1;true_checks+=1
                residual=float(torch.linalg.vector_norm(self.true))
                if not math.isfinite(residual):raise RuntimeError('nonfinite EP true residual')
                if residual>tolerance:self._restart()
        if residual>tolerance:raise RuntimeError('EP PCG iteration limit reached')
        # Publication owns its memory; later solves cannot overwrite a saved voltage.
        result=self.x.clone() if perm is None else torch.empty_like(self.x).scatter_(0,perm,self.x)
        self.last_diagnostics=dict(executed_iteration_slots=iterations,matvecs=matvecs,
                                   true_residual_checks=true_checks)
        return result,SolveInfo(iterations,residual,norm,tolerance)

    def close(self):
        self.graphs.clear();self._prepared=False
