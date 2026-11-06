"""Original quadrature IB pair restricted to fixed-end FE trial/test space.

Only the mass projection uses free DOFs. Spreading uses the same restricted
inverse, preserving adjointness. Root reactions are carried by the support.
No fluid equation, stencil, interpolation kernel or time update is changed.
"""
import torch
from .transfer import TriangleTransfer
from ..mac.mass_solver import MassSolver
from ..mac.mass_graph import GraphMassSolver


class FixedEndMassSolver:
    def __init__(self,mass,diagonal,options,fixed_nodes,backend):
        free=torch.ones(mass.shape[0],device=mass.device,dtype=mass.dtype)
        free[fixed_nodes]=0
        coo=mass.to_sparse_coo().coalesce()
        row,col=coo.indices()
        # Retain positive root diagonal entries. Zero RHS and warm starts
        # keep these coefficients exactly zero without penalty stiffness.
        keep=(free[row]*free[col]+((row==col)&(free[row]==0)).to(free.dtype))
        restricted=torch.sparse_coo_tensor(coo.indices(),coo.values()*keep,coo.shape,
            device=mass.device,dtype=mass.dtype,check_invariants=True).coalesce().to_sparse_csr()
        self.free=free[:,None]
        self.base=(GraphMassSolver if backend=='graph' else MassSolver)(restricted,diagonal,options)

    def solve(self,rhs,initial=None):
        return self.base.solve(rhs*self.free,None if initial is None else initial*self.free)


class FixedEndTriangleTransfer(TriangleTransfer):
    def __init__(self,*args,fixed_nodes,**kwargs):
        super().__init__(*args,**kwargs)
        ids=torch.unique(fixed_nodes)
        if (ids.dtype!=torch.int64 or ids.device!=self.mass.device or ids.numel()==0
            or int(ids.min())<0 or int(ids.max())>=self.mass.shape[0]):
            raise ValueError('valid nonempty fixed endpoint DOFs required')
        self.fixed_nodes=ids
        self.solver=FixedEndMassSolver(self.mass,self.diagonal,self.solver.options,
            ids,self.mass_backend)
