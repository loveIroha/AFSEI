"""Benchmark adapter for pinned, unmodified TorchCor assembly and JIT CG.

Only PDE core modules are imported: no optional ECG/visualization dependencies.
No TorchCor source is copied or patched. External snapshots use AFSEI V/cm/s
units and the common theta equation theta*K + Sv*Cm*M/dt.
"""
import hashlib
import importlib
import sys
import types
from pathlib import Path
import torch

COMMIT='b02daee87d4283888981cbfc3a6b00f57789a820'
SOURCE_SHA256={
    'torchcor/core/assemble.py':'2a10b3602d1c455bc906b15d9eeada7f23271a58abf6fce26854d8196d33b67d',
    'torchcor/core/preconditioner.py':'9b7e9b701742f67940ec3f85ee64387f28c711683ecea13826cf70e64881ab9a',
    'torchcor/core/solver.py':'bb654b360a8b9897648f571ea609c94227605d3c2bf374c0bbaed34ef5153a7d',
}


def load_core(root):
    root=Path(root).resolve()
    if not (root/'torchcor/core/solver.py').is_file():
        raise FileNotFoundError('pinned TorchCor source directory required')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_SHA256}
    if hashes!=SOURCE_SHA256:
        raise ValueError('TorchCor PDE source differs from the pinned unmodified reference')
    # Import real top-level TorchCor, but skip core/__init__'s unrelated imports.
    sys.path.insert(0,str(root))
    tc=importlib.import_module('torchcor')
    if Path(tc.__file__).resolve().parent!=root/'torchcor':
        raise ValueError('a different TorchCor installation was already imported')
    if 'torchcor.core' not in sys.modules:
        package=types.ModuleType('torchcor.core')
        package.__path__=[str(root/'torchcor/core')]
        sys.modules['torchcor.core']=package
    classes=[getattr(importlib.import_module('torchcor.core.'+module),name) for module,name in
             [('assemble','Matrices3D'),('preconditioner','Preconditioner'),('solver','ConjugateGradient')]]
    return classes,dict(commit=COMMIT,source_dir=str(root),source_sha256=hashes)


class TorchCorCG:
    """Same starting Vm and conservative common TRUE-residual tolerance.

    TorchCor stops on ||D^-1 r||, not ||r||. Since ||r|| <= max(D)*||D^-1 r||,
    adjust its relative stopping threshold without changing its CG code.
    The nominal common tolerance is max(atol,rtol*||b||); .25 is a guard factor
    also used by AFSEI's AmgX adapter. Calibration is outside timed replays.
    """
    def __init__(self,A,options,classes):
        _,Preconditioner,ConjugateGradient=classes
        self.matrix=A
        self.options=options
        self.preconditioner=Preconditioner()
        self.preconditioner.create_Jocobi(A.to_sparse_coo())
        self.cg=ConjugateGradient(self.preconditioner,A,dtype=torch.float64)
        self.cg.initialize(A.values().new_zeros(A.shape[0]),linear_guess=False)
        self.diagonal_max=(1/self.preconditioner.PC).max().item()

    def calibrate(self,rhs):
        rhs_norm=torch.linalg.vector_norm(rhs).item()
        target=max(self.options.atol,self.options.rtol*rhs_norm)
        preb_norm=torch.linalg.vector_norm(self.preconditioner.apply(rhs)).item()
        if preb_norm==0: raise ValueError('nonzero representative PDE right-hand side required')
        return dict(relative_tolerance=.25*target/(self.diagonal_max*preb_norm),
                    true_tolerance=target,rhs_norm=rhs_norm)

    def solve(self,rhs,initial,calibration):
        self.cg.x.copy_(initial)
        result,count=self.cg.solve(rhs,a_tol=0.,r_tol=calibration['relative_tolerance'],
                                   max_iter=self.options.max_iterations)
        return result,min(count+1,self.options.max_iterations)


class PDEMethod:
    """Shared PDE source/RHS operations and input checks for the controls."""
    def __init__(self,X,mass,mass_diagonal,A,parameters,dt,solver,*,nodal=False,stiffness=None,theta=1.):
        self.X,self.mass,self.mass_diagonal,self.matrix=X,mass,mass_diagonal,A
        self.parameters,self.dt,self.solver,self.nodal=parameters,dt,solver,nodal
        self.stiffness,self.theta=stiffness,theta

    def rhs(self,V,I,S):
        p=self.parameters
        source=p.surface_to_volume_per_cm*p.capacitance_uf_per_cm2/self.dt*V+p.surface_to_volume_per_cm*(S-I)
        rhs=self.mass_diagonal*source if self.nodal else torch.sparse.mm(self.mass,source[:,None]).flatten()
        if self.theta!=1.:
            rhs=rhs-(1-self.theta)*torch.sparse.mm(self.stiffness,V[:,None]).flatten()
        return rhs

    def step(self,V,I,S,calibration=None):
        from afsi_torch.ep import Monodomain
        for value,name in ((V,'voltage'),(I,'ionic current'),(S,'stimulus')):
            Monodomain._field(self,value,name)
        rhs=self.rhs(V,I,S)
        if isinstance(self.solver,TorchCorCG):
            return self.solver.solve(rhs,V,calibration)
        result,info=self.solver.solve(rhs,initial=V)
        return result,info.iterations
