"""Pinned native TorchCor P1/CN RHS and CG, with a common external Euler ODE.

Global matrices reuse the exact-mesh affine P1 cache. Local formulas are
independently checked against unmodified Matrices3D. Assembly is not timed.
"""
from pathlib import Path
import ast,hashlib
from types import SimpleNamespace
import torch
from afsi_torch.fluid.solvers import SolveInfo
from .torchcor_native_pde import load_native_electric

class TorchCorEulerPDE:
    def __init__(self,X,M,K,parameters,dt_s,options,source_dir):
        classes,_,provenance=load_native_electric(source_dir)
        _,Preconditioner,ConjugateGradient=classes
        path=Path(source_dir)/'torchcor/simulator/monodomain.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Monodomain')
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        clone=next(n for n in step.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='b')
        start=next(j for j,n in enumerate(step.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='Istim')
        stop=next(j for j,n in enumerate(step.body[start:],start) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple))
        template=ast.parse('def native_rhs(self,u,t):\n return b\n').body[0]
        template.body=[clone,*step.body[start:stop],template.body[0]]
        module=ast.fix_missing_locations(ast.Module(body=[template],type_ignores=[]));scope={'torch':torch}
        exec(compile(module,str(path),'exec'),scope);self.native_rhs=scope['native_rhs']
        self.X=X;self.parameters=parameters;self.options=options;self.dt_s=dt_s
        if parameters.capacitance_uf_per_cm2!=1:raise ValueError('Cm=1 required by pinned native b=u.clone()')
        self.dt=dt_s*1000;self.theta=.5;self.Cm=1.;self.M=M
        self.K=(K.to_sparse_coo()/(parameters.surface_to_volume_per_cm*1000)).coalesce().to_sparse_csr()
        self.A=(M.to_sparse_coo()+self.K.to_sparse_coo()*(self.dt*self.theta)).coalesce().to_sparse_csr()
        pc=Preconditioner();pc.create_Jocobi(self.A.to_sparse_coo())
        self.cg=ConjugateGradient(pc,self.A,dtype=torch.float64);self.cg.initialize(X.new_zeros(len(X)),linear_guess=False)
        self.max_diagonal=float((1/pc.PC).max());self.scale=1000*dt_s/parameters.surface_to_volume_per_cm
        self.stimuli=SimpleNamespace(apply=lambda t:self.source)
        self.provenance=provenance|dict(native_rhs_lines=[step.body[start].lineno,step.body[stop-1].end_lineno],
            wrapper='Unmodified CN RHS AST and unmodified native JIT Jacobi-CG; common external FP64 GPB Euler',
            matrices='Shared exact-mesh affine P1 cache; local Matrices3D equivalence preflight',
            stopping='Native preconditioned absolute tolerance = common physical true tolerance * equation scale / max diagonal, using ||r||<=Dmax||D^-1 r||. Actual true residual checked each step.',
            timing='Includes units/source conversion, native RHS once, tolerance mapping, native CG, true residual and owned voltage publication. No duplicate RHS for calibration.')
    @torch.no_grad()
    def step(self,V,I,S,dt_s):
        if dt_s!=self.dt_s:raise ValueError('fixed benchmark EP dt required')
        u=V*1000;self.source=(S-I)*100
        b=self.native_rhs(self,u,0.)
        rhs_norm=float(torch.linalg.vector_norm(b))/self.scale
        tolerance=max(self.options.atol,self.options.rtol*rhs_norm)
        self.cg.x.copy_(u)
        result,count=self.cg.solve(b,a_tol=tolerance*self.scale/self.max_diagonal,r_tol=0.,max_iter=self.options.max_iterations)
        residual=float(torch.linalg.vector_norm(b-self.A@result))/self.scale
        if not residual<=tolerance:
            raise RuntimeError(f'TorchCor true residual acceptance failed: {residual}/{tolerance}')
        return result/1000,SolveInfo(min(count+1,self.options.max_iterations),residual,rhs_norm,tolerance)
    def close(self):pass
