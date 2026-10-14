"""Execute the pinned TorchCor electric statements without AFSEI field checks."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from .torchcor_pde import load_core,COMMIT

MONODOMAIN_SHA256='ad46ebba22086531240f77f4348b3eb0cbbe4e9c04d55a2523828e541374f668'


def load_native_electric(root):
    classes,provenance=load_core(root)
    path=Path(root)/'torchcor/simulator/monodomain.py'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=MONODOMAIN_SHA256:
        raise ValueError('native TorchCor electric source differs from pinned reference')
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Monodomain')
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    start=next(i for i,n in enumerate(step.body) if isinstance(n,ast.Assign)
               and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='Istim')
    stop=next(i for i,n in enumerate(step.body[start:],start) if isinstance(n,ast.Assign)
              and isinstance(n.targets[0],ast.Tuple) and isinstance(n.value,ast.Call)
              and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='solve')
    clone=next(n for n in step.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
               and n.targets[0].id=='b')
    template=ast.parse('def electric(self,u,t,a_tol,r_tol,max_iter):\n    return u,n_iter\n').body[0]
    template.body=[clone,*step.body[start:stop+1],template.body[0]]
    module=ast.fix_missing_locations(ast.Module(body=[template],type_ignores=[]))
    scope={'torch':torch};exec(compile(module,str(path),'exec'),scope)
    provenance['native_monodomain_sha256']=MONODOMAIN_SHA256
    provenance['native_electric_lines']=[step.body[start].lineno,step.body[stop].end_lineno]
    provenance['extraction']='unmodified native b=u.clone() and electric statements; ODE, NVML and output omitted'
    return classes,scope['electric'],provenance


class NativeTorchCorPDE:
    """Native Vm=mV, dt=ms; conductivity converted from common cm/s units.

    Captured electric statements keep native stimulus /100 conversion. All
    representation conversions and stopping calibration occur before timing.
    """
    def __init__(self,M,K,parameters,dt_s,options,classes,electric):
        if parameters.capacitance_uf_per_cm2!=1.:
            raise ValueError('native PDE-only b=u.clone() benchmark requires the declared Cm=1')
        _,Preconditioner,ConjugateGradient=classes
        self.M=M;self.K=(K.to_sparse_coo()*(1/(parameters.surface_to_volume_per_cm*1000))).coalesce().to_sparse_csr()
        self.dt=dt_s*1000;self.theta=.5;self.Cm=1.
        self.A=(self.M.to_sparse_coo()+self.K.to_sparse_coo()*(self.dt*self.theta)).coalesce().to_sparse_csr()
        preconditioner=Preconditioner();preconditioner.create_Jocobi(self.A.to_sparse_coo())
        self.cg=ConjugateGradient(preconditioner,self.A,dtype=torch.float64)
        self.cg.initialize(M.values().new_zeros(M.shape[0]),linear_guess=False)
        self.electric=electric;self.options=options
        self.native_equation_scale=dt_s/parameters.surface_to_volume_per_cm
        self.max_diagonal=(1/preconditioner.PC).max().item()

    def rhs(self,u,source):
        return self.M@(u+self.dt*source)-(1-self.theta)*self.dt*(self.K@u)

    def prepare_input(self,V,I,S):
        u=V*1000;source=S-I;rhs=self.rhs(u,source)
        rhs_physical=rhs/(1000*self.native_equation_scale)
        tolerance=max(self.options.atol,self.options.rtol*torch.linalg.vector_norm(rhs_physical).item())
        target_native=tolerance*1000*self.native_equation_scale
        preb=torch.linalg.vector_norm(self.cg.preconditioner.apply(rhs)).item()
        relative=.25*target_native/(self.max_diagonal*preb)
        stimulus=SimpleNamespace(apply=lambda t,field=source*100:field)
        return u,stimulus,relative,rhs_physical

    def step(self,prepared):
        u,self.stimuli,relative,_=prepared
        self.cg.x.copy_(u)
        result,count=self.electric(self,u,0.,0.,relative,self.options.max_iterations)
        return result,min(count+1,self.options.max_iterations)
