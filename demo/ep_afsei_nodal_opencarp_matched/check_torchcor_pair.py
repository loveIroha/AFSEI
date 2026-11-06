from pathlib import Path
import sys,json,itertools
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from validation.torchcor_pde import load_core,COMMIT
from validation.torchcor_euler import TorchCorEulerPDE
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.fluid.solvers import SolverOptions
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
base=ROOT/'results/ep_opencarp_same_mesh_20261030';common=json.loads((base/'common_configuration.json').read_text())
src=ROOT/'.deps'/('torchcor-'+COMMIT);classes,provenance=load_core(src)
# Representative native 5-tet topology, including the coordinate range/extents.
X=np.load(base/'coordinates_cm.npy');mesh=common['original_mesh_base']
rows=[]
with open(mesh+'.elem') as f:
    next(f)
    rows=[list(map(int,line.split()[1:5])) for line in itertools.islice(f,1024)]
C=np.array(rows,dtype=np.int64);nodes,inverse=np.unique(C,return_inverse=True)
x=torch.as_tensor(X[nodes],device='cuda');cells=torch.as_tensor(inverse.reshape(-1,4),device='cuda')
p=MonodomainParameters(**common['parameters']);o=SolverOptions(rtol=1e-10,atol=1e-13,max_iterations=1000,check_every=5,recompute_every=40)
f=torch.zeros_like(x);s=f.clone();f[:,0]=1;s[:,1]=1
af=Monodomain(x,cells,f,s,parameters=p,numerics=MonodomainNumerics(discretization='consistent_p1',diffusion_quadrature='consistent_p1',solver=o))
K,M=classes[0](x,cells,x.device,x.dtype).assemble_matrices(torch.diag(x.new_tensor(p.conductivity_us_per_cm)))
mass_gap=float((M-af.mass.to_sparse_coo()).coalesce().values().abs().max())/float(M.values().abs().max())
stiff_gap=float((K-af.stiffness.to_sparse_coo()).coalesce().values().abs().max())/float(K.values().abs().max())
assert mass_gap<1e-11 and stiff_gap<1e-11,(mass_gap,stiff_gap)
tc=TorchCorEulerPDE(x,M.to_sparse_csr(),K.to_sparse_csr(),p,2e-5,o,src)
v=-.08+.02*torch.sin(torch.arange(len(x),device='cuda')*.11);I=torch.cos(torch.arange(len(x),device='cuda')*.07)
stim=torch.zeros_like(v);stim[:3]=35.71
u,info=tc.step(v,I,stim,2e-5);w,winfo=af.step(v,I,stim,2e-5)
gap=float((u-w).abs().max())*1000
assert gap<1e-5,(gap,info,winfo)
report=dict(passed=True,torchcor=tc.provenance,cells=len(cells),nodes=len(x),relative_mass_gap=mass_gap,
    relative_stiffness_gap=stiff_gap,maximum_solution_gap_mv=gap,torchcor_true_residual_ratio=info.residual_norm/info.tolerance,
    afsei_true_residual_ratio=winfo.residual_norm/winfo.tolerance,scope='Native TorchCor local assembly, native CN RHS/CG, common physical units and acceptance criterion')
out=ROOT/'results/ep_euler_fem_pair_20261031';out.mkdir(exist_ok=True)
(out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
