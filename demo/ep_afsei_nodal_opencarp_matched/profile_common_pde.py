"""One fixed-input PDE control, no extra EP trajectory or ionic solve.

Separates standard/nodal quadrature with a shared optimized backend from
native-TorchCor/optimized-AFSEI execution on the same consistent system.
"""
from pathlib import Path
import sys,time,json
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.fluid.solvers import SolverOptions
from validation.torchcor_euler import TorchCorEulerPDE
from validation.torchcor_pde import COMMIT
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
out=ROOT/'results/ep_euler_fem_pair_20261031';cache=ROOT/'results/ep_opencarp_same_mesh_20261030'
common=json.loads((cache/'common_configuration.json').read_text());N=common['nodes'];dt=common['dt_s']
X=torch.as_tensor(np.load(cache/'coordinates_cm.npy'),device='cuda')
with np.load(cache/'operators.npz') as z:
    rows=torch.as_tensor(z['crow'],device='cuda');cols=torch.as_tensor(z['col'],device='cuda')
    K=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['K'],device='cuda'),size=(N,N))
    M=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['M'],device='cuda'),size=(N,N))
    lump=torch.as_tensor(z['lump'],device='cuda');kd=torch.as_tensor(z['kd'],device='cuda')
params=MonodomainParameters(**common['parameters']);opts=SolverOptions(rtol=1e-10,atol=1e-13,max_iterations=1000,check_every=5,recompute_every=40)
methods={}
for label,nodal in [('consistent_afsei',False),('nodal_afsei',True)]:
    p=Monodomain.__new__(Monodomain);p.X=X;p.parameters=params;p.solver_options=opts;p._systems={}
    p.numerics=MonodomainNumerics(discretization='nodal_p1' if nodal else 'consistent_p1',
        diffusion_quadrature='nodal_p1' if nodal else 'consistent_p1',solver=opts)
    p.stiffness=K;p.stiffness_diagonal=kd;p.mass_diagonal=lump if nodal else .4*lump
    p.mass=M if not nodal else torch.sparse_csr_tensor(torch.arange(N+1,device='cuda'),torch.arange(N,device='cuda'),lump,size=(N,N))
    p.prepare(dt);methods[label]=p
methods['consistent_torchcor']=TorchCorEulerPDE(X,M,K,params,dt,opts,ROOT/'.deps'/('torchcor-'+COMMIT))
# Real wavefront voltage snapshot; identical prescribed source fields for all.
voltage=np.memmap(out/'nodal_afsei/vm.igb',mode='r',dtype='<f4',offset=1024,shape=(51,N))
V=torch.as_tensor(np.asarray(voltage[15],dtype=np.float64)/1000,device='cuda')
I=torch.zeros_like(V);S=I.masked_fill((X<=.15+1e-12).all(1),35.71)
reports={};solutions={}
for label in ['consistent_torchcor','consistent_afsei','nodal_afsei']:
    p=methods[label]
    for _ in range(2):p.step(V,I,S,dt)
    samples=[];iterations=[];ratios=[]
    for _ in range(5):
        torch.cuda.synchronize();beg=time.perf_counter();v,info=p.step(V,I,S,dt);torch.cuda.synchronize()
        samples.append((time.perf_counter()-beg)*1000);iterations.append(info.iterations);ratios.append(info.residual_norm/info.tolerance)
    reports[label]=dict(median_ms=float(np.median(samples)),samples_ms=samples,iterations=iterations,max_true_residual_ratio=max(ratios))
    solutions[label]=v
gap=float((solutions['consistent_torchcor']-solutions['consistent_afsei']).abs().max())*1000
assert gap<1e-4,gap
report=dict(timings=reports,consistent_solutions_max_difference_mv=gap,
    shared_AFSEI_backend_standard_over_nodal_time_ratio=reports['consistent_afsei']['median_ms']/reports['nodal_afsei']['median_ms'],
    same_consistent_system_native_TorchCor_over_AFSEI_time_ratio=reports['consistent_torchcor']['median_ms']/reports['consistent_afsei']['median_ms'],
    scope='Fixed common V=actual nodal 15ms snapshot, I=0, same corner stimulus. PDE only, no EP/ODE rerun. GPU0, 2 warmups + 5 synchronized wall samples; both true tolerances unchanged.',
    interpretation='Shared-backend ratio combines mass quadrature/RHS work and its linear-system iterations. Native/AFSEI ratio on identical consistent system measures execution plus required norm/unit adaptation; not FEM approximation error.')
(out/'common_pde_control.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
