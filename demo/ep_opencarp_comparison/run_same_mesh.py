"""Same geometry, connectivity, Grandi update, CN and forcing; vary mass quadrature."""
from pathlib import Path
import argparse,json,sys,time,os
import numpy as np,torch
root=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(root/'src'))
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.fluid.solvers import SolverOptions
from shared_cell import module

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['consistent','nodal'],required=True)
    parser.add_argument('--steps',type=int,default=2500);parser.add_argument('--suffix',default='')
    args=parser.parse_args();torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    base=root/'results/ep_opencarp_same_mesh_20261030';common=json.loads((base/'common_configuration.json').read_text())
    folder=base/(args.method+args.suffix);folder.mkdir(exist_ok=False)
    start=time.perf_counter();dt=common['dt_s'];N=common['nodes'];device='cuda'
    X=torch.as_tensor(np.load(base/'coordinates_cm.npy'),device=device)
    p=Monodomain.__new__(Monodomain);p.X=X;p.parameters=MonodomainParameters(**common['parameters'])
    p.numerics=MonodomainNumerics(discretization='nodal_p1' if args.method=='nodal' else 'consistent_p1',
        diffusion_quadrature='nodal_p1',backend='torch_pcg_jacobi_light',time_scheme='crank_nicolson',
        solver=SolverOptions(rtol=1e-10,atol=1e-13,max_iterations=1000,check_every=5,recompute_every=40))
    p.solver_options=p.numerics.solver;p._systems={}
    with np.load(base/'operators.npz') as z:
        rows=torch.as_tensor(z['crow'],device=device);cols=torch.as_tensor(z['col'],device=device)
        p.stiffness=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['K'],device=device),size=(N,N))
        p.stiffness_diagonal=torch.as_tensor(z['kd'],device=device)
        lump=torch.as_tensor(z['lump'],device=device)
        if args.method=='consistent':
            p.mass=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['M'],device=device),size=(N,N));p.mass_diagonal=lump*.4
        else:
            p.mass_diagonal=lump;p.mass=torch.sparse_csr_tensor(torch.arange(N+1,device=device),torch.arange(N,device=device),lump,size=(N,N))
    p.prepare(dt)
    meta=json.loads((Path(__file__).parent/'shared_cell_provenance.json').read_text());cell=module();ns=meta['state_count']
    y=torch.empty((ns,N),device=device);yn=torch.empty_like(y);v_mv=torch.empty(N,device=device)
    current=torch.empty_like(v_mv);guards=torch.empty(N,device=device,dtype=torch.int64)
    cell.initialize(y,v_mv,dt*1000);v=v_mv/1000
    zero=torch.zeros_like(v);stimulus=zero.masked_fill((X<=.15+1e-12).all(1),35.71)
    # Warmup scalar/vector kernels and PCG outside all reported trajectory timers.
    cell.step(y,v_mv,yn,current,guards,dt*1000)
    p.step(v,zero,zero,dt)
    cell.initialize(y,v_mv,dt*1000);v=v_mv/1000
    activation=torch.full_like(v,float('nan'));trace=[];ode=[];pde=[];iterations=[];residuals=[];minmax=[]
    probes=torch.tensor([0,444344,888688,N-1],device=device)
    steps=args.steps;stride=50;frames=steps//stride+1
    raw=np.memmap(folder/'vm.igb',dtype='<f4',mode='w+',offset=1024,shape=(frames,N))
    header=f'x:{N} y:1 z:1 t:{frames} type:float systeme:little_endian inc_t:1 unites_t:ms unites:mV '
    with (folder/'vm.igb').open('r+b') as f:f.write(header.encode().ljust(1023,b' ')+b'\x0c')
    torch.cuda.synchronize();setup=time.perf_counter()-start;beg=time.perf_counter();maxguards=0;wall_blocks=[]
    for step in range(steps+1):
        if step%stride==0:
            raw[step//stride]=v.cpu().numpy()*1000
            minmax.append([step*dt*1000,float(v.min())*1000,float(v.max())*1000])
        trace.append(v[probes].cpu().numpy()*1000)
        if step==steps:break
        torch.cuda.synchronize();t=time.perf_counter()
        v_mv.copy_(v).mul_(1000)
        cell.step(y,v_mv,yn,current,guards,dt*1000)
        # Same native first-order reaction update; Cm=1 microF/cm2.
        source=stimulus if step<100 else zero
        vstar=v+dt*(source-current)
        torch.cuda.synchronize();ode.append(time.perf_counter()-t)
        t=time.perf_counter();vn,info=p.step(vstar,zero,zero,dt)
        torch.cuda.synchronize();pde.append(time.perf_counter()-t);iterations.append(info.iterations)
        if step%250==0 or step==steps-1:
            rhs=p._rhs_buffers[dt](vstar,zero,zero).clone();A=p._systems[dt][1]
            r=float(torch.linalg.vector_norm(rhs-A@vn));tol=max(1e-13,1e-10*float(torch.linalg.vector_norm(rhs)))
            if r>tol:raise RuntimeError(f'True residual failed: {r/tol}')
            residuals.append(r/tol);maxguards=max(maxguards,int(guards.max()))
            if not torch.isfinite(yn).all():raise RuntimeError('Nonfinite Grandi state')
        crossing=torch.isnan(activation)&(v<0)&(vn>=0)
        den=torch.where(crossing,vn-v,torch.ones_like(v))
        activation=torch.where(crossing,step*dt-dt*v/den,activation)
        v=vn;y,yn=yn,y
        if (step+1)%250==0:
            torch.cuda.synchronize();progress=dict(method=args.method,steps=step+1,time_ms=(step+1)*dt*1000,
                elapsed_s=time.perf_counter()-beg,voltage_min_mv=float(v.min())*1000,voltage_max_mv=float(v.max())*1000)
            (folder/'progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    torch.cuda.synchronize();wall=time.perf_counter()-beg;raw.flush()
    act=activation.cpu().numpy()*1000;np.save(folder/'activation_ms.npy',act)
    np.savez(folder/'traces.npz',time_ms=np.arange(steps+1)*dt*1000,voltage_mv=np.array(trace),probe_ids=probes.cpu().numpy(),
        pde_step_s=np.array(pde),ode_step_s=np.array(ode),iterations=np.array(iterations),minmax=np.array(minmax))
    report=dict(status='completed',method=args.method,nodes=N,cells=common['cells'],time_ms=steps*dt*1000,steps=steps,
        gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),setup_s=setup,
        wall_s=wall,pde_s=sum(pde),ode_s=sum(ode),pde_mean_ms=float(np.mean(pde)*1000),ode_mean_ms=float(np.mean(ode)*1000),
        mean_iterations=float(np.mean(iterations)),max_true_residual_ratio=max(residuals),maximum_cell_clamps_per_node_step=maxguards,
        activated_fraction=float(np.isfinite(act).mean()),far_corner_activation_ms=float(act[-1]),
        activation_min_ms=float(np.nanmin(act)),activation_max_ms=float(np.nanmax(act)),
        maximum_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        timing_scope='Synchronized wall times; PDE includes RHS, input checks and Jacobi-PCG; ODE includes common reaction update; wall includes matching 1ms output and probes.',
        cell='Benchmark-only mechanically adapted official Grandi v20; native Rush-Larsen and float gates; production GPB/Land not changed')
    (folder/'report.json').write_text(json.dumps(report,indent=2));p.close();print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
