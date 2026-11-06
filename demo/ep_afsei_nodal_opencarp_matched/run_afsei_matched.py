"""Actual AFSEI nodal/CN/Graph-Jacobi-PCG, native openCARP physical parameters.

This benchmark adapter uses frozen-V ionic-current-source CN with selectable
SDIRK or explicit Euler ODE, not openCARP's reaction-voltage splitting. Cached operators were
assembled on the exact native five-tetrahedra-per-cube connectivity.
"""
from pathlib import Path
import argparse,json,sys,time,os,traceback
import numpy as np
import torch
ROOT=Path('/mnt/large2/gjh/AFSEI')
sys.path.insert(0,str(ROOT/'src'))
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.fluid.solvers import SolverOptions
from reference_cell import module

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());here=Path(__file__).resolve().parent
    # Same SDIRK method; explicitly choose "reference" to reproduce old execution.
    cfg.setdefault('ode_execution','fused_cached_sdirk')
    cfg.setdefault('ode_method','sdirk')
    cfg.setdefault('pde_method','nodal_afsei')
    if cfg['pde_method'] not in ('nodal_afsei','consistent_afsei','torchcor_cg'):raise ValueError('unknown FEM method')
    if cfg['ode_method'] not in ('sdirk','explicit_euler'):
        raise ValueError('ode_method must be sdirk or explicit_euler')
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(cfg,indent=2))
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    cache=Path(cfg['operator_cache']);common=json.loads((cache/'common_configuration.json').read_text())
    N=common['nodes'];dt=cfg['dt_s'];steps=round(cfg['duration_s']/dt)
    stride=round(.001/dt);substeps=round(dt*1000/cfg['ode_dt_ms'])
    assert abs(substeps*cfg['ode_dt_ms']-dt*1000)<1e-12
    start=time.perf_counter();X=torch.as_tensor(np.load(cache/'coordinates_cm.npy'),device='cuda')
    p=Monodomain.__new__(Monodomain);p.X=X;p.parameters=MonodomainParameters(**common['parameters'])
    p.numerics=MonodomainNumerics(discretization='nodal_p1' if cfg['pde_method']=='nodal_afsei' else 'consistent_p1',diffusion_quadrature='nodal_p1' if cfg['pde_method']=='nodal_afsei' else 'consistent_p1',
        backend='torch_pcg_jacobi_light',time_scheme='crank_nicolson',solver=SolverOptions(**cfg['solver']))
    p.solver_options=p.numerics.solver;p._systems={}
    with np.load(cache/'operators.npz') as z:
        rows=torch.as_tensor(z['crow'],device='cuda');cols=torch.as_tensor(z['col'],device='cuda')
        p.stiffness=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['K'],device='cuda'),size=(N,N))
        p.stiffness_diagonal=torch.as_tensor(z['kd'],device='cuda');lump=torch.as_tensor(z['lump'],device='cuda')
        if cfg['pde_method']=='nodal_afsei':
            p.mass_diagonal=lump;p.mass=torch.sparse_csr_tensor(torch.arange(N+1,device='cuda'),torch.arange(N,device='cuda'),lump,size=(N,N))
        else:
            p.mass_diagonal=lump*.4
            p.mass=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['M'],device='cuda'),size=(N,N))
    if cfg['pde_method']!='torchcor_cg':p.prepare(dt)
    else:
        sys.path.insert(0,str(ROOT))
        from validation.torchcor_pde import COMMIT
        from validation.torchcor_euler import TorchCorEulerPDE
        p=TorchCorEulerPDE(X,p.mass,p.stiffness,p.parameters,dt,p.solver_options,ROOT/'.deps'/('torchcor-'+COMMIT))
        (output/'torchcor_provenance.json').write_text(json.dumps(p.provenance,indent=2))
    if cfg.get('ode_execution','reference') in ('fused_cached_sdirk','fused_cached'):
        import importlib.util
        spec=importlib.util.spec_from_file_location('optimized_native_gpb',ROOT/'demo/cell_optimization_20261030/fused.py')
        execution=importlib.util.module_from_spec(spec);spec.loader.exec_module(execution)
        cell=execution.native()
    elif cfg.get('ode_execution','reference')=='reference':
        cell=module()
    else:
        raise ValueError('unknown ODE execution backend')
    ode_advance=cell.euler if cfg['ode_method']=='explicit_euler' else cell.advance
    ode_description=('Explicit forward Euler, frozen Vm, averaged substep-end current'
        if cfg['ode_method']=='explicit_euler' else
        'Original AFSEI two-stage fixed-point SDIRK, frozen Vm, averaged substep current')
    ini=np.load(here/'native_initial_state.npz');initial=torch.as_tensor(ini['gpb'],device='cuda')
    y=initial.expand(-1,N).contiguous();yn=torch.empty_like(y)
    v=torch.full((N,),float(ini['voltage_mv'][0])/1000,device='cuda');vm=torch.empty_like(v)
    current=torch.empty_like(v);guard=torch.empty(N,device='cuda',dtype=torch.int64)
    zero=torch.zeros_like(v);mask=(X<=.15+1e-12).all(1);stim=zero.masked_fill(mask,35.71)
    # Compile/capture with disposable values. Neither warmup changes initial state.
    vm.copy_(v).mul_(1000);ode_advance(y,vm,yn,current,guard,cfg['ode_dt_ms'],substeps)
    p.step(v,zero,stim,dt);torch.cuda.synchronize()
    if cfg['pde_method']!='torchcor_cg':
        backend=p._systems[dt][2]
        assert type(backend.optimized).__name__=='GraphJacobiPCG'
        assert type(p._rhs_buffers[dt]).__name__=='BufferedRHS'
        backend_name=type(backend.optimized).__module__+'.'+type(backend.optimized).__name__;rhs_name=type(p._rhs_buffers[dt]).__name__
    else:backend_name=type(p.cg).__module__+'.'+type(p.cg).__name__;rhs_name='Pinned native TorchCor CN RHS statements'
    setup=time.perf_counter()-start
    shape=np.rint(np.asarray(cfg['geometry_mm'])/cfg['mesh_mm']).astype(int)+1
    assert int(np.prod(shape))==N,'Probe lattice must match cached mesh'
    q=np.rint(np.array([0,.25,.5,1])[:,None]*(shape-1)).astype(int)
    probes=torch.as_tensor(q[:,0]+shape[0]*q[:,1]+shape[0]*shape[1]*q[:,2],device='cuda')
    trace=torch.empty((steps+1,len(probes)),device='cuda');activation=torch.full_like(v,float('nan'))
    cai_index=int(np.flatnonzero(ini['state_names']=='Cai')[0])
    calcium_trace=torch.empty_like(trace);current_trace=torch.empty((steps,len(probes)),device='cuda')
    frames=steps//stride+1;raw=np.memmap(output/'vm.igb',dtype='<f4',mode='w+',offset=1024,shape=(frames,N))
    header=f'x:{N} y:1 z:1 t:{frames} type:float systeme:little_endian inc_t:1 unites_t:ms unites:mV '
    with (output/'vm.igb').open('r+b') as f:f.write(header.encode().ljust(1023,b' ')+b'\x0c')
    ode=[];pde=[];iterations=[];residuals=[];limits=[];maxguard=0
    guard_block=torch.zeros((),device='cuda',dtype=torch.int64)
    began=time.perf_counter()
    for step in range(steps+1):
        trace[step]=v[probes]*1000
        calcium_trace[step]=y[cai_index,probes]*1000
        if step%stride==0:
            raw[step//stride]=v.cpu().numpy()*1000
        if step==steps:break
        torch.cuda.synchronize();t=time.perf_counter()
        vm.copy_(v).mul_(1000)
        ode_advance(y,vm,yn,current,guard,cfg['ode_dt_ms'],substeps)
        torch.maximum(guard_block,guard.max(),out=guard_block)
        torch.cuda.synchronize();ode.append(time.perf_counter()-t)
        t=time.perf_counter()
        vn,info=p.step(v,current,stim if step*dt<.002-1e-14 else zero,dt)
        torch.cuda.synchronize();pde.append(time.perf_counter()-t)
        iterations.append(info.iterations);residuals.append(info.residual_norm/info.tolerance)
        current_trace[step]=current[probes]
        crossing=torch.isnan(activation)&(v<0)&(vn>=0)
        activation=torch.where(crossing,step*dt-dt*v/torch.where(crossing,vn-v,torch.ones_like(v)),activation)
        v=vn;y,yn=yn,y
        if (step+1)%stride==0 or step==steps-1:
            bad=int(guard_block);maxguard=max(maxguard,bad);guard_block.zero_()
            if bad or not torch.isfinite(y).all():raise RuntimeError(f'Invalid GPB state at step {step+1}, guard={bad}')
            limits.append([(step+1)*dt*1000,float(v.min())*1000,float(v.max())*1000])
            progress=dict(status='running',steps=step+1,time_ms=(step+1)*dt*1000,
                elapsed_s=time.perf_counter()-began,pde_s=sum(pde),ode_s=sum(ode),
                v_min_mv=limits[-1][1],v_max_mv=limits[-1][2],last_iterations=info.iterations)
            (output/'progress.json').write_text(json.dumps(progress,indent=2))
            print(json.dumps(progress),flush=True)
    torch.cuda.synchronize();wall=time.perf_counter()-began;raw.flush()
    act=activation.cpu().numpy()*1000;np.save(output/'activation_ms.npy',act)
    np.savez(output/'traces.npz',time_ms=np.arange(steps+1)*dt*1000,voltage_mv=trace.cpu().numpy(),
        probe_ids=probes.cpu().numpy(),pde_step_s=np.array(pde),ode_step_s=np.array(ode),
        iterations=np.array(iterations),minmax=np.array(limits),true_residual_ratios=np.array(residuals))
    np.savez(output/'cell_traces.npz',time_ms=np.arange(steps+1)*dt*1000,calcium_um=calcium_trace.cpu().numpy(),
        ionic_current_ua_cm2=current_trace.cpu().numpy(),current_interval_start_ms=np.arange(steps)*dt*1000,
        probe_ids=probes.cpu().numpy())
    checkpoint_s=0.
    if cfg.get('save_final_cell_state',False):
        t=time.perf_counter();np.save(output/'final_gpb_state.npy',y.cpu().numpy());checkpoint_s=time.perf_counter()-t
        (output/'state_names.json').write_text(json.dumps(ini['state_names'].tolist(),indent=2))
    report=dict(status='completed',nodes=N,cells=common['cells'],steps=steps,duration_ms=steps*dt*1000,
        gpu=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),setup_s=setup,wall_s=wall,
        pde_s=sum(pde),ode_s=sum(ode),pde_mean_ms=float(np.mean(pde)*1000),ode_mean_ms=float(np.mean(ode)*1000),
        mean_iterations=float(np.mean(iterations)),max_true_residual_ratio=max(residuals),max_cell_guard=maxguard,
        activated_fraction=float(np.isfinite(act).mean()),far_corner_activation_ms=float(act[-1]),
        activation_min_ms=float(np.nanmin(act)),activation_max_ms=float(np.nanmax(act)),
        initial_voltage_mv=float(ini['voltage_mv'][0]),stimulated_nodes=int(mask.sum()),
        initial_state='Complete native Grandi default, no conditioning; uniform across all nodes',
        numerical_methods=dict(PDE=('AFSEI nodal P1 vertex quadrature, CN, Jacobi-PCG' if cfg['pde_method']=='nodal_afsei' else
            'AFSEI standard continuous P1 Galerkin, consistent mass, CN, Jacobi-PCG' if cfg['pde_method']=='consistent_afsei' else
            'TorchCor standard continuous P1 Galerkin, consistent mass, native CN and Jacobi-CG'),
            ODE=ode_description,
            backend=backend_name,rhs=rhs_name,AMG=False,torchcor_solver=cfg['pde_method']=='torchcor_cg'),
        physical_model='Official openCARP v20 Grandi RHS/parameters/initial state; benchmark-only adapter',
        ode_execution=cfg.get('ode_execution','reference'),
        ode_method=cfg['ode_method'],ode_dt_ms=cfg['ode_dt_ms'],ode_substeps_per_ep=substeps,
        pde_method=cfg['pde_method'],
        final_cell_checkpoint_s=checkpoint_s,final_calcium_range_um=[float(y[cai_index].min())*1000,float(y[cai_index].max())*1000],
        maximum_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        timing_scope=('Synchronized wall: ODE includes Vm conversion; PDE includes field checks, buffered RHS and true-residual Graph-PCG; wall includes 1ms output and LAT/probes.'
            if cfg['pde_method']!='torchcor_cg' else
            'Synchronized wall: ODE includes Vm conversion; PDE includes unit/source conversion, pinned native CN RHS, common norm mapping, native TorchCor CG and true-residual verification; wall includes 1ms output and LAT/probes.'))
    (output/'report.json').write_text(json.dumps(report,indent=2));p.close()
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
