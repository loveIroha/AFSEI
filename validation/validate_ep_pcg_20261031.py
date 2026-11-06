"""GPU regression and fixed-input PDE timing; no full benchmark rerun."""
from pathlib import Path
import sys,importlib.util,time,json,os,subprocess,argparse
import numpy as np,torch
ROOT=Path('/mnt/large2/gjh/AFSEI');sys.path[:0]=[str(ROOT),str(ROOT/'src')]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results/ep_pcg_optimization_20261031_final')
OUT=parser.parse_args().output;OUT.mkdir(parents=True,exist_ok=False)
torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
backup=ROOT/'backups/ep_pcg_20261031'
def old_module(name,filename,replacements=()):
    import types
    full='afsi_torch.ep.'+name;m=types.ModuleType(full);m.__package__='afsi_torch.ep';m.__file__=str(backup/filename)
    sys.modules[full]=m;source=(backup/filename).read_text()
    for a,b in replacements:source=source.replace(a,b)
    exec(compile(source,str(backup/filename),'exec'),m.__dict__);return m
import afsi_torch.ep
oldgpu=old_module('_gpu_before','gpu_kernels.py')
oldgraph=old_module('_graph_before','graph_pcg.py',[('.gpu_kernels','._gpu_before')])
oldjac=old_module('_jacobi_before','jacobi.py',[('.graph_pcg','._graph_before')])
oldmono=old_module('_mono_before','monodomain.py',[('.jacobi import','._jacobi_before import'),('.gpu_kernels import','._gpu_before import')])
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.graph_pcg import GraphJacobiPCG
from afsi_torch.fluid.solvers import SolverOptions
opts=SolverOptions(rtol=1e-10,atol=1e-13,max_iterations=1000,check_every=5,recompute_every=40)
def timed(fn,repeats=5):
    for _ in range(2):fn()
    samples=[]
    for _ in range(repeats):
        torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();samples.append((time.perf_counter()-t)*1000)
    return dict(median_ms=float(np.median(samples)),samples_ms=samples)
# Small systems exercise zero residual, nonzero initial guess, periodic restart,
# graph tail, FP32/FP64, RCM, independent true residual, owned publication.
small=[]
for dtype in (torch.float64,torch.float32):
    n=301;i=torch.arange(n,device='cuda');idx=torch.cat((torch.stack((i,i)),torch.stack((i[:-1],i[1:])),torch.stack((i[1:],i[:-1]))),1)
    val=torch.cat((torch.full((n,),2.05,device='cuda',dtype=dtype),torch.full((2*(n-1),),-1.,device='cuda',dtype=dtype)))
    A=torch.sparse_coo_tensor(idx,val,(n,n)).coalesce().to_sparse_csr();d=torch.full((n,),2.05,device='cuda',dtype=dtype)
    o=SolverOptions(rtol=1e-9 if dtype==torch.float64 else 1e-4,atol=1e-12 if dtype==torch.float64 else 1e-6,max_iterations=503,check_every=7,recompute_every=21)
    for order in ('natural','rcm'):
        p=GraphJacobiPCG(A,d,o,ordering=order);b=torch.sin(i.to(dtype));initial=torch.full_like(b,.2)
        x,info=p.solve(b,initial=initial);true=float(torch.linalg.vector_norm(b-A@x));assert true<=info.tolerance*1.001
        saved=x.clone();p.solve(b*2,initial=initial);assert torch.equal(saved,x)
        z,zi=p.solve(torch.zeros_like(b),initial=torch.zeros_like(b),initial_residual=torch.zeros_like(b));assert zi.iterations==0
        small.append(dict(dtype=str(dtype),ordering=order,residual=true,tolerance=info.tolerance,iterations=info.iterations))
        p.close()
(OUT/'small_tests.json').write_text(json.dumps(small,indent=2));print('Small GPU tests passed',flush=True)
cache=ROOT/'results/ep_opencarp_same_mesh_20261030';common=json.loads((cache/'common_configuration.json').read_text());N=common['nodes'];dt=common['dt_s']
X=torch.as_tensor(np.load(cache/'coordinates_cm.npy'),device='cuda')
with np.load(cache/'operators.npz') as z:
    rows=torch.as_tensor(z['crow'],device='cuda');cols=torch.as_tensor(z['col'],device='cuda')
    K=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['K'],device='cuda'),size=(N,N))
    M=torch.sparse_csr_tensor(rows,cols,torch.as_tensor(z['M'],device='cuda'),size=(N,N))
    lump=torch.as_tensor(z['lump'],device='cuda');kd=torch.as_tensor(z['kd'],device='cuda')
params=MonodomainParameters(**common['parameters'])
def make(cls,nodal):
    p=cls.__new__(cls);p.X=X;p.parameters=params;p.solver_options=opts;p._systems={}
    p.numerics=MonodomainNumerics(discretization='nodal_p1' if nodal else 'consistent_p1',diffusion_quadrature='nodal_p1' if nodal else 'consistent_p1',solver=opts)
    p.stiffness=K;p.stiffness_diagonal=kd;p.mass_diagonal=lump if nodal else .4*lump
    p.mass=M if not nodal else torch.sparse_csr_tensor(torch.arange(N+1,device='cuda'),torch.arange(N,device='cuda'),lump,size=(N,N))
    p.prepare(dt);return p
voltage=np.memmap(ROOT/'results/ep_euler_fem_pair_20261031/nodal_afsei/vm.igb',mode='r',dtype='<f4',offset=1024,shape=(51,N))
I=torch.zeros(N,device='cuda');S=I.masked_fill((X<=.15+1e-12).all(1),35.71)
reports=[]
for nodal in (True,False):
    before=make(oldmono.Monodomain,nodal);after=make(Monodomain,nodal)
    for frame in (0,5,15,30,50):
        V=torch.as_tensor(np.asarray(voltage[frame],dtype=np.float64)/1000,device='cuda')
        a,ai=before.step(V,I,S,dt);b,bi=after.step(V,I,S,dt)
        gap=float((a-b).abs().max())*1000;assert gap<1e-4,gap
        A=after._systems[dt][1];buf=after._rhs_buffers[dt];rhs=buf(V,I,S)
        true=float(torch.linalg.vector_norm(rhs-A@b));assert true<=bi.tolerance*1.001,(true,bi.tolerance)
        legacy=timed(lambda:before.step(V,I,S,dt));new=timed(lambda:after.step(V,I,S,dt))
        row=dict(nodal=nodal,frame_ms=frame,before=legacy,after=new,speedup=legacy['median_ms']/new['median_ms'],max_difference_mv=gap,
            true_residual_ratio=true/bi.tolerance,before_slots=ai.iterations,after_slots=bi.iterations,diagnostics=after._systems[dt][2].optimized.last_diagnostics)
        reports.append(row);print(json.dumps(row),flush=True)
    # Matvec-only check; cuSPARSE allocation retained in reported cost.
    graph=after._systems[dt][2].optimized;dst=torch.empty_like(V)
    custom=timed(lambda:graph.action(V,dst));library=timed(lambda:torch.mv(graph.matrix,V,out=dst))
    print('SPMV',nodal,custom,library,flush=True)
    reports.append(dict(spmv_nodal=nodal,default_action=custom,library=library))
    before.close();after.close();del before,after;torch.cuda.empty_cache()
(OUT/'fixed_input.json').write_text(json.dumps(reports,indent=2))
# Full GPB+PDE short regression, 100 EP steps (2ms); no full 50ms trajectories.
base=json.loads((ROOT/'demo/ep_afsei_nodal_opencarp_matched/nodal_afsei.json').read_text())
base['duration_s']=.002
runner=ROOT/'demo/ep_afsei_nodal_opencarp_matched/run_afsei_matched.py'
for label in ('before','after'):
    cfg=base|dict(output=str(OUT/label));config=OUT/(label+'.json');config.write_text(json.dumps(cfg,indent=2))
    if label=='before':
        # Install the original modules under canonical names only in this child.
        worker=OUT/'run_before.py'
        worker.write_text("import sys,types,runpy\nfrom pathlib import Path\nr=Path('/mnt/large2/gjh/AFSEI');sys.path[:0]=[str(r/'src'),str(r/'demo/ep_afsei_nodal_opencarp_matched')]\nimport afsi_torch.ep\n"+
            "for name in ('gpu_kernels','graph_pcg','jacobi','monodomain'):\n p=r/'backups/ep_pcg_20261031'/(name+'.py');m=types.ModuleType('afsi_torch.ep.'+name);m.__package__='afsi_torch.ep';m.__file__=str(p);sys.modules[m.__name__]=m;exec(compile(p.read_text(),str(p),'exec'),m.__dict__)\n"+
            "sys.argv=[str(r/'demo/ep_afsei_nodal_opencarp_matched/run_afsei_matched.py'),'--config',sys.argv[1]]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n")
        cmd=[sys.executable,str(worker),str(config)]
    else:cmd=[sys.executable,str(runner),'--config',str(config)]
    with (OUT/(label+'.log')).open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
a=np.load(OUT/'before/final_gpb_state.npy',mmap_mode='r');b=np.load(OUT/'after/final_gpb_state.npy',mmap_mode='r');gap=float(np.max(np.abs(a-b)));assert gap<1e-8,gap
av=np.memmap(OUT/'before/vm.igb',mode='r',dtype='<f4',offset=1024);bv=np.memmap(OUT/'after/vm.igb',mode='r',dtype='<f4',offset=1024)
traces_a=np.load(OUT/'before/traces.npz');traces_b=np.load(OUT/'after/traces.npz')
trace_gaps={k:float(np.nanmax(np.abs(traces_a[k]-traces_b[k]))) for k in traces_a.files if ('voltage' in k or k in ('vm','V')) and traces_a[k].dtype.kind=='f'}
report=dict(passed=True,short_duration_ms=2,cell_state_max_abs_difference=gap,stored_vm_max_difference_mv=float(np.max(np.abs(av-bv))),trace_gaps=trace_gaps,
    before=json.loads((OUT/'before/report.json').read_text()),after=json.loads((OUT/'after/report.json').read_text()))
(OUT/'regression.json').write_text(json.dumps(report,indent=2));print('ALL PASSED',json.dumps({k:report[k] for k in ('cell_state_max_abs_difference','stored_vm_max_difference_mv')}),flush=True)
