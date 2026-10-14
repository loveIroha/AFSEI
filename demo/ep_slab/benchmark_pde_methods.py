"""PDE-only TorchCor P1-Galerkin / AFSEI nodal FEM + AMG comparison."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from afsi_torch.config import load_config
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.ep import Monodomain,MonodomainNumerics
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.ep.amgx import AmgXPCG
from afsi_torch.fluid.solvers import SolverOptions
from validation.torchcor_pde import load_core,TorchCorCG,PDEMethod,COMMIT

LABELS=('torchcor_p1_jacobi','p1_amgx','nodal_torchcor_jacobi','afsei_nodal_amgx')


def synchronized_time(fn):
    torch.cuda.synchronize();start=perf_counter();value=fn();torch.cuda.synchronize()
    return value,perf_counter()-start


def max_sparse_relative_difference(a,b):
    delta=(a.to_sparse_coo()-b.to_sparse_coo()).coalesce().values()
    return delta.abs().max().item()/b.values().abs().max().item() if len(delta) else 0.


@torch.no_grad()
def benchmark_case(folder,settings,classes):
    config=load_config(folder/'config_resolved.json',EPTissueConfig)
    report=json.loads((folder/'report.json').read_text())
    if report['status']!='completed' or config.ep_numerics.discretization!='nodal_p1':
        raise ValueError('completed nodal reference snapshots required')
    dt=config.time.dt_ep_s;p=config.parameters
    scheme=settings.get('time_scheme','backward_euler')
    theta=.5 if scheme=='crank_nicolson' else 1.
    with np.load(folder/'mesh.npz',allow_pickle=False) as saved:
        X=torch.as_tensor(saved['coordinates_cm'],device='cuda',dtype=torch.float64)
        cells=torch.as_tensor(saved['cells'],device='cuda',dtype=torch.int64)
        mask=torch.as_tensor(saved['stimulus_region'],device='cuda',dtype=torch.bool)
    with np.load(folder/'voltage_snapshots.npz',allow_pickle=False) as saved:
        all_times=saved['time_s']*1000
        ids=[int(np.argmin(abs(all_times-t))) for t in settings['snapshot_times_ms']]
        if any(abs(all_times[i]-t)>1e-8 for i,t in zip(ids,settings['snapshot_times_ms'])):
            raise ValueError('requested snapshot is not saved exactly')
        volts=torch.as_tensor(saved['voltage_v'][ids],device='cuda',dtype=torch.float64)
    f=X.new_tensor(config.domain.fiber).expand(len(X),-1)
    s=X.new_tensor(config.domain.sheet).expand(len(X),-1)
    n=torch.linalg.cross(f[0],s[0]);directions=torch.stack((f[0],s[0],n))
    sigma=torch.einsum('a,ai,aj->ij',X.new_tensor(p.conductivity_us_per_cm),directions,directions)
    alpha=p.surface_to_volume_per_cm*p.capacitance_uf_per_cm2/dt
    options=SolverOptions(rtol=settings['rtol'],atol=settings['atol'],max_iterations=settings['max_iterations'],check_every=5)
    # Native TorchCor assembly converts Python lists back to default-dtype tensors.
    # FP64 default is explicitly common to both methods to avoid downcasting.
    Matrices3D=classes[0]
    (K,M),tc_assembly=synchronized_time(lambda:Matrices3D(X,cells,X.device,X.dtype).assemble_matrices(sigma))
    K,M=K.to_sparse_csr(),M.to_sparse_csr()
    A=(theta*K.to_sparse_coo()+alpha*M.to_sparse_coo()).coalesce().to_sparse_csr()
    diag=torch.zeros(len(X),device=X.device,dtype=X.dtype)
    coo=M.to_sparse_coo();idx=coo.indices();select=idx[0]==idx[1]
    diag[idx[0,select]]=coo.values()[select]
    numerics=MonodomainNumerics('nodal_p1','amgx_pcg_amg',options,time_scheme=scheme)
    ep,nodal_assembly=synchronized_time(lambda:Monodomain(X,cells,f,s,parameters=p,numerics=numerics))
    _,nodal_setup=synchronized_time(lambda:ep.prepare(dt))
    stiffness_gap=max_sparse_relative_difference(ep.stiffness,K)
    lumped=torch.sparse.sum(M.to_sparse_coo(),dim=1).to_dense()
    mass_gap=((lumped-ep.mass_diagonal).abs().max()/ep.mass_diagonal.abs().max()).item()
    if max(stiffness_gap,mass_gap)>1e-10:
        raise RuntimeError('constant-direction TorchCor / AFSEI assembly equivalence check failed')
    consistent_cg,tc_setup=synchronized_time(lambda:TorchCorCG(A,options,classes))
    consistent_amg,consistent_amg_setup=synchronized_time(lambda:AmgXPCG(A,options))
    nodal_matrix=ep._systems[dt][1]
    nodal_cg,nodal_cg_setup=synchronized_time(lambda:TorchCorCG(nodal_matrix,options,classes))
    methods=[PDEMethod(X,M,diag,A,p,dt,consistent_cg,stiffness=K,theta=theta),
             PDEMethod(X,M,diag,A,p,dt,consistent_amg,stiffness=K,theta=theta),
             PDEMethod(X,ep.mass,ep.mass_diagonal,nodal_matrix,p,dt,nodal_cg,nodal=True,
                       stiffness=ep.stiffness,theta=theta),ep]
    setups=[tc_setup,consistent_amg_setup,nodal_cg_setup,nodal_setup]
    zero=X.new_zeros(len(X));stim=zero.masked_fill(mask,config.stimulus.amplitude_ua_per_cm2)
    inputs=[(V,zero,stim if 0<=t<config.stimulus.duration_s*1000 else zero)
            for V,t in zip(volts,settings['snapshot_times_ms'])]
    calibrations=[]
    for method in methods:
        calibrations.append([method.solver.calibrate(method.rhs(*item)) for item in inputs]
                            if isinstance(method,PDEMethod) and isinstance(method.solver,TorchCorCG) else [None]*len(inputs))
    def call(k,j):
        if k==3:
            value,info=methods[k].step(*inputs[j],dt)
            return value,info.iterations
        return methods[k].step(*inputs[j],calibrations[k][j])
    # Validate every representative RHS outside timings; compare same equations.
    accepted=[];ratios=[];iteration_samples=[]
    for k,method in enumerate(methods):
        values=[];errors=[];counts=[]
        for j in range(len(inputs)):
            value,count=call(k,j)
            if k==3:
                source=alpha*inputs[j][0]+p.surface_to_volume_per_cm*(inputs[j][2]-inputs[j][1])
                rhs=ep.mass_diagonal*source
                if theta!=1.:rhs=rhs-(1-theta)*torch.sparse.mm(ep.stiffness,inputs[j][0][:,None]).flatten()
            else:rhs=method.rhs(*inputs[j])
            matrix=nodal_matrix if k>=2 else A
            tolerance=max(options.atol,options.rtol*torch.linalg.vector_norm(rhs).item())
            ratio=torch.linalg.vector_norm(rhs-torch.sparse.mm(matrix,value[:,None]).flatten()).item()/tolerance
            if not torch.isfinite(value).all() or ratio>1:raise RuntimeError(f'{LABELS[k]} true-residual acceptance failed')
            values.append(value.clone());errors.append(ratio);counts.append(count)
        accepted.append(values);ratios.append(max(errors));iteration_samples.append(sum(counts)/len(counts))
    gaps=[max((a-b).abs().max().item()*1000 for a,b in zip(accepted[0],accepted[1])),
          max((a-b).abs().max().item()*1000 for a,b in zip(accepted[2],accepted[3]))]
    if max(gaps)>1e-5:raise RuntimeError('same-equation solver solutions differ excessively')
    for k in range(4):
        for _ in range(settings['warmup_rounds']):
            for j in range(len(inputs)):call(k,j)
    samples=[[] for _ in methods]
    count=len(inputs)*settings['rounds_per_repeat']
    for repeat in range(settings['repeats']):
        for k in [(repeat+i)%4 for i in range(4)]:
            def block():
                for _ in range(settings['rounds_per_repeat']):
                    for j in range(len(inputs)):call(k,j)
            _,elapsed=synchronized_time(block)
            samples[k].append(elapsed/count)
    variants=[dict(method=label,mean_pde_ms=1000*sum(times)/len(times),median_pde_ms=1000*median(times),
                   repeat_pde_ms=[1000*t for t in times],solver_setup_s=setup,
                   representative_mean_iterations=iterations,max_true_residual_to_tolerance=ratio,
                   timed_solves=len(times)*count)
              for label,times,setup,iterations,ratio in zip(LABELS,samples,setups,iteration_samples,ratios)]
    result=dict(case=folder.name,spacing_mm=config.domain.spacing_cm*10,node_count=len(X),cell_count=len(cells),
        dt_ep_ms=dt*1000,time_scheme=scheme,theta=theta,diffusion_cm2_s=report['diffusion_cm2_s'],variants=variants,
        warm_pde_speedup=variants[0]['mean_pde_ms']/variants[-1]['mean_pde_ms'],
        torchcor_assembly_s=tc_assembly,afsei_nodal_assembly_s=nodal_assembly,
        relative_stiffness_difference=stiffness_gap,relative_nodal_mass_difference=mass_gap,
        same_consistent_equation_solver_difference_mv=gaps[0],same_nodal_equation_solver_difference_mv=gaps[1],
        observed_discretization_difference_mv=max((a-b).abs().max().item()*1000 for a,b in zip(accepted[0],accepted[3])))
    consistent_amg.close();ep.close()
    del methods,accepted,volts,consistent_cg,consistent_amg,nodal_cg,ep,X,cells,K,M,A
    gc.collect();torch.cuda.empty_cache()
    return result


def plot_report(folder):
    folder=Path(folder);report=json.loads((folder/'report.json').read_text())
    os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=report['cases'];x=np.arange(len(cases));width=.18
    scheme=report['settings'].get('time_scheme','backward_euler')
    scheme_label='Crank-Nicolson' if scheme=='crank_nicolson' else 'backward Euler'
    colors=['#64748b','#a78bfa','#38bdf8','#f97316']
    names=['TorchCor P1 + Jacobi-PCG','P1 + AMG-PCG','Nodal P1 + TorchCor Jacobi-PCG','AFSEI nodal P1 + AMG-PCG']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for k,(name,color) in enumerate(zip(names,colors)):
        vals=[c['variants'][k]['mean_pde_ms'] for c in cases]
        repeats=[c['variants'][k]['repeat_pde_ms'] for c in cases]
        bounds=np.array([[min(r),max(r)] for r in repeats])
        errors=np.stack((np.array(vals)-bounds[:,0],bounds[:,1]-np.array(vals)))
        bars=axes[0].bar(x+(k-1.5)*width,vals,width,label=name,color=color,yerr=errors,capsize=3)
        axes[0].bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
    axes[0].set(ylabel='PDE time per step (ms)',title='Warm PDE step: RHS + linear solve',xticks=x,
                xticklabels=[f'{c["spacing_mm"]:g} mm\n{c["node_count"]:,} nodes' for c in cases])
    axes[0].legend(fontsize=8,loc='upper left');axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.27)
    bars=axes[1].bar(x,[c['warm_pde_speedup'] for c in cases],color=colors[-1],width=.55)
    axes[1].bar_label(bars,fmt='%.2fx',padding=4)
    axes[1].axhline(1,color='#64748b',ls='--',lw=1)
    axes[1].set(ylabel='TorchCor P1 time / AFSEI nodal + AMG time',title='Measured PDE speed ratio',xticks=x,
                xticklabels=[f'{c["spacing_mm"]:g} mm' for c in cases])
    axes[1].set_ylim(0,max(1.15,max(c['warm_pde_speedup'] for c in cases)*1.25))
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'RTX 4090 | FP64 | {scheme_label} 0.05 ms | common true residual tolerance')
    fig.savefig(folder/'pde_methods_comparison.png',dpi=200)
    fig.savefig(folder/'pde_methods_comparison.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for offset,k in ((-.18,0),(.18,3)):
        vals=[c['variants'][k]['mean_pde_ms'] for c in cases]
        repeats=[c['variants'][k]['repeat_pde_ms'] for c in cases]
        errors=np.array([[v-min(r),max(r)-v] for v,r in zip(vals,repeats)]).T
        bars=axes[0].bar(x+offset,vals,.34,label=names[k],color=colors[k],yerr=errors,capsize=3)
        axes[0].bar_label(bars,fmt='%.2f ms',fontsize=9,padding=4)
    axes[0].set(ylabel='PDE time per step (ms)',title='P1-Galerkin vs nodal FEM + AMG',xticks=x,
        xticklabels=[f'{c["spacing_mm"]:g} mm\n{c["node_count"]:,} nodes' for c in cases])
    axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.18);axes[0].legend(fontsize=8)
    ratios=[c['warm_pde_speedup'] for c in cases]
    bars=axes[1].bar(x,ratios,color=[colors[3] if v>=1 else colors[0] for v in ratios],width=.55)
    axes[1].bar_label(bars,fmt='%.2fx',padding=4)
    axes[1].axhline(1,color='#64748b',ls='--',lw=1)
    axes[1].set(ylabel='TorchCor P1 time / AFSEI nodal + AMG time',title='>1: AFSEI faster; <1: TorchCor faster',
                xticks=x,xticklabels=[f'{c["spacing_mm"]:g} mm' for c in cases],
                ylim=(0,max(1.15,max(ratios)*1.25)))
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'PDE only | RTX 4090 | FP64 | {scheme_label} dt = 0.05 ms | warm averaged wall time')
    fig.savefig(folder/'pde_main_comparison.png',dpi=200)
    fig.savefig(folder/'pde_main_comparison.pdf');plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/pde_methods_benchmark.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args(argv)
    if args.plot_only:return plot_report(args.output)
    settings=json.loads(args.config.read_text())
    expected={'demo','schema_version','time_scheme','torchcor_commit','torchcor_source_dir','source_results_root','cases',
              'snapshot_times_ms','warmup_rounds','rounds_per_repeat','repeats','max_iterations','rtol','atol'}
    if (set(settings)!=expected or settings['demo']!='ep-pde-methods-benchmark' or
            settings['schema_version']!=1 or settings['torchcor_commit']!=COMMIT):
        parser.error('use the pinned declared benchmark schema')
    if settings['time_scheme'] not in ('backward_euler','crank_nicolson'):
        parser.error('use backward_euler or crank_nicolson')
    if any(type(settings[k]) is not int or settings[k]<1 for k in ('warmup_rounds','rounds_per_repeat','repeats','max_iterations')):
        parser.error('positive integer sample counts required')
    if not torch.cuda.is_available():parser.error('actual CUDA required')
    if args.output.exists() and any(args.output.iterdir()):parser.error('new output directory required')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_default_dtype(torch.float64)
    classes,provenance=load_core(settings['torchcor_source_dir'])
    atomic_json(args.output/'config_resolved.json',settings)
    cases=[]
    for name in settings['cases']:
        print(f'Benchmark PDE: {name}',flush=True)
        result=benchmark_case(Path(settings['source_results_root'])/name,settings,classes)
        cases.append(result);atomic_json(args.output/'progress.json',dict(status='running',cases=cases))
        print(json.dumps(result,indent=2),flush=True)
    report=dict(status='completed',gpu=torch.cuda.get_device_name(),dtype='float64',torch_version=torch.__version__,
        torchcor=provenance,settings=settings,cases=cases,
        timing_scope='synchronized wall-clock block mean: finite-input checks + source/RHS + linear solver; no ODE, I/O or plots',
        input_scope='saved GPB voltage snapshots; zero ionic current; common corner stimulus active only at t=0 snapshot',
        true_residual_rule='norm(b-Ax) <= max(atol, rtol*norm(b)); all representative inputs independently verified',
        initial_guess='same prescribed snapshot voltage for all methods; TorchCor linear extrapolation disabled',
        time_scheme=settings['time_scheme'],theta=.5 if settings['time_scheme']=='crank_nicolson' else 1.,
        cold_scope='assembly/setup recorded separately; not included in warm PDE speed ratio',
        waveform_accuracy_or_full_EP_speed_established=False)
    atomic_json(args.output/'report.json',report);plot_report(args.output)
    print(json.dumps(dict(status='completed',output=str(args.output)),indent=2),flush=True)


if __name__=='__main__':main()
