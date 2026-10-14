"""Two-method PDE experiment: native TorchCor vs PyTorch nodal FEM + AMG."""
import argparse,gc,json,os,sys
from pathlib import Path
from statistics import median
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from afsi_torch.config import load_config
from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.ep import Monodomain,MonodomainNumerics
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.fluid.solvers import SolverOptions
from validation.torchcor_native_pde import load_native_electric,NativeTorchCorPDE
from demo.ep_slab.benchmark_pde_methods import synchronized_time,max_sparse_relative_difference


def data(folder,settings):
    cfg=load_config(folder/'config_resolved.json',EPTissueConfig)
    if json.loads((folder/'report.json').read_text())['status']!='completed':raise ValueError('completed source required')
    with np.load(folder/'mesh.npz',allow_pickle=False) as saved:
        X=torch.as_tensor(saved['coordinates_cm'],dtype=torch.float64,device='cuda')
        cells=torch.as_tensor(saved['cells'],dtype=torch.int64,device='cuda')
        mask=torch.as_tensor(saved['stimulus_region'],dtype=torch.bool,device='cuda')
    with np.load(folder/'voltage_snapshots.npz',allow_pickle=False) as saved:
        times=saved['time_s']*1000
        ids=[int(np.argmin(abs(times-t))) for t in settings['snapshot_times_ms']]
        if any(abs(times[i]-t)>1e-8 for i,t in zip(ids,settings['snapshot_times_ms'])):raise ValueError('exact snapshots required')
        volts=torch.as_tensor(saved['voltage_v'][ids],dtype=torch.float64,device='cuda')
    f=X.new_tensor(cfg.domain.fiber).expand(len(X),-1);s=X.new_tensor(cfg.domain.sheet).expand(len(X),-1)
    zero=X.new_zeros(len(X));stim=zero.masked_fill(mask,cfg.stimulus.amplitude_ua_per_cm2)
    inputs=[(V,zero,stim if 0<=t<cfg.stimulus.duration_s*1000 else zero)
            for V,t in zip(volts,settings['snapshot_times_ms'])]
    options=SolverOptions(rtol=settings['rtol'],atol=settings['atol'],max_iterations=settings['max_iterations'])
    return cfg,X,cells,f,s,inputs,options


def make_ep(d,profile):
    cfg,X,cells,f,s,inputs,options=d
    ep=Monodomain(X,cells,f,s,parameters=cfg.parameters,numerics=MonodomainNumerics(
        solver=options,time_scheme='crank_nicolson',amg_profile=profile))
    ep.prepare(cfg.time.dt_ep_s)
    return ep


def rhs_ep(ep,item,dt):
    V,I,S=item;p=ep.parameters
    return ep.mass_diagonal*(p.surface_to_volume_per_cm*p.capacitance_uf_per_cm2/dt*V+
                            p.surface_to_volume_per_cm*(S-I))-.5*(ep.stiffness@V)


def acceptance(A,value,rhs,options):
    norm=torch.linalg.vector_norm(rhs).item();tolerance=max(options.atol,options.rtol*norm)
    ratio=torch.linalg.vector_norm(rhs-A@value).item()/tolerance
    if not torch.isfinite(value).all() or ratio>1:raise RuntimeError('common true PDE residual failed')
    return ratio


def warm(call,n,settings):
    for _ in range(settings['warmup_rounds']):
        for j in range(n):call(j)


def block(call,n,settings):
    def replay():
        for _ in range(settings['rounds_per_repeat']):
            for j in range(n):call(j)
    _,seconds=synchronized_time(replay)
    return seconds/(n*settings['rounds_per_repeat'])*1000


def pilot(d,settings):
    dt=d[0].time.dt_ep_s;inputs=d[5];options=d[6];rows=[]
    for profile in settings['amg_profiles']:
        ep=make_ep(d,profile);A=ep._systems[dt][1]
        ratios=[]
        for item in inputs:
            value,_=ep.step(*item,dt);ratios.append(acceptance(A,value,rhs_ep(ep,item,dt),options))
        call=lambda j:ep.step(*inputs[j],dt)
        warm(call,len(inputs),settings)
        times=[block(call,len(inputs),settings) for _ in range(settings['repeats'])]
        rows.append(dict(amg_profile=profile,mean_ms=sum(times)/len(times),repeat_ms=times,max_residual_ratio=max(ratios)))
        ep.close();del ep;gc.collect();torch.cuda.empty_cache()
    selected=min(rows,key=lambda r:r['mean_ms'])['amg_profile']
    return dict(pilot_case=settings['pilot_case'],selection='lowest mean time among predeclared valid AMG profiles',
                candidates=rows,selected_profile=selected)


def paired(d,settings,classes,electric,profile):
    cfg,X,cells,f,s,inputs,options=d;dt=cfg.time.dt_ep_s
    dirs=torch.stack((f[0],s[0],torch.linalg.cross(f[0],s[0])))
    sigma=torch.einsum('a,ai,aj->ij',X.new_tensor(cfg.parameters.conductivity_us_per_cm),dirs,dirs)
    Matrices3D=classes[0]
    (K,M),native_assembly=synchronized_time(lambda:Matrices3D(X,cells,X.device,X.dtype).assemble_matrices(sigma))
    K,M=K.to_sparse_csr(),M.to_sparse_csr()
    native,native_setup=synchronized_time(lambda:NativeTorchCorPDE(M,K,cfg.parameters,dt,options,classes,electric))
    ep,afsei_preparation=synchronized_time(lambda:make_ep(d,profile))
    gap=max_sparse_relative_difference(ep.stiffness,K)
    mass_gap=((torch.sparse.sum(M.to_sparse_coo(),dim=1).to_dense()-ep.mass_diagonal).abs().max()/ep.mass_diagonal.abs().max()).item()
    if max(gap,mass_gap)>1e-10:raise RuntimeError('geometry/conductivity equivalence failed')
    prepared=[native.prepare_input(*item) for item in inputs]
    A_native=(native.A.to_sparse_coo()*(1/native.native_equation_scale)).coalesce().to_sparse_csr();A_ours=ep._systems[dt][1]
    def native_call(j):return native.step(prepared[j])
    def afsei_call(j):return ep.step(*inputs[j],dt)
    ratios=[[],[]];iterations=[[],[]]
    for j,item in enumerate(inputs):
        V,count=native_call(j);ratios[0].append(acceptance(A_native,V/1000,prepared[j][3],options));iterations[0].append(count)
        V,info=afsei_call(j);ratios[1].append(acceptance(A_ours,V,rhs_ep(ep,item,dt),options));iterations[1].append(info.iterations)
    for call in (native_call,afsei_call):warm(call,len(inputs),settings)
    samples=[[],[]]
    for repeat in range(settings['repeats']):
        for k in (range(2) if repeat%2==0 else (1,0)):
            samples[k].append(block((native_call,afsei_call)[k],len(inputs),settings))
    rows=[dict(method=name,mean_pde_ms=sum(times)/len(times),median_pde_ms=median(times),repeat_pde_ms=times,
               mean_iterations=sum(count)/len(count),max_true_residual_ratio=max(r),
               timed_solves=len(inputs)*settings['rounds_per_repeat']*settings['repeats'])
          for name,times,count,r in zip(('torchcor_native_p1','afsei_pytorch_nodal_amgx'),samples,iterations,ratios)]
    result=dict(case=cfg.case_name,spacing_mm=cfg.domain.spacing_cm*10,node_count=len(X),cell_count=len(cells),
        dt_ep_ms=dt*1000,time_scheme='crank_nicolson',amg_profile=profile,methods=rows,
        speedup=rows[0]['mean_pde_ms']/rows[1]['mean_pde_ms'],
        native_assembly_s=native_assembly,native_setup_s=native_setup,afsei_preparation_s=afsei_preparation,
        relative_stiffness_difference=gap,relative_lumped_mass_difference=mass_gap)
    ep.close();del ep,native,K,M;gc.collect();torch.cuda.empty_cache()
    return result


def plot(folder):
    os.environ.setdefault('MPLCONFIGDIR',str(folder.resolve()/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((folder/'report.json').read_text());cases=report['cases'];x=np.arange(len(cases))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for k,(name,color) in enumerate((('TorchCor native P1-Galerkin','#64748b'),('AFSEI PyTorch nodal FEM + AMG','#f97316'))):
        vals=[c['methods'][k]['mean_pde_ms'] for c in cases]
        errors=np.array([[v-min(c['methods'][k]['repeat_pde_ms']),max(c['methods'][k]['repeat_pde_ms'])-v]
                          for v,c in zip(vals,cases)]).T
        bars=axes[0].bar(x+(k-.5)*.35,vals,.35,label=name,color=color,yerr=errors,capsize=3)
        axes[0].bar_label(bars,fmt='%.2f ms',fontsize=9,padding=4)
    labels=[f'{c["spacing_mm"]:g} mm\n{c["node_count"]:,} nodes' for c in cases]
    axes[0].set(ylabel='PDE time per step (ms)',title='Two-method comparison: same benchmark',xticks=x,xticklabels=labels)
    axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.2);axes[0].legend(fontsize=8)
    bars=axes[1].bar(x,[c['speedup'] for c in cases],color=['#f97316' if c['speedup']>=1 else '#64748b' for c in cases])
    axes[1].bar_label(bars,fmt='%.2fx',padding=4);axes[1].axhline(1,color='#64748b',ls='--')
    axes[1].set(ylabel='TorchCor native time / AFSEI time',title='>1: AFSEI faster; <1: TorchCor faster',xticks=x,xticklabels=labels,
                ylim=(0,max(1.2,max(c['speedup'] for c in cases)*1.25)))
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'RTX 4090 | FP64 | CN 0.05 ms | same true residual | AMG {report["pilot"]["selected_profile"]}')
    fig.savefig(folder/'native_pair_pde.png',dpi=200);fig.savefig(folder/'native_pair_pde.pdf');plt.close(fig)


@torch.no_grad()
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/native_pair_benchmark.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--resume-experiment',action='store_true',help='reuse matching completed pilot/cases')
    a=parser.parse_args(argv)
    if a.plot_only:return plot(a.output)
    settings=json.loads(a.config.read_text())
    if settings['demo']!='ep-native-pair-benchmark' or settings['schema_version']!=1 or settings['time_scheme']!='crank_nicolson':
        parser.error('declared native CN benchmark required')
    if not torch.cuda.is_available():parser.error('actual CUDA required')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume_experiment:parser.error('new output or explicit resume required')
    a.output.mkdir(parents=True,exist_ok=True);torch.set_default_dtype(torch.float64)
    classes,electric,provenance=load_native_electric(settings['torchcor_source_dir'])
    saved=a.output/'config_resolved.json'
    if saved.exists() and json.loads(saved.read_text())!=settings:parser.error('resume cannot change conditions')
    atomic_json(saved,settings)
    root=Path(settings['source_results_root'])
    print('Pilot: optimize declared AMG profiles on the fixed 0.25mm case',flush=True)
    if (a.output/'pilot.json').exists():
        selected=json.loads((a.output/'pilot.json').read_text());print('Reuse completed AMG pilot',flush=True)
    else:selected=pilot(data(root/settings['pilot_case'],settings),settings)
    atomic_json(a.output/'pilot.json',selected);print(json.dumps(selected,indent=2),flush=True)
    cases=[]
    for name in settings['cases']:
        print(f'Native pair: {name}',flush=True)
        cached=a.output/f'{name}_pair.json'
        if cached.exists():
            result=json.loads(cached.read_text());print('Reuse completed paired case',flush=True)
        else:
            result=paired(data(root/name,settings),settings,classes,electric,selected['selected_profile'])
            atomic_json(cached,result)
        cases.append(result);atomic_json(a.output/'progress.json',dict(status='running',cases=cases))
        print(json.dumps(result,indent=2),flush=True)
    report=dict(status='completed',gpu=torch.cuda.get_device_name(),dtype='float64',settings=settings,
        torchcor=provenance,pilot=selected,cases=cases,
        scope='native TorchCor PDE statements vs actual AFSEI Monodomain.step; no nodal+TorchCor control',
        timing='warm synchronized wall-clock; RHS and solver; excludes ODE, I/O and one-time preparation',
        source='same saved voltage snapshots, zero ionic current, common prescribed pulse; not a new physiological trajectory',
        common_precision='same true residual tolerance, initial prescribed Vm; linear extrapolation disabled for frozen replay',
        input_units='native TorchCor mV/ms representation vs AFSEI V/s; equivalent unit conversion verified',
        source_field_checks='native baseline has only upstream checks; AFSEI production input and true residual checks retained')
    atomic_json(a.output/'report.json',report);plot(a.output)
    print(json.dumps(dict(status='completed',output=str(a.output)),indent=2),flush=True)


if __name__=='__main__':main()
