"""Old-parameter GPB trajectories with shared PDE wrapper and controlled CG kernels."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import meshio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compare_ep_rectangles import mesh,timed
from afsi_torch.config import load_config
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.ep.cells import prepare_cell_model
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.jacobi import JacobiPCG
from afsi_torch.fluid.solvers import SolverOptions
from validation.torchcor_pde import load_core,TorchCorCG,PDEMethod,COMMIT

NAMES=['consistent_tc','nodal_tc','nodal_afsei']


def rhs(method,V,I,S):return method.rhs(V,I,S)


def prepare(method,item):
    return method.solver.calibrate(method.rhs(*item)) if isinstance(method.solver,TorchCorCG) else None


def check(method,v,item,options):
    b=method.rhs(*item);r=torch.linalg.vector_norm(b-method.matrix@v).item()
    ratio=r/max(options.atol,options.rtol*torch.linalg.vector_norm(b).item())
    if not torch.isfinite(v).all() or ratio>1:raise RuntimeError(f'true residual {ratio}')
    return ratio


def comparison(a,b):
    mask=np.isfinite(a['activation_s'])&np.isfinite(b['activation_s'])
    delta=(b['activation_s'][mask]-a['activation_s'][mask])*1000
    return {'activation_mean_abs_ms':float(abs(delta).mean()),'activation_max_abs_ms':float(abs(delta).max()),
            'activation_mean_signed_ms':float(delta.mean()),
            'snapshot_voltage_max_difference_mv':float(abs(a['voltage_snapshots_v']-b['voltage_snapshots_v']).max()*1000),
            'snapshot_voltage_rmse_mv':float(np.sqrt(np.mean((a['voltage_snapshots_v']-b['voltage_snapshots_v'])**2))*1000)}


def frozen(methods,inputs,dt,options):
    """Same saved V/I/S for all kernels. Includes old-style zero-I control."""
    report={}
    for scope in ['zero_ionic','actual_ionic']:
        items=[(v,torch.zeros_like(i) if scope=='zero_ionic' else i,s) for v,i,s in inputs]
        calibrations=[[prepare(m,item) for item in items] for m in methods]
        for k,m in enumerate(methods):
            for j,item in enumerate(items):
                v,_=m.step(*item,calibrations[k][j]);check(m,v,item,options)
            for _ in range(2):
                for j,item in enumerate(items):m.step(*item,calibrations[k][j])
        samples=[[] for _ in methods];counts=[[] for _ in methods]
        for repeat in range(3):
            for k in [(repeat+j)%len(methods) for j in range(len(methods))]:
                def block():
                    for _ in range(5):
                        for j,item in enumerate(items):
                            _,n=methods[k].step(*item,calibrations[k][j]);counts[k].append(n)
                _,seconds=timed(block);samples[k].append(seconds/(5*len(items))*1000)
        report[scope]={name:{'pde_mean_ms':float(np.mean(t)),'repeat_ms':t,'mean_iterations':float(np.mean(n)),
                            'timed_solves':75} for name,t,n in zip(NAMES,samples,counts)}
    return report


@torch.no_grad()
def run_case(dim,c,cfg,scalar,classes,root,primary_only=False):
    dest=root/f'{dim}d';dest.mkdir();X,cells=mesh(dim,c)
    coords=X.cpu().numpy();top=cells.cpu().numpy();dt=c['dt_ep_s'];steps=round(c['duration_s']/dt)
    options=SolverOptions(rtol=1e-10,atol=1e-13,max_iterations=2000,check_every=5)
    f=X.new_zeros((len(X),dim));s=f.clone();f[:,0]=1;s[:,1]=1
    ep=Monodomain(X,cells,f,s,parameters=cfg.parameters,numerics=MonodomainNumerics(backend='torch_pcg_jacobi'))
    assembler=classes[0]
    if dim==2:
        import importlib
        assembler=importlib.import_module('torchcor.core.assemble').Matrices2D
    (K,M),assembly_time=timed(lambda:assembler(X,cells,X.device,X.dtype).assemble_matrices(
        torch.diag(X.new_tensor(c['parameters']['conductivity_us_per_cm'][:dim]))))
    K,M=K.to_sparse_csr(),M.to_sparse_csr()
    gap=(K.to_sparse_coo()-ep.stiffness.to_sparse_coo()).coalesce().values().abs().max().item()/K.values().abs().max().item()
    if gap>1e-11:raise RuntimeError('stiffness mismatch')
    alpha=cfg.parameters.surface_to_volume_per_cm*cfg.parameters.capacitance_uf_per_cm2/dt
    Aconsistent=(.5*K.to_sparse_coo()+alpha*M.to_sparse_coo()).coalesce().to_sparse_csr()
    Anodal=(.5*K.to_sparse_coo()+alpha*ep.mass.to_sparse_coo()).coalesce().to_sparse_csr()
    diagonal=alpha*ep.mass_diagonal+.5*ep.stiffness_diagonal
    solvers=[TorchCorCG(Aconsistent,options,classes),TorchCorCG(Anodal,options,classes),JacobiPCG(Anodal,diagonal,options)]
    methods=[PDEMethod(X,mass,ep.mass_diagonal,A,cfg.parameters,dt,solver,nodal=nodal,stiffness=K,theta=.5)
             for mass,A,solver,nodal in zip([M,ep.mass,ep.mass],[Aconsistent,Anodal,Anodal],solvers,[False,True,True])]
    if primary_only:methods=methods[:2]
    zero=X.new_zeros(len(X));pulse=zero.masked_fill((X<=c['stimulus_extent_cm']+1e-12).all(1),c['stimulus_ua_cm2'])
    # The uniformly conditioned scalar state is copied identically to each node/method.
    from afsi_torch.ep.cells import GPBCell
    def initial():
        cell=GPBCell(dt,c['dt_ode_ms']);state=cell.initial(zero)
        state=replace(state,voltage_v=scalar.voltage_v.expand(len(X)).clone(),gpb=scalar.gpb.expand(len(X),-1).clone())
        return cell,state
    for m in methods:
        cell,state=initial();_,current=cell.advance_ode(state,0.)
        item=(state.voltage_v,current,pulse);cal=prepare(m,item)
        for _ in range(2):m.step(*item,cal)
    output={};data={};inputs=[];kind='tetra' if dim==3 else 'triangle'
    xyz=coords if dim==3 else np.column_stack((coords,np.zeros(len(coords))))
    for name,m in zip(NAMES,methods):
        folder=dest/name;folder.mkdir();cell,state=initial();activation=zero+float('nan')
        trace=[];snap=[];snapshot_times=[];pde=[];ode_seconds=[];counts=[];ratios=[];files=[]
        start=time.perf_counter()
        for step in range(steps+1):
            if step%200==0 or step==steps:
                values=state.voltage_v.cpu().numpy().copy();snap.append(values);snapshot_times.append(step*dt)
                file=f'ep_{step:05d}.vtu';meshio.write(folder/file,meshio.Mesh(xyz,[(kind,top)],point_data={
                    'voltage_mV':values*1000,'activation_ms':activation.cpu().numpy()*1000}))
                files.append((step*dt,file))
            trace.append(state.voltage_v[[0,len(X)-1]].cpu().numpy())
            if step==steps:break
            (ode,current),elapsed=timed(lambda:cell.advance_ode(state,step*dt));ode_seconds.append(elapsed)
            S=pulse if step<round(c['stimulus_duration_s']/dt) else zero
            item=(state.voltage_v,current,S)
            # Common preparation policy; calibration excluded from all PDE timings.
            calibration=prepare(m,item)
            (v,n),elapsed=timed(lambda:m.step(*item,calibration));pde.append(elapsed);counts.append(n)
            if step%100==0:ratios.append(check(m,v,item,options))
            if name=='consistent_tc' and step in [0,40,200,400,800]:inputs.append(tuple(t.clone() for t in item))
            crossing=torch.isnan(activation)&(state.voltage_v<0)&(v>=0)
            den=torch.where(crossing,v-state.voltage_v,torch.ones_like(v))
            activation=torch.where(crossing,step*dt-dt*state.voltage_v/den,activation)
            # TorchCor CG reuses self.x in subsequent solves. Publish an owned
            # state for every method so the previous time level stays immutable.
            state=replace(ode,voltage_v=v.clone())
            if (step+1)%600==0:print(dim,name,'time_ms',(step+1)*dt*1000,flush=True)
        data[name]={'X':coords,'cells':top,'activation_s':activation.cpu().numpy(),
                    'voltage_snapshots_v':np.array(snap),'snapshot_times_s':np.array(snapshot_times),
                    'traces_v':np.array(trace),'pde_step_s':np.array(pde),'ode_step_s':np.array(ode_seconds)}
        np.savez_compressed(folder/'fields.npz',**data[name])
        (folder/'ep.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1"><Collection>'+''.join(
            f'<DataSet timestep="{t}" file="{f}"/>' for t,f in files)+'</Collection></VTKFile>')
        output[name]={'pde_mean_ms':float(np.mean(pde)*1000),'pde_total_s':sum(pde),'ode_total_s':sum(ode_seconds),
                      'wall_with_output_and_calibration_s':time.perf_counter()-start,'mean_iterations':float(np.mean(counts)),
                      'max_residual_ratio':max(ratios),'activated_fraction':float(np.isfinite(data[name]['activation_s']).mean()),
                      'far_corner_activation_ms':float(data[name]['activation_s'][-1]*1000)}
    replay=frozen(methods,inputs,dt,options)
    result={'dimension':dim,'nodes':len(X),'cells':len(cells),'methods':output,'frozen_replay':replay,
            'mesh_sha256':hashlib.sha256(coords.tobytes()+top.tobytes()).hexdigest(),'relative_stiffness_gap':gap,
            'assembly_time_s':assembly_time,'spatial_discretization_difference':comparison(data[NAMES[0]],data[NAMES[1]])}
    if not primary_only:result['solver_implementation_difference']=comparison(data[NAMES[1]],data[NAMES[2]])
    (dest/'report.json').write_text(json.dumps(result,indent=2));return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--primary-only',action='store_true');p.add_argument('--initial-state-reference');a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    cfg=load_config(a.config,EPTissueConfig)
    if cfg.parameters.conductivity_us_per_cm[0]!=.1171 or cfg.initial.conditioning_s!=.2:
        raise ValueError('old controlled benchmark parameters required')
    root=Path(a.output);root.mkdir(exist_ok=False,parents=True)
    c={'lengths_cm':list(cfg.domain.lengths_cm),'spacing_cm':cfg.domain.spacing_cm,'dt_ep_s':cfg.time.dt_ep_s,
       'dt_ode_ms':cfg.time.dt_ode_ms,'duration_s':cfg.time.end_time_s,'parameters':cfg.parameters.__dict__,
       'stimulus_extent_cm':cfg.stimulus.maximum_cm[0],'stimulus_ua_cm2':cfg.stimulus.amplitude_ua_per_cm2,
       'stimulus_duration_s':cfg.stimulus.duration_s,'conditioning_s':cfg.initial.conditioning_s}
    (root/'config.json').write_text(json.dumps(c,indent=2));(root/'original_config.json').write_bytes(Path(a.config).read_bytes())
    classes,provenance=load_core(Path('.deps')/('torchcor-'+COMMIT));rows=[]
    try:
        if a.initial_state_reference:
            ref=Path(a.initial_state_reference)
            previous=json.loads((ref/'original_config.json').read_text())
            for key in ['parameters','initial','time']:
                if previous[key]!=json.loads(Path(a.config).read_text())[key]:
                    raise ValueError('conditioned initial state parameters differ')
            from afsi_torch.ep.tissue import initial_state
            scalar=initial_state(torch.zeros(1,device='cuda'))
            with np.load(ref/'conditioned_initial_state.npz',allow_pickle=False) as z:
                scalar=replace(scalar,voltage_v=torch.as_tensor(z['voltage_v'],device='cuda'),gpb=torch.as_tensor(z['gpb'],device='cuda'))
            initialization=json.loads((ref/'initialization.json').read_text())
            initialization['reused_from']=str(ref)
        else:
            cell,scalar=prepare_cell_model(cfg,torch.zeros(1,device='cuda'))
            initialization=cell.initialization_report
        (root/'initialization.json').write_text(json.dumps(initialization,indent=2))
        np.savez(root/'conditioned_initial_state.npz',voltage_v=scalar.voltage_v.cpu().numpy(),gpb=scalar.gpb.cpu().numpy())
        for dim in [2,3]:
            rows.append(run_case(dim,c,cfg,scalar,classes,root,a.primary_only));(root/'progress.json').write_text(json.dumps(rows,indent=2))
        (root/'report.json').write_text(json.dumps({'status':'completed','cases':rows,'torchcor':provenance,
            'methods':NAMES[:2] if a.primary_only else NAMES,'scope':'same old parameters, GPB and PDEMethod wrapper; same TorchCor CG for spatial pair',
            'timing':'RHS + checks + CG; calibration/setup/ODE/I/O excluded; frozen replays interleaved, trajectories sequential',
            'frozen_scope':'common evolving consistent-GPB snapshots at 0/2/10/20/40 ms; actual-I and zero-I controls; not the historical snapshot files'},indent=2))
    except Exception as e:
        (root/'failure.json').write_text(json.dumps({'error':repr(e)}));raise


if __name__=='__main__':main()
