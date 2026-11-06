"""Full GPB trajectories: AFSEI nodal/CN versus pinned TorchCor consistent/CN."""
import argparse, hashlib, importlib, json, os, sys, time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import meshio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.cells import GPBCell
from afsi_torch.geometry.tissue_slab import structured_tissue_slab
from afsi_torch.fluid.solvers import SolverOptions
from validation.torchcor_native_pde import load_native_electric,NativeTorchCorPDE
from validation.torchcor_pde import COMMIT


def timed(fn):
    torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize()
    return v,time.perf_counter()-t


def mesh(dim,c):
    h=c['spacing_cm']; lengths=c['lengths_cm'][:dim]
    if dim==3:
        m=structured_tissue_slab(lengths,h,device='cuda');return m.X,m.cells
    nx,ny=[round(v/h) for v in lengths]
    X=torch.stack(torch.meshgrid(torch.linspace(0,lengths[0],nx+1,device='cuda'),
                                torch.linspace(0,lengths[1],ny+1,device='cuda'),indexing='ij'),-1).reshape(-1,2)
    base=(torch.arange(nx,device='cuda')[:,None]*(ny+1)+torch.arange(ny,device='cuda')).flatten()
    offsets=torch.tensor([[0,ny+1,ny+2],[0,ny+2,1]],device='cuda')
    return X,(base[:,None,None]+offsets).reshape(-1,3)


@torch.no_grad()
def run_case(dim,c,root,classes,electric):
    folder=root/f'{dim}d';folder.mkdir()
    X,cells=mesh(dim,c);coords=X.cpu().numpy();top=cells.cpu().numpy()
    digest=hashlib.sha256(coords.tobytes()+top.tobytes()).hexdigest()
    dt=c['dt_ep_s'];steps=round(c['duration_s']/dt)
    opts=SolverOptions(rtol=c['rtol'],atol=c['atol'],max_iterations=2000,check_every=5)
    p=MonodomainParameters(**c['parameters'])
    f=X.new_zeros((len(X),dim));s=f.clone();f[:,0]=1;s[:,1]=1
    ep,ours_setup=timed(lambda:Monodomain(X,cells,f,s,parameters=p,numerics=MonodomainNumerics(
        backend=c['nodal_backend'],solver=opts,time_scheme='crank_nicolson')))
    _,extra=timed(lambda:ep.prepare(dt));ours_setup+=extra
    cls=classes[0] if dim==3 else importlib.import_module('torchcor.core.assemble').Matrices2D
    sigma=torch.diag(X.new_tensor(c['parameters']['conductivity_us_per_cm'][:dim]))
    (K,M),native_assembly=timed(lambda:cls(X,cells,X.device,X.dtype).assemble_matrices(sigma))
    K,M=K.to_sparse_csr(),M.to_sparse_csr()
    native,native_setup=timed(lambda:NativeTorchCorPDE(M,K,p,dt,opts,classes,electric))
    gap=(K.to_sparse_coo()-ep.stiffness.to_sparse_coo()).coalesce().values().abs().max().item()/K.values().abs().max().item()
    if gap>1e-11:raise RuntimeError('diffusion operator mismatch')
    # A corner BOX is held fixed across methods; zero-flux on every boundary.
    mask=(X<=c['stimulus_extent_cm']+1e-12).all(1)
    pulse=X.new_zeros(len(X)).masked_fill(mask,c['stimulus_ua_cm2']);zero=torch.zeros_like(pulse)
    probes=torch.stack([torch.linalg.vector_norm(X-X.new_tensor(lengths),dim=1).argmin()
                        for lengths in [tuple(0 for _ in range(dim)),tuple(c['lengths_cm'][:dim])]])
    outputs={}; arrays={};kind='tetra' if dim==3 else 'triangle'
    xyz=coords if dim==3 else np.column_stack((coords,np.zeros(len(coords))))
    # Compile and warm both paths on disposable states, not the simulated trajectory.
    warm=GPBCell(dt,c['dt_ode_ms']);state=warm.initial(zero)
    warm.advance_ode(state,0.)
    ep.step(state.voltage_v,zero,pulse,dt)
    prepared=native.prepare_input(state.voltage_v,zero,pulse);native.step(prepared)
    for method in ['torchcor_cg','afsei_nodal']:
        cell=GPBCell(dt,c['dt_ode_ms']);state=cell.initial(zero)
        activation=X.new_full((len(X),),float('nan'));trace=[];snap=[];pde_s=[];ode_s=[];iterations=[];ratios=[];files=[]
        dest=folder/method;dest.mkdir();start=time.perf_counter()
        for i in range(steps+1):
            if i%200==0 or i==steps:
                v=state.voltage_v.cpu().numpy().copy();snap.append(v)
                name=f'ep_{i:05d}.vtu';meshio.write(dest/name,meshio.Mesh(xyz,[(kind,top)],point_data={
                    'voltage_mV':v*1000,'activation_ms':activation.cpu().numpy()*1000,'stimulus':mask.cpu().numpy().astype('uint8')}))
                files.append((i*dt,name))
            trace.append(state.voltage_v[probes].cpu().numpy())
            if i==steps:break
            (ode,current),sec=timed(lambda:cell.advance_ode(state,i*dt));ode_s.append(sec)
            stimulus=pulse if i<round(c['stimulus_duration_s']/dt) else zero
            if method=='afsei_nodal':
                (v,info),sec=timed(lambda:ep.step(state.voltage_v,current,stimulus,dt));count=info.iterations
                A=ep._systems[dt][1]
            else:
                # Include native RHS calibration/unit conversion in reported PDE time.
                def solve():
                    prepared=native.prepare_input(state.voltage_v,current,stimulus)
                    value,n=native.step(prepared);return value/1000,n
                (v,count),sec=timed(solve)
                A=(native.A.to_sparse_coo()*(1/native.native_equation_scale)).coalesce().to_sparse_csr() if i%100==0 else None
            pde_s.append(sec);iterations.append(count)
            if i%100==0:
                mass=ep.mass if method=='afsei_nodal' else M
                stiff=ep.stiffness if method=='afsei_nodal' else K
                b=mass@(p.surface_to_volume_per_cm/dt*state.voltage_v+p.surface_to_volume_per_cm*(stimulus-current))-.5*(stiff@state.voltage_v)
                ratio=(torch.linalg.vector_norm(b-A@v)/max(opts.atol,opts.rtol*torch.linalg.vector_norm(b).item())).item()
                if ratio>1 or not torch.isfinite(v).all():raise RuntimeError(f'{method} residual {ratio}')
                ratios.append(ratio)
            crossing=torch.isnan(activation)&(state.voltage_v<0)&(v>=0)
            den=torch.where(crossing,v-state.voltage_v,torch.ones_like(v))
            activation=torch.where(crossing,i*dt+dt*(-state.voltage_v)/den,activation)
            state=replace(ode,voltage_v=v)
            if (i+1)%400==0:print(dim,method,'ms',(i+1)*dt*1000,flush=True)
        np.savez_compressed(dest/'fields.npz',X=coords,cells=top,activation_s=activation.cpu().numpy(),
                            voltage_snapshots_v=np.array(snap),traces_v=np.array(trace),pde_step_s=pde_s,ode_step_s=ode_s)
        (dest/'ep.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1"><Collection>'+''.join(
            f'<DataSet timestep="{t}" file="{name}"/>' for t,name in files)+'</Collection></VTKFile>')
        outputs[method]={'pde_total_s':sum(pde_s),'pde_mean_ms':np.mean(pde_s)*1000,'ode_total_s':sum(ode_s),
            'wall_with_output_s':time.perf_counter()-start,'mean_iterations':np.mean(iterations),
            'max_residual_ratio':max(ratios),'activated_fraction':torch.isfinite(activation).double().mean().item()}
        arrays[method]=(activation.cpu().numpy(),np.array(trace),np.array(snap))
    a,b=[arrays[k] for k in outputs];both=np.isfinite(a[0])&np.isfinite(b[0]);delta=(b[0][both]-a[0][both])*1000
    report={'dimension':dim,'nodes':len(X),'cells':len(cells),'mesh_sha256':digest,'relative_stiffness_gap':gap,
            'stimulated_nodes':int(mask.sum()),'methods':outputs,'activation_difference_ms':{
            'mean_signed':float(delta.mean()) if len(delta) else None,'mean_absolute':float(abs(delta).mean()) if len(delta) else None,
            'max_absolute':float(abs(delta).max()) if len(delta) else None},
            'snapshot_voltage_rmse_mv':float(np.sqrt(np.mean((a[2]-b[2])**2))*1000),
            'pde_speedup_torchcor_over_nodal':outputs['torchcor_cg']['pde_total_s']/outputs['afsei_nodal']['pde_total_s']}
    (folder/'report.json').write_text(json.dumps(report,indent=2));ep.close();return report


def plot(root,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,row in enumerate(rows):
        dim=row['dimension'];vals=[]
        for name,label in [('torchcor_cg','TorchCor consistent P1 + GPB'),('afsei_nodal','AFSEI nodal P1 + GPB')]:
            z=np.load(root/f'{dim}d'/name/'fields.npz');a=z['activation_s'];X=z['X'];line=np.all(abs(X[:,1:])<1e-10,axis=1)
            axes[0,j].plot(X[line,0],a[line]*1000,label=label)
            vals.append(row['methods'][name]['pde_mean_ms'])
        axes[0,j].set(xlabel='x (cm)',ylabel='Activation time (ms)',title=f'{dim}D: edge from stimulus corner');axes[0,j].legend(fontsize=8)
        bars=axes[1,j].bar(['TorchCor CG','AFSEI nodal'],vals,color=['#64748b','#f97316']);axes[1,j].bar_label(bars,fmt='%.3f')
        axes[1,j].set(ylabel='PDE wall time per step (ms)',title=f'{dim}D: same mesh / GPB / CN')
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)


@torch.no_grad()
def main():
    pa=argparse.ArgumentParser();pa.add_argument('--config',required=True);pa.add_argument('--output',required=True);a=pa.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    c=json.loads(Path(a.config).read_text());root=Path(a.output);root.mkdir(exist_ok=False,parents=True)
    (root/'config.json').write_text(json.dumps(c,indent=2))
    classes,electric,provenance=load_native_electric(Path('.deps')/('torchcor-'+COMMIT))
    rows=[]
    try:
        for dim in [2,3]:
            rows.append(run_case(dim,c,root,classes,electric))
            (root/'progress.json').write_text(json.dumps(rows,indent=2))
        plot(root,rows)
        (root/'report.json').write_text(json.dumps({'status':'completed','cases':rows,'torchcor':provenance,
            'scope':'Native pinned TorchCor assembly/electric/CG; shared AFSEI GPB CUDA implementation, same splitting; no fluid/solid/Land',
            'timing':'synchronized full evolving trajectory; includes RHS/calibration; excludes ODE/setup/VTK',
            'accuracy_scope':'same-grid method differences; neither solution is a converged reference'},indent=2))
    except Exception as e:
        (root/'failure.json').write_text(json.dumps({'error':repr(e)}));raise


if __name__=='__main__':main()
