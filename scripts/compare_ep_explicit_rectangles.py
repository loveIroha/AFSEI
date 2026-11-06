"""Rerun only explicit nodal; reuse immutable completed TorchCor trajectories."""
import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
import meshio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.compare_ep_rectangles import mesh,timed
from afsi_torch.ep.monodomain import Monodomain,MonodomainParameters
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.cells import GPBCell
from afsi_torch.ep.explicit import ExplicitNodal


@torch.no_grad()
def run(dim,c,root,reference):
    folder=root/f'{dim}d';folder.mkdir()
    old_report=json.loads((reference/f'{dim}d/report.json').read_text())
    old_fields=reference/f'{dim}d/torchcor_cg/fields.npz'
    # A relative directory link retains original PVD paths; no baseline recomputation.
    (folder/'torchcor_cg').symlink_to(os.path.relpath(old_fields.parent,folder),target_is_directory=True)
    baseline=np.load(old_fields)
    X,cells=mesh(dim,c);coords=X.cpu().numpy();top=cells.cpu().numpy()
    digest=hashlib.sha256(coords.tobytes()+top.tobytes()).hexdigest()
    if digest!=old_report['mesh_sha256']:raise ValueError('baseline mesh differs')
    dt=c['dt_ep_s'];steps=round(c['duration_s']/dt)
    f=X.new_zeros((len(X),dim));s=f.clone();f[:,0]=1;s[:,1]=1
    ep,setup=timed(lambda:Monodomain(X,cells,f,s,parameters=MonodomainParameters(**c['parameters']),
        numerics=MonodomainNumerics(backend='torch_pcg_jacobi')))
    explicit=ExplicitNodal(ep,dt)
    print('CFL',dim,json.dumps({'bound_s':explicit.dt_bound,'substeps':explicit.substeps,'sub_dt_s':explicit.sub_dt}),flush=True)
    mask=(X<=c['stimulus_extent_cm']+1e-12).all(1)
    zero=X.new_zeros(len(X));pulse=zero.masked_fill(mask,c['stimulus_ua_cm2'])
    cell=GPBCell(dt,c['dt_ode_ms']);state=cell.initial(zero)
    cell.advance_ode(state,0.);explicit.step(state.voltage_v,zero,pulse,dt)
    activation=zero+float('nan');snap=[];trace=[];pde=[];ode_times=[];files=[];ranges=[]
    dest=folder/'afsei_nodal';dest.mkdir();kind='tetra' if dim==3 else 'triangle'
    xyz=coords if dim==3 else np.column_stack((coords,np.zeros(len(coords))))
    start=time.perf_counter()
    for i in range(steps+1):
        if i%200==0 or i==steps:
            v=state.voltage_v.cpu().numpy().copy();snap.append(v)
            name=f'ep_{i:05d}.vtu';meshio.write(dest/name,meshio.Mesh(xyz,[(kind,top)],point_data={
                'voltage_mV':v*1000,'activation_ms':activation.cpu().numpy()*1000,'stimulus':mask.cpu().numpy().astype('uint8')}))
            files.append((i*dt,name))
        trace.append(state.voltage_v[[0,len(X)-1]].cpu().numpy())
        if i==steps:break
        (ode,current),sec=timed(lambda:cell.advance_ode(state,i*dt));ode_times.append(sec)
        S=pulse if i<round(c['stimulus_duration_s']/dt) else zero
        (v,_),sec=timed(lambda:explicit.step(state.voltage_v,current,S,dt));pde.append(sec)
        if i%100==0:
            if not torch.isfinite(v).all():raise RuntimeError('nonfinite explicit voltage')
            ranges.append([float(v.min()*1000),float(v.max()*1000)])
        crossing=torch.isnan(activation)&(state.voltage_v<0)&(v>=0)
        den=torch.where(crossing,v-state.voltage_v,torch.ones_like(v))
        activation=torch.where(crossing,i*dt+dt*(-state.voltage_v)/den,activation)
        state=replace(ode,voltage_v=v)
        if (i+1)%400==0:print(dim,'explicit ms',(i+1)*dt*1000,flush=True)
    np.savez_compressed(dest/'fields.npz',X=coords,cells=top,activation_s=activation.cpu().numpy(),
        voltage_snapshots_v=np.array(snap),traces_v=np.array(trace),pde_step_s=pde,ode_step_s=ode_times)
    (dest/'ep.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1"><Collection>'+''.join(
        f'<DataSet timestep="{t}" file="{name}"/>' for t,name in files)+'</Collection></VTKFile>')
    ours={'pde_total_s':sum(pde),'pde_mean_ms':np.mean(pde)*1000,'ode_total_s':sum(ode_times),
          'wall_with_output_s':time.perf_counter()-start,'activated_fraction':torch.isfinite(activation).double().mean().item(),
          'diffusion_substeps_per_ep':explicit.substeps,'explicit_sub_dt_s':explicit.sub_dt,
          'gershgorin_stability_limit_s':explicit.dt_bound,'sampled_voltage_ranges_mv':ranges,'setup_s':setup}
    lat=activation.cpu().numpy();other=baseline['activation_s'];both=np.isfinite(lat)&np.isfinite(other)
    delta=(lat[both]-other[both])*1000
    result={'dimension':dim,'nodes':len(X),'cells':len(cells),'mesh_sha256':digest,
        'methods':{'torchcor_cg':old_report['methods']['torchcor_cg'],'afsei_nodal':ours},
        'baseline_fields_sha256':hashlib.sha256(old_fields.read_bytes()).hexdigest(),
        'activation_difference_ms':{'mean_signed':float(delta.mean()),'mean_absolute':float(abs(delta).mean()),'max_absolute':float(abs(delta).max())},
        'snapshot_voltage_rmse_mv':float(np.sqrt(np.mean((np.array(snap)-baseline['voltage_snapshots_v'])**2))*1000),
        'pde_speedup_torchcor_over_nodal':old_report['methods']['torchcor_cg']['pde_total_s']/ours['pde_total_s']}
    (folder/'report.json').write_text(json.dumps(result,indent=2));return result


def plot(root,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,row in enumerate(rows):
        dim=row['dimension'];vals=[]
        for method,label in [('torchcor_cg','TorchCor consistent P1 / CN'),('afsei_nodal','AFSEI nodal / explicit Euler')]:
            z=np.load(root/f'{dim}d'/method/'fields.npz');X=z['X'];line=np.all(abs(X[:,1:])<1e-10,axis=1)
            axes[0,col].plot(X[line,0],z['activation_s'][line]*1000,label=label);vals.append(row['methods'][method]['pde_mean_ms'])
        axes[0,col].set(xlabel='x (cm)',ylabel='Activation time (ms)',title=f'{dim}D: edge from stimulus corner');axes[0,col].legend(fontsize=8)
        bars=axes[1,col].bar(['TorchCor CN (saved)','Nodal explicit'],vals,color=['#64748b','#f97316'])
        axes[1,col].bar_label(bars,fmt='%.3f');axes[1,col].set(ylabel='PDE wall time per EP step (ms)',title=f'{dim}D: same mesh / GPB / EP dt')
    fig.savefig(root/'comparison.png',dpi=180);plt.close(fig)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    reference=Path(a.reference).resolve();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    c=json.loads((reference/'config.json').read_text());c['nodal_backend']='explicit_forward_euler';c['diffusion_cfl_safety']=.9
    (root/'config.json').write_text(json.dumps(c,indent=2));rows=[]
    try:
        for dim in [2,3]:
            rows.append(run(dim,c,root,reference));(root/'progress.json').write_text(json.dumps(rows,indent=2))
        plot(root,rows)
        (root/'report.json').write_text(json.dumps({'status':'completed','cases':rows,'reference':str(reference),
            'baseline_recomputed':False,'method':'nodal mass; forward Euler diffusion; CFL subcycling; no AMG or linear solve',
            'scope':'same GPB, EP macro dt and mesh; different diffusion time schemes; historical TorchCor timings'},indent=2))
    except Exception as e:
        (root/'failure.json').write_text(json.dumps({'error':repr(e)}));raise


if __name__=='__main__':main()
