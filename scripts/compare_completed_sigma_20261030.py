"""Compare the requested rerun with the existing openCARP result; no solver calls."""
from pathlib import Path
import json,csv,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/mnt/large2/gjh/AFSEI/results')
ours=root/'ep_opencarp_sigma_nativeinit_20261030';native=root/'opencarp_gpb_h0p0625_20261030'
out=ours/'comparison';out.mkdir(exist_ok=True)
report=json.loads((ours/'report.json').read_text());row=report['cases'][0]
nr=json.loads((native/'analysis/report.json').read_text())
mesh=json.loads((native/'mesh_validation.json').read_text())
nativeX=np.loadtxt(mesh['mesh_path']+'.pts',skiprows=1)/1000 # mm
nativeA=np.loadtxt(native/'simulation_v2/init_acts_vm_act-thresh.dat')
nativeV=np.memmap(native/'simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,len(nativeX)))
labels={'openCARP':'openCARP, 0.0625 mm','consistent_tc':'Standard CG, 0.125 mm','nodal_tc':'Nodal FEM, 0.125 mm'}
colors={'openCARP':'#333333','consistent_tc':'#347eb8','nodal_tc':'#ed8b32'}
data={'openCARP':dict(X=nativeX,activation=nativeA)}
for name in ('consistent_tc','nodal_tc'):
    with np.load(ours/'3d'/name/'fields.npz') as z:
        data[name]={k:z[k] for k in ['X','activation_s','voltage_snapshots_v','snapshot_times_s','traces_v','pde_step_s','ode_step_s']}
    data[name]['X']*=10;data[name]['activation']=data[name]['activation_s']*1000
X=data['consistent_tc']['X']
if not np.array_equal(X,data['nodal_tc']['X']):raise RuntimeError('AFSEI comparison meshes differ')
ijk=np.rint(X/.0625).astype(np.int64)
ids=ijk[:,0]+321*ijk[:,1]+321*113*ijk[:,2]
if np.max(abs(nativeX[ids]-X))>1e-7:raise RuntimeError('Coarse nodes do not match native fine-grid coordinates')
metrics={}
for name in ('consistent_tc','nodal_tc'):
    d=data[name];a=d['activation'];ref=nativeA[ids];mask=np.isfinite(a)&np.isfinite(ref)&(ref>=0)
    delta=a[mask]-ref[mask]
    times=d['snapshot_times_s']*1000;take=np.flatnonzero(times<=50+1e-8)
    diff=np.array([d['voltage_snapshots_v'][i]*1000-nativeV[int(round(times[i])),ids] for i in take])
    trace=d['traces_v']*1000;traceT=np.arange(len(trace))*.05
    trace_native=np.asarray(nativeV[:,[0,len(nativeX)-1]])
    aligned=np.column_stack([np.interp(np.arange(51),traceT,trace[:,j]) for j in range(2)])
    perf=row['methods'][name]
    metrics[name]=dict(far_corner_activation_ms=float(a[-1]),
        far_corner_difference_ms=float(a[-1]-nativeA[-1]),
        far_corner_relative_difference_percent=float((a[-1]/nativeA[-1]-1)*100),
        activation_mean_absolute_difference_ms=float(abs(delta).mean()),activation_max_absolute_difference_ms=float(abs(delta).max()),
        activation_mean_signed_difference_ms=float(delta.mean()),activation_p95_absolute_difference_ms=float(np.percentile(abs(delta),95)),
        activation_comparison_nodes=int(mask.sum()),activated_fraction=float(np.isfinite(a).mean()),
        common_snapshot_times_ms=times[take].tolist(),common_snapshot_voltage_rmse_mv=float(np.sqrt(np.mean(diff**2))),
        common_snapshot_voltage_max_absolute_difference_mv=float(abs(diff).max()),
        two_probe_voltage_rmse_mv=float(np.sqrt(np.mean((aligned-trace_native)**2))),
        two_probe_voltage_min_mv=float(trace.min()),two_probe_voltage_max_mv=float(trace.max()),
        pde_mean_ms=perf['pde_mean_ms'],pde_total_s=perf['pde_total_s'],ode_total_s=perf['ode_total_s'],
        wall_with_output_and_calibration_s=perf['wall_with_output_and_calibration_s'],
        first_50ms_pde_s=float(d['pde_step_s'][:1000].sum()),first_50ms_ode_s=float(d['ode_step_s'][:1000].sum()))
result=dict(scope='Requested conductivity alignment and raw AFSEI initial state; other prior settings unchanged',
    openCARP_reused=True,openCARP=nr,afsei=metrics,initial_frame_audit=json.loads((ours/'initial_frame_audit.json').read_text()),
    same_grid_nodal_pde_speedup=row['methods']['consistent_tc']['pde_total_s']/row['methods']['nodal_tc']['pde_total_s'],
    same_grid_nodal_pde_time_reduction_percent=(1-row['methods']['nodal_tc']['pde_total_s']/row['methods']['consistent_tc']['pde_total_s'])*100,
    standard_vs_nodal=row['spatial_discretization_difference'],
    alignment=json.loads((ours/'alignment.json').read_text()),
    interpretation='Differences against openCARP are cross-configuration differences, not isolated FEM error. No cross-software algorithmic speedup is claimed.')
(out/'report.json').write_text(json.dumps(result,indent=2))
def plane(d,values):
    mask=np.isclose(d['X'][:,2],0);p=d['X'][mask]
    x=np.unique(p[:,0]);y=np.unique(p[:,1]);z=np.empty((len(y),len(x)))
    z[np.searchsorted(y,p[:,1]),np.searchsorted(x,p[:,0])]=values[mask]
    return x,y,z
plt.rcParams.update({'font.size':10,'axes.titleweight':'bold'})
fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
vmax=max(np.nanmax(d['activation']) for d in data.values())
for ax,name in zip(axes[0],data):
    x,y,z=plane(data[name],data[name]['activation'])
    im=ax.pcolormesh(x,y,z,cmap='turbo',vmin=0,vmax=vmax,shading='auto',rasterized=True)
    ax.set(title=labels[name],xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,shrink=.75,label='Activation [ms]')
ax=axes[1,0]
for name,d in data.items():
    divs=np.array([320,112,48]) if name=='openCARP' else np.array([160,56,24]);g=int(np.gcd.reduce(divs))
    targets=np.arange(g+1)[:,None]/g*np.array([20,7,3])
    # Native-node layout is x-fastest; AFSEI layout is determined from actual coordinates.
    ii=[np.argmin(np.sum((d['X']-p)**2,axis=1)) for p in targets]
    ax.plot(np.linalg.norm(d['X'][ii],axis=1),d['activation'][ii],color=colors[name],marker='o',ms=3,label=labels[name])
ax.set(xlabel='Body diagonal distance [mm]',ylabel='Activation [ms]',title='Body-diagonal propagation');ax.grid(alpha=.2);ax.legend(fontsize=8)
diffs=[data[n]['activation']-nativeA[ids] for n in ('consistent_tc','nodal_tc')]
lim=max(np.nanmax(abs(d)) for d in diffs)
for ax,name,delta in zip(axes[1,1:],('consistent_tc','nodal_tc'),diffs):
    x,y,z=plane(data[name],delta)
    im=ax.pcolormesh(x,y,z,cmap='coolwarm',vmin=-lim,vmax=lim,shading='auto',rasterized=True)
    ax.set(title=labels[name]+' minus openCARP',xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,shrink=.75,label='Difference [ms]')
fig.suptitle('Existing openCARP versus conductivity-aligned AFSEI pair | z = 0 mm\nGPB versions, mesh sizes, time steps and initial states remain different as requested',fontsize=15)
fig.savefig(out/'activation_comparison.png',dpi=160);fig.savefig(out/'activation_comparison.pdf',dpi=160)
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for j,ax in enumerate(axes):
    ax.plot(np.arange(51),nativeV[:,[0,-1][j]],color=colors['openCARP'],marker='o',ms=3,lw=1.4,label=labels['openCARP']+' (1 ms output)')
    for name in ('consistent_tc','nodal_tc'):
        d=data[name];t=np.arange(len(d['traces_v']))*.05;mask=t<=50+1e-9
        ax.plot(t[mask],d['traces_v'][mask,j]*1000,lw=1.5,color=colors[name],label=labels[name]+' (0.05 ms traces)')
    low=min(float(nativeV[:,[0,-1][j]].min()),*[float(data[n]['traces_v'][:,j].min()*1000) for n in ('consistent_tc','nodal_tc')])
    ax.set(title=['Stimulus corner (0,0,0) mm','Far corner (20,7,3) mm'][j],xlabel='Time [ms]',ylabel='Vm [mV]',xlim=(0,50),ylim=(min(-90,low-5),90))
    ax.axhline(0,color='gray',ls=':',lw=.7);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Voltage waveforms | conductivity aligned; original GPB models retained',fontsize=14)
fig.savefig(out/'voltage_comparison.png',dpi=160);fig.savefig(out/'voltage_comparison.pdf',dpi=160)
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
names=['openCARP\n16 CPU ranks\nh=0.0625 mm\ndt=0.02 ms','Standard CG\nRTX 4090\nh=0.125 mm\ndt=0.05 ms','Nodal FEM\nRTX 4090\nh=0.125 mm\ndt=0.05 ms']
cs=[colors[n] for n in data]
values=[nr['electrics_compute_ms_per_ep_step']]+[metrics[n]['pde_mean_ms'] for n in ('consistent_tc','nodal_tc')]
bars=axes[0].bar(names,values,color=cs);axes[0].bar_label(bars,fmt='%.3f',padding=3)
axes[0].set(ylabel='Reported time per timestep [ms]',title='Electrics module / PDE wrapper',yscale='log',ylim=(min(values)*.6,max(values)*2))
values=[sum(nr['physics_compute_s'].values())]+[metrics[n]['pde_total_s']+metrics[n]['ode_total_s'] for n in ('consistent_tc','nodal_tc')]
bars=axes[1].bar(names,values,color=cs);axes[1].bar_label(bars,fmt='%.1f',padding=3)
axes[1].set(ylabel='PDE + ODE compute time [s]',title='Compute for 50 / 120 / 120 ms of simulation',ylim=(0,max(values)*1.18))
v=[metrics[n]['pde_mean_ms'] for n in ('consistent_tc','nodal_tc')]
bars=axes[2].bar(['Standard CG','Nodal FEM'],v,color=cs[1:]);axes[2].bar_label(bars,fmt='%.3f',padding=3)
axes[2].set(ylabel='PDE time per step [ms]',title=f'Same AFSEI conditions: {result["same_grid_nodal_pde_speedup"]:.3f}x',ylim=(0,max(v)*1.2))
fig.suptitle('Measured performance | raw case timings; openCARP uses a larger mesh and smaller dt',fontsize=14)
fig.supxlabel('Cross-software bars have different hardware and timing scopes; the right panel isolates the two FEM discretizations.',fontsize=10)
for ax in axes[:2]:ax.tick_params(axis='x',labelsize=8)
fig.savefig(out/'performance_comparison.png',dpi=160);fig.savefig(out/'performance_comparison.pdf',dpi=160)
with (out/'metrics.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['method','far_corner_ms','activation_MAE_ms_vs_openCARP','PDE_mean_ms','PDE_plus_ODE_compute_s'])
    writer.writerow(['openCARP',nr['far_corner_activation_ms'],0,nr['electrics_compute_ms_per_ep_step'],sum(nr['physics_compute_s'].values())])
    for name,m in metrics.items():writer.writerow([name,m['far_corner_activation_ms'],m['activation_mean_absolute_difference_ms'],m['pde_mean_ms'],m['pde_total_s']+m['ode_total_s']])
print(json.dumps(dict(metrics=metrics,nodal_pde_speedup=result['same_grid_nodal_pde_speedup']),indent=2))
