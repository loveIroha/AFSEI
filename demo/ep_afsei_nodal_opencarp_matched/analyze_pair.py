"""Both Euler trajectories on GPU0, compared with immutable openCARP."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/large2/gjh/AFSEI');pair=ROOT/'results/ep_euler_fem_pair_20261031'
oc=ROOT/'results/opencarp_gpb_h0p0625_20261030';dest=pair/'comparison';dest.mkdir(exist_ok=True)
folders={'Classic':pair/'torchcor_cg','Nodal':pair/'nodal_afsei'}
reports={name:json.loads((folder/'report.json').read_text()) for name,folder in folders.items()}
# The first classic run inherited the old nodal timing-description string;
# correct only that metadata label, retaining the original text for audit.
if 'Graph-PCG' in reports['Classic']['timing_scope']:
    reports['Classic']['timing_scope_original_label']=reports['Classic']['timing_scope']
    reports['Classic']['timing_scope']='Synchronized wall: ODE includes Vm conversion; PDE includes unit/source conversion, pinned native CN RHS, common norm mapping, native TorchCor CG and true-residual verification; wall includes 1ms output and LAT/probes.'
    reports['Classic']['metadata_correction']='Timing-description label only; numerical data and recorded times unchanged.'
    (folders['Classic']/'report.json').write_text(json.dumps(reports['Classic'],indent=2))
original=json.loads((oc/'analysis/report.json').read_text());N=original['nodes']
for r in reports.values():
    assert r['status']=='completed' and r['duration_ms']==50 and r['physical_gpu']=='0'
    assert r['ode_method']=='explicit_euler' and r['ode_dt_ms']==.001 and r['ode_substeps_per_ep']==20
    assert r['max_cell_guard']==0 and r['max_true_residual_ratio']<=1 and r['activated_fraction']==1
cfg={name:json.loads((folder/'configuration.json').read_text()) for name,folder in folders.items()}
for key in ['operator_cache','dt_s','ode_dt_ms','geometry_mm','mesh_mm','fiber','boundary','stimulus_cube_mm','stimulus_ua_cm2','stimulus_ms','solver','ode_method','ode_execution']:
    assert cfg['Classic'][key]==cfg['Nodal'][key],key
common=json.loads((Path(cfg['Nodal']['operator_cache'])/'common_configuration.json').read_text())
X=np.load(Path(cfg['Nodal']['operator_cache'])/'coordinates_cm.npy')*10
act={name:np.load(folder/'activation_ms.npy') for name,folder in folders.items()}
act['openCARP']=np.loadtxt(oc/'simulation_v2/init_acts_vm_act-thresh.dat')
volts={name:np.memmap(folder/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N)) for name,folder in folders.items()}
volts['openCARP']=np.memmap(oc/'simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))
tr={name:np.load(folder/'traces.npz') for name,folder in folders.items()}
cell={name:np.load(folder/'cell_traces.npz') for name,folder in folders.items()}
labels={'Classic':'TorchCor classic P1 + Euler 1 us','Nodal':'AFSEI nodal P1 + Euler 1 us','openCARP':'openCARP: reference P1 + native RL'}
colors={'Classic':'#397eaf','Nodal':'#e58a35','openCARP':'#344653'}
metrics={}
for first,second in [('Classic','openCARP'),('Nodal','openCARP'),('Nodal','Classic')]:
    delta=act[first]-act[second];finite=np.isfinite(delta)&(act[second]>=0);d=delta[finite];late=finite&(act[second]>=5)
    sq=0.;vmax=0.;frames=[]
    for k in range(51):
        dv=volts[first][k].astype(np.float64)-volts[second][k]
        sq+=float(np.dot(dv,dv));vmax=max(vmax,float(abs(dv).max()));frames.append([k,float(np.sqrt(np.mean(dv*dv))),float(abs(dv).max())])
    metrics[first+'_vs_'+second]=dict(far_corner_delta_ms=float(act[first][-1]-act[second][-1]),
        far_corner_relative_delta_percent=float((act[first][-1]/act[second][-1]-1)*100),
        activation_MAE_ms=float(abs(d).mean()),activation_RMSE_ms=float(np.sqrt(np.mean(d*d))),
        activation_p95_abs_delta_ms=float(np.percentile(abs(d),95)),activation_max_abs_delta_ms=float(abs(d).max()),
        activation_p95_relative_delta_percent_where_ref_ge5ms=float(np.percentile(abs(act[first][late]/act[second][late]-1)*100,95)),
        common_51frame_voltage_RMSE_mv=float(np.sqrt(sq/(51*N))),common_51frame_voltage_max_delta_mv=vmax,
        far_corner_within5percent=bool(abs(act[first][-1]/act[second][-1]-1)<=.05))
    with (dest/f'frame_errors_{first}_vs_{second}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['time_ms','Vm_RMSE_mV','Vm_max_delta_mV']);w.writerows(frames)
assert all(float(abs(volts[name][0]-volts['openCARP'][0]).max())==0 for name in folders)
control=json.loads((pair/'common_pde_control.json').read_text())
perf=dict(classic_over_nodal_PDE_time_ratio=reports['Classic']['pde_s']/reports['Nodal']['pde_s'],
    classic_over_nodal_compute_time_ratio=(reports['Classic']['pde_s']+reports['Classic']['ode_s'])/(reports['Nodal']['pde_s']+reports['Nodal']['ode_s']),
    classic_over_nodal_wall_time_ratio=reports['Classic']['wall_s']/reports['Nodal']['wall_s'])
for name in reports:
    times=tr[name]['pde_step_s']*1000
    reports[name]['PDE_median_ms']=float(np.median(times));reports[name]['PDE_p95_ms']=float(np.percentile(times,95))
result=dict(scope='Same GPU0, same native 3D mesh and GPB Euler 1us, same EP CN dt20us, different mass quadrature/PDE implementation. openCARP reused.',
    runs=reports,openCARP=original,comparison=metrics,performance=perf,common_PDE_control=control,
    preflight=json.loads((pair/'preflight.json').read_text()),
    torchcor=json.loads((folders['Classic']/'torchcor_provenance.json').read_text()),
    mesh_and_physical_parameters=common,
    caveats=['Classic group uses actual pinned TorchCor native CN RHS and CG; global affine P1 matrices are shared cached assembly independently checked against native local formulas. Assembly time excluded.',
        'Native stopping norm differs; a proven norm bound maps the common physical true residual tolerance. Each accepted step is verified; no extra quarter safety factor.',
        'Whole-trajectory PDE ratio includes discretization, matrix iterations, native/optimized execution and required unit/norm wrappers. Fixed-input controls separate these effects.',
        'openCARP uses 16 CPU ranks and native ionic/time integration; CPU/GPU and module/wrapper timer scopes differ. No equal-accuracy speedup claim follows solely from raw timings.',
        '50ms is activation and early plateau, not complete repolarization, moving myocardium or Land feedback.'])
result['mesh_and_physical_parameters'].pop('operator_splitting',None)
(dest/'report.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'font.size':10,'axes.titleweight':'bold'})
nx,ny=321,113;x=np.linspace(0,20,nx);y=np.linspace(0,7,ny)
def plane(a):return a[:nx*ny].reshape(ny,nx)
names=['openCARP','Classic','Nodal'];maxact=max(float(np.nanmax(a)) for a in act.values())
fig,axes=plt.subplots(2,3,figsize=(17,8.5),layout='constrained')
for ax,name in zip(axes[0],names):
    im=ax.pcolormesh(x,y,plane(act[name]),cmap='turbo',vmin=0,vmax=maxact,shading='auto',rasterized=True)
    ax.contour(x,y,plane(act[name]),levels=np.arange(5,maxact,5),colors='white',linewidths=.6)
    ax.set(title=labels[name],xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.7,label='Activation [ms]')
for ax,name in zip(axes[1,:2],['Classic','Nodal']):
    d=act[name]-act['openCARP'];lim=max(float(abs(d).max()),1e-8)
    im=ax.pcolormesh(x,y,plane(d),cmap='coolwarm',vmin=-lim,vmax=lim,shading='auto',rasterized=True)
    ax.set(title=f'{name} minus openCARP\nmax absolute difference = {lim:.4f} ms',xlabel='x [mm]',ylabel='y [mm]')
    ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.7,label='Difference [ms]')
q=np.rint(np.arange(17)[:,None]/16*np.array([320,112,48])).astype(int);ids=q[:,0]+321*q[:,1]+321*113*q[:,2]
distance=np.linalg.norm(X[ids],axis=1);ax=axes[1,2]
for name in names:ax.plot(distance,act[name][ids],'-o',color=colors[name],ms=3,label=labels[name])
ax.set(title='Body-diagonal propagation',xlabel='Distance [mm]',ylabel='Activation [ms]');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Classic P1-Galerkin versus nodal P1 | both use identical GPB Euler 1 us\nSame 20 x 7 x 3 mm mesh, h=0.0625 mm, CN EP dt=0.02 ms | maps: z=0 mm',fontsize=14)
for ext in ['png','pdf']:fig.savefig(dest/f'activation_comparison.{ext}',dpi=160)
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');probes=tr['Nodal']['probe_ids']
for j,ax in enumerate(axes.flat):
    for name in names:
        t=np.arange(51) if name=='openCARP' else tr[name]['time_ms']
        v=volts[name][:,probes[j]] if name=='openCARP' else tr[name]['voltage_mv'][:,j]
        ax.plot(t,v,color=colors[name],lw=1.4,marker='o' if name=='openCARP' else None,ms=2,label=labels[name])
    ax.set(title='('+', '.join(f'{p:g}' for p in X[probes[j]])+') mm',xlabel='Time [ms]',ylabel='Vm [mV]',xlim=(0,50));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Voltage waveforms | same GPB/Euler for both GPU FEM methods; openCARP output every 1 ms',fontsize=13)
for ext in ['png','pdf']:fig.savefig(dest/f'voltage_comparison.{ext}',dpi=160)
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
vals=[reports[n]['pde_mean_ms'] for n in ['Classic','Nodal']]
bars=axes[0].bar(['TorchCor classic\nGPU0','AFSEI nodal\nGPU0'],vals,color=[colors[n] for n in ['Classic','Nodal']]);axes[0].bar_label(bars,fmt='%.3f',padding=4)
axes[0].set(title='Full trajectory: PDE time',ylabel='Mean time per EP step [ms]',ylim=(0,max(vals)*1.2))
vals=[sum(original['physics_compute_s'].values())]+[reports[n]['pde_s']+reports[n]['ode_s'] for n in ['Classic','Nodal']]
bars=axes[1].bar(['openCARP\n16 CPU ranks','Classic Euler\nGPU0','Nodal Euler\nGPU0'],vals,color=[colors[n] for n in names]);axes[1].bar_label(bars,fmt='%.2f',padding=4)
axes[1].set(title='PDE + ODE, complete 50 ms',ylabel='Compute [s]',ylim=(0,max(vals)*1.2))
keys=['consistent_torchcor','consistent_afsei','nodal_afsei'];vals=[control['timings'][k]['median_ms'] for k in keys]
bars=axes[2].bar(['Classic\nTorchCor CG','Classic\nAFSEI CG','Nodal\nAFSEI CG'],vals,color=['#397eaf','#7e78a6','#e58a35']);axes[2].bar_label(bars,fmt='%.3f',padding=4)
axes[2].set(title='Fixed-input PDE control',ylabel='Median PDE [ms]',ylim=(0,max(vals)*1.2))
fig.suptitle('Measured performance | both Euler runs sequentially on the same RTX4090 GPU0',fontsize=14)
fig.supxlabel('Fixed-input controls use the same V/I/stimulus. CPU/module versus GPU/wrapper timing scopes differ for openCARP.',fontsize=9)
for ax in axes:ax.grid(axis='y',alpha=.15);ax.tick_params(axis='x',labelsize=8)
for ext in ['png','pdf']:fig.savefig(dest/f'performance_comparison.{ext}',dpi=160)
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for j,ax in enumerate(axes.flat):
    for name in ['Classic','Nodal']:ax.plot(cell[name]['time_ms'],cell[name]['calcium_um'][:,j],color=colors[name],label=labels[name])
    ax.set(title=f'Node {probes[j]}',xlabel='Time [ms]',ylabel='Cai [uM]');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Calcium at the same physical probes | differences include different propagated Vm',fontsize=13)
for ext in ['png','pdf']:fig.savefig(dest/f'calcium_comparison.{ext}',dpi=160)
fig=plt.figure(figsize=(15,5),layout='constrained');boundary=np.any(np.isclose(X,0)|np.isclose(X,[20,7,3]),axis=1);ii=np.flatnonzero(boundary)[::3]
for j,name in enumerate(names):
    ax=fig.add_subplot(1,3,j+1,projection='3d');sc=ax.scatter(*X[ii].T,c=act[name][ii],cmap='turbo',vmin=0,vmax=maxact,s=2,depthshade=False,rasterized=True)
    ax.set(title=labels[name],xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',xticks=[0,10,20],yticks=[0,3.5,7],zticks=[0,1.5,3]);ax.set_box_aspect([20,7,3]);ax.view_init(23,-58);fig.colorbar(sc,ax=ax,shrink=.55,label='Activation [ms]')
fig.suptitle('Full 3D activation: all six outer faces, same color range',fontsize=13)
for ext in ['png','pdf']:fig.savefig(dest/f'activation_3d_comparison.{ext}',dpi=160)
with (dest/'diagonal_activation.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['distance_mm','openCARP_ms','Classic_Euler_ms','Nodal_Euler_ms']);w.writerows(zip(distance,act['openCARP'][ids],act['Classic'][ids],act['Nodal'][ids]))
print(json.dumps(dict(comparison=metrics,performance=perf),indent=2),flush=True)
