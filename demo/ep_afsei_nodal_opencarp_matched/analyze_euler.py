"""Completed full-EP comparison; reuse SDIRK and openCARP results."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/large2/gjh/AFSEI');results=ROOT/'results'
ours=results/'ep_afsei_nodal_euler1us_20261031';sd=results/'ep_afsei_nodal_opencarp_matched_20261030'
oc=results/'opencarp_gpb_h0p0625_20261030';out=ours/'comparison';out.mkdir(exist_ok=True)
r=json.loads((ours/'report.json').read_text());s=json.loads((sd/'report.json').read_text());c=json.loads((oc/'analysis/report.json').read_text())
assert r['status']=='completed' and r['duration_ms']==s['duration_ms']==c['duration_ms']==50
assert r['ode_method']=='explicit_euler' and r['ode_substeps_per_ep']==20
N=r['nodes'];X=np.load(results/'ep_opencarp_same_mesh_20261030/coordinates_cm.npy')*10
A={'Euler':np.load(ours/'activation_ms.npy'),'SDIRK':np.load(sd/'activation_ms.npy'),
   'openCARP':np.loadtxt(oc/'simulation_v2/init_acts_vm_act-thresh.dat')}
V={'Euler':np.memmap(ours/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N)),
   'SDIRK':np.memmap(sd/'vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N)),
   'openCARP':np.memmap(oc/'simulation_v2/vm.igb',dtype='<f4',mode='r',offset=1024,shape=(51,N))}
tr={'Euler':np.load(ours/'traces.npz'),'SDIRK':np.load(sd/'traces.npz')}
colors={'openCARP':'#364553','SDIRK':'#2d8ab8','Euler':'#e0802b'}
labels={'openCARP':'openCARP: P1 + native RL','SDIRK':'AFSEI nodal + SDIRK 2 us','Euler':'AFSEI nodal + Euler 1 us'}
metrics={}
for ref in ['SDIRK','openCARP']:
    finite=np.isfinite(A['Euler'])&np.isfinite(A[ref])&(A[ref]>=0)
    delta=A['Euler'][finite]-A[ref][finite];sq=0.;maximum=0.;rows=[]
    for k in range(51):
        diff=V['Euler'][k].astype(np.float64)-V[ref][k]
        sq+=float(np.dot(diff,diff));maximum=max(maximum,float(abs(diff).max()))
        rows.append([k,float(np.sqrt(np.mean(diff*diff))),float(abs(diff).max())])
    late=finite&(A[ref]>=5)
    metrics[ref]=dict(far_corner_difference_ms=float(A['Euler'][-1]-A[ref][-1]),
        far_corner_difference_percent=float((A['Euler'][-1]/A[ref][-1]-1)*100),
        activation_MAE_ms=float(abs(delta).mean()),activation_RMSE_ms=float(np.sqrt(np.mean(delta*delta))),
        activation_p95_absolute_difference_ms=float(np.percentile(abs(delta),95)),activation_max_absolute_difference_ms=float(abs(delta).max()),
        activation_p95_relative_difference_percent_where_ref_ge5ms=float(np.percentile(abs(A['Euler'][late]/A[ref][late]-1)*100,95)),
        common_51frame_voltage_RMSE_mv=float(np.sqrt(sq/(51*N))),common_51frame_voltage_max_difference_mv=maximum)
    with (out/f'voltage_errors_vs_{ref}.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['time_ms','RMSE_mV','max_difference_mV']);writer.writerows(rows)
old_optimized=np.load(results/'ep_afsei_nodal_ode_optimized_2ms_20261030/traces.npz')
performance=dict(ODE_speedup_vs_completed_SDIRK=s['ode_s']/r['ode_s'],
    compute_speedup_vs_completed_SDIRK=(s['ode_s']+s['pde_s'])/(r['ode_s']+r['pde_s']),
    wall_speedup_vs_completed_SDIRK=s['wall_s']/r['wall_s'],
    first2ms_ODE_speedup_vs_optimized_SDIRK=float(old_optimized['ode_step_s'].sum()/tr['Euler']['ode_step_s'][:100].sum()),
    openCARP_compute_time_ratio=sum(c['physics_compute_s'].values())/(r['ode_s']+r['pde_s']),
    note='Full SDIRK reference predates execution optimizations. First-2ms comparison reuses optimized SDIRK data. CPU openCARP and GPU AFSEI have different timing scopes.')
assert r['max_cell_guard']==0 and r['max_true_residual_ratio']<=1
initial_gap={ref:float(abs(V['Euler'][0]-V[ref][0]).max()) for ref in ['SDIRK','openCARP']}
assert max(initial_gap.values())==0
source=ROOT/'src/afsi_torch/ep'
report=dict(Euler=r,SDIRK=s,openCARP=c,comparison=metrics,performance=performance,
    initial_frame_max_difference_mv=initial_gap,all_reference_runs_reused=True,
    validation=json.loads((ROOT/'demo/ep_afsei_nodal_opencarp_matched/euler_validation_20261031.json').read_text()),
    PDE_source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['monodomain.py','jacobi.py','graph_pcg.py','gpu_kernels.py']},
    interpretation='Only GPB ODE integration/substep and execution changed; nodal FEM, CN diffusion, EP dt, physical model, native mesh, stimulus and initial state match SDIRK. Land/mechanics are not advanced in this EP-only benchmark.')
(out/'report.json').write_text(json.dumps(report,indent=2))
nx,ny=321,113;x=np.linspace(0,20,nx);y=np.linspace(0,7,ny)
def plane(a):return a[:nx*ny].reshape(ny,nx)
plt.rcParams.update({'font.size':10,'axes.titleweight':'bold'})
fig,axes=plt.subplots(2,3,figsize=(17,8.5),layout='constrained')
vmax=max(float(np.nanmax(a)) for a in A.values())
for ax,name in zip(axes[0],['openCARP','SDIRK','Euler']):
    im=ax.pcolormesh(x,y,plane(A[name]),shading='auto',cmap='turbo',vmin=0,vmax=vmax,rasterized=True)
    ax.contour(x,y,plane(A[name]),levels=np.arange(5,vmax,5),colors='white',linewidths=.6)
    ax.set(title=labels[name],xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,shrink=.7,label='Activation [ms]')
for ax,ref in zip(axes[1,:2],['SDIRK','openCARP']):
    delta=A['Euler']-A[ref];lim=max(float(np.nanmax(abs(delta))),1e-8)
    im=ax.pcolormesh(x,y,plane(delta),shading='auto',cmap='coolwarm',vmin=-lim,vmax=lim,rasterized=True)
    ax.set(title=f'Euler minus {ref}\nmax absolute difference = {lim:.4f} ms',xlabel='x [mm]',ylabel='y [mm]');ax.set_aspect('equal')
    fig.colorbar(im,ax=ax,shrink=.7,label='Difference [ms]')
q=np.rint(np.arange(17)[:,None]/16*np.array([320,112,48])).astype(int)
ii=q[:,0]+nx*q[:,1]+nx*ny*q[:,2];distance=np.linalg.norm(X[ii],axis=1)
ax=axes[1,2]
for name in ['openCARP','SDIRK','Euler']:ax.plot(distance,A[name][ii],'-o',color=colors[name],label=labels[name],ms=3)
ax.set(title='Body-diagonal propagation',xlabel='Distance [mm]',ylabel='Activation [ms]');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Complete 3D EP, 0-50 ms | same 0.0625 mm mesh, GPB parameters and EP dt 0.02 ms\nTop/difference maps: z = 0 mm; AFSEI retains CN diffusion and Jacobi-PCG, no AMG',fontsize=14)
for ext in ['png','pdf']:fig.savefig(out/f'activation_comparison.{ext}',dpi=160)
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained');ids=tr['Euler']['probe_ids']
for j,ax in enumerate(axes.flat):
    for name in ['openCARP','SDIRK','Euler']:
        t=np.arange(51) if name=='openCARP' else tr[name]['time_ms']
        values=V[name][:,ids[j]] if name=='openCARP' else tr[name]['voltage_mv'][:,j]
        ax.plot(t,values,color=colors[name],lw=1.4,marker='o' if name=='openCARP' else None,ms=2,label=labels[name])
    ax.set(title='('+', '.join(f'{p:g}' for p in X[ids[j]])+') mm',xlabel='Time [ms]',ylabel='Vm [mV]',xlim=(0,50));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Voltage waveforms | openCARP sampled every 1 ms; AFSEI traces every 0.02 ms',fontsize=13)
for ext in ['png','pdf']:fig.savefig(out/f'voltage_comparison.{ext}',dpi=160)
fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
names=['openCARP','SDIRK','Euler'];xticks=['openCARP\n16 CPU ranks','AFSEI SDIRK\nRTX4090','AFSEI Euler\nRTX4090']
series=[([c['electrics_compute_ms_per_ep_step'],s['pde_mean_ms'],r['pde_mean_ms']],'Electrics module / PDE wrapper','Mean time per step [ms]'),
    ([c['physics_compute_s']['Myocard Ionics'],s['ode_s'],r['ode_s']],'GPB ODE','Total ODE time [s]'),
    ([sum(c['physics_compute_s'].values()),s['ode_s']+s['pde_s'],r['ode_s']+r['pde_s']],'PDE + ODE','Compute for 50 ms [s]')]
for ax,(values,title,ylabel) in zip(axes,series):
    bars=ax.bar(xticks,values,color=[colors[n] for n in names]);ax.bar_label(bars,fmt='%.2f',padding=4)
    ax.set(title=title,ylabel=ylabel,ylim=(0,max(values)*1.2));ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
fig.suptitle('Measured complete EP performance | same mesh and 2500 EP steps',fontsize=14)
fig.supxlabel('Existing full SDIRK precedes execution optimizations; openCARP CPU/GPU hardware and timing scopes differ.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(out/f'performance_comparison.{ext}',dpi=160)
fig=plt.figure(figsize=(15,5),layout='constrained');boundary=np.any(np.isclose(X,0)|np.isclose(X,[20,7,3]),axis=1);face=np.flatnonzero(boundary)[::3]
for j,name in enumerate(names):
    ax=fig.add_subplot(1,3,j+1,projection='3d');sc=ax.scatter(*X[face].T,c=A[name][face],vmin=0,vmax=vmax,cmap='turbo',s=2,depthshade=False,rasterized=True)
    ax.set(title=labels[name],xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',xticks=[0,10,20],yticks=[0,3.5,7],zticks=[0,1.5,3]);ax.set_box_aspect([20,7,3]);ax.view_init(23,-58)
    fig.colorbar(sc,ax=ax,shrink=.55,label='Activation [ms]')
fig.suptitle('Full 3D activation on the six outer faces | common color range',fontsize=13)
for ext in ['png','pdf']:fig.savefig(out/f'activation_3d_comparison.{ext}',dpi=160)
with (out/'diagonal_activation.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['distance_mm','openCARP_ms','SDIRK_ms','Euler_ms']);w.writerows(zip(distance,A['openCARP'][ii],A['SDIRK'][ii],A['Euler'][ii]))
print(json.dumps(dict(comparison=metrics,performance=performance),indent=2),flush=True)
