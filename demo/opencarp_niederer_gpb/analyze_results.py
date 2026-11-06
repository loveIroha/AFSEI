"""Inspect actual openCARP outputs; no simulation or numerical changes."""
from pathlib import Path
import csv,json,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('/mnt/large2/gjh/AFSEI/results/opencarp_gpb_h0p0625_20261030')
sim=out/'simulation_v2';dest=out/'analysis';dest.mkdir(exist_ok=True)
meshinfo=json.loads((out/'mesh_validation.json').read_text())
points=np.loadtxt(meshinfo['mesh_path']+'.pts',skiprows=1)/1000 # mm
lat=np.loadtxt(sim/'init_acts_vm_act-thresh.dat')
if lat.shape!=(len(points),):raise RuntimeError('LAT/node count mismatch')
with (sim/'vm.igb').open('rb') as f:header=f.read(1024).decode('ascii',errors='replace')
fields=dict(re.findall(r'(\w+):([^\s]+)',header))
dtype=np.dtype('<f4' if fields['systeme']=='little_endian' else '>f4')
if fields['type']!='float':raise RuntimeError('Unexpected IGB precision')
vm=np.memmap(sim/'vm.igb',dtype=dtype,mode='r',offset=1024,shape=(int(fields['t']),len(points)))
times=np.arange(len(vm))*float(fields['inc_t'])
activated=np.isfinite(lat)&(lat>=0)&(lat<=50)
probe_points=np.array([[0,0,0],[5,1.75,.75],[10,3.5,1.5],[20,7,3]])
probe_ids=[int(np.argmin(np.sum((points-p)**2,axis=1))) for p in probe_points]
probes=[dict(target_mm=target.tolist(),actual_mm=points[idx].tolist(),node=idx,
    activation_ms=float(lat[idx]),initial_vm_mv=float(vm[0,idx]),peak_vm_mv=float(vm[:,idx].max()),
    final_vm_mv=float(vm[-1,idx])) for target,idx in zip(probe_points,probe_ids)]
axiscounts=meshinfo['axis_node_counts'];divs=np.array(axiscounts)-1
diaggcd=int(np.gcd.reduce(divs))
diagpoints=np.arange(diaggcd+1)[:,None]/diaggcd*np.array([20,7,3])
diagids=[int(np.argmin(np.sum((points-p)**2,axis=1))) for p in diagpoints]
diagdist=np.linalg.norm(points[diagids],axis=1)
def table(path,header,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
table(dest/'probe_waveforms.csv',['time_ms']+[f'Vm_node_{i}_mV' for i in probe_ids],np.column_stack([times,vm[:,probe_ids]]))
table(dest/'diagonal_activation.csv',['distance_mm','activation_ms','node_id'],zip(diagdist,lat[diagids],diagids))
log=(out/'run.log').read_text()
timings={name:float(value) for name,value in re.findall(r'(Electrics|Myocard Ionics):\s+Init:\s+[\d.]+ sec\s+Compute:\s+([\d.]+) sec',log)}
status=json.loads((out/'run_status.json').read_text())
steps=2500
report=dict(status=status['status'],scope='3D Niederer geometry with Grandi GPB; native openCARP mesh',
    nodes=len(points),tetrahedra=meshinfo['tetrahedra'],nominal_spacing_mm=.0625,
    dimensions_mm=[20,7,3],duration_ms=50,dt_ms=.02,steps=steps,mpi_ranks=16,
    activated_nodes=int(activated.sum()),activated_fraction=float(activated.mean()),
    activation_min_ms=float(lat[activated].min()),activation_max_ms=float(lat[activated].max()),
    far_corner_activation_ms=float(lat[probe_ids[-1]]),
    diagonal_monotone=bool(np.all(np.diff(lat[diagids])>=-1e-6)),
    voltage_frames=len(vm),voltage_output_interval_ms=1,
    voltage_all_finite=bool(np.isfinite(vm).all()),voltage_min_mv=float(vm.min()),voltage_max_mv=float(vm.max()),
    initial_voltage_range_mv=[float(vm[0].min()),float(vm[0].max())],final_voltage_range_mv=[float(vm[-1].min()),float(vm[-1].max())],
    whole_job_wall_s=status['elapsed_s'],physics_compute_s=timings,
    physics_compute_share={name:value/sum(timings.values()) for name,value in timings.items()},
    electrics_compute_ms_per_ep_step=timings.get('Electrics',0)/steps*1000,
    ionics_compute_ms_per_ep_step=timings.get('Myocard Ionics',0)/steps*1000,
    whole_job_ms_per_ep_step=status['elapsed_s']/steps*1000,
    petsc_error_log_bytes=(sim/'petsc_err_log.txt').stat().st_size,probes=probes,
    limitations=['Vm snapshots every 1 ms; LAT uses solver-time threshold detection at 0 mV.',
        '50 ms covers activation and early plateau, not full action-potential repolarization.',
        'GPB replacement means published ten Tusscher benchmark activation values are not an exact reference.',
        'Electrics compute is the openCARP physics-module timing, not a standalone kernel-only PDE timer.'])
stimulus=np.all(points<=1.5+1e-9,axis=1)
node_peaks=np.max(vm,axis=0)
high=node_peaks>60
peak_flat=int(np.argmax(vm));peak_frame,peak_node=np.unravel_index(peak_flat,vm.shape)
report['stimulus_peak_inspection']=dict(stimulus_nodes=int(stimulus.sum()),
    peak_in_stimulus_mv=float(node_peaks[stimulus].max()),
    peak_outside_stimulus_mv=float(node_peaks[~stimulus].max()),
    nodes_exceeding_60mv=int(high.sum()),nodes_exceeding_60mv_outside_stimulus=int((high&~stimulus).sum()),
    maximum_sampled_voltage_time_ms=float(times[peak_frame]),maximum_sampled_voltage_position_mm=points[peak_node].tolist())
(dest/'report.json').write_text(json.dumps(report,indent=2))
plt.rcParams.update({'font.size':11,'axes.titleweight':'bold'})
fig=plt.figure(figsize=(14,10),layout='constrained')
vmin,vmax=0,float(lat[activated].max());norm=plt.Normalize(vmin,vmax)
ax=fig.add_subplot(221,projection='3d')
boundary=np.any(np.isclose(points,0)|np.isclose(points,[20,7,3]),axis=1)
ids=np.flatnonzero(boundary&activated)[::3]
sc=ax.scatter(*points[ids].T,c=lat[ids],cmap='turbo',norm=norm,s=2,depthshade=False,rasterized=True)
ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title='Activation on all six outer faces')
ax.set_box_aspect([20,7,3]);ax.view_init(23,-58)
fig.colorbar(sc,ax=ax,shrink=.6,pad=.03,label='Activation time [ms]')
ax=fig.add_subplot(222)
idx=np.flatnonzero(np.isclose(points[:,2],0));p=points[idx];nx,ny=axiscounts[:2]
plane=np.empty((ny,nx));plane[np.rint(p[:,1]/.0625).astype(int),np.rint(p[:,0]/.0625).astype(int)]=lat[idx]
im=ax.pcolormesh(np.linspace(0,20,nx),np.linspace(0,7,ny),plane,cmap='turbo',norm=norm,shading='auto',rasterized=True)
levels=np.arange(5,vmax,5)
if len(levels):
    cs=ax.contour(np.linspace(0,20,nx),np.linspace(0,7,ny),plane,levels=levels,colors='white',linewidths=.7)
    ax.clabel(cs,fmt='%g',fontsize=8)
ax.add_patch(plt.Rectangle((0,0),1.5,1.5,fill=False,lw=1.5,edgecolor='black',ls='--'))
ax.set(xlabel='x [mm]',ylabel='y [mm]',title='Activation on z = 0 mm; dashed box: stimulus')
ax.set_aspect('equal');fig.colorbar(im,ax=ax,shrink=.6,label='Activation time [ms]')
ax=fig.add_subplot(223)
ax.plot(diagdist,lat[diagids],color='#2379b5',marker='o',ms=4,lw=2)
ax.set(xlabel='Distance along body diagonal [mm]',ylabel='Activation time [ms]',title='Body-diagonal propagation: exact mesh nodes')
ax.grid(alpha=.2)
ax=fig.add_subplot(224)
for target,idx in zip(probe_points,probe_ids):
    ax.plot(times,vm[:,idx],lw=1.8,label='('+', '.join(f'{a:g}' for a in target)+') mm')
    ax.axvline(lat[idx],color='gray',lw=.6,alpha=.3)
ax.axhline(0,color='black',ls=':',lw=.8)
ax.set(xlabel='Time [ms]',ylabel='Vm [mV]',title='Voltage traces; 1 ms output sampling',xlim=(0,50))
ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.2)
fig.suptitle('openCARP | GPB (Grandi) | 20 x 7 x 3 mm | h = 0.0625 mm\nCN + consistent mass | dt = 0.02 ms | 16 CPU MPI ranks',fontsize=15)
fig.savefig(dest/'activation_and_voltage.png',dpi=160)
fig.savefig(dest/'activation_and_voltage.pdf',dpi=160)
fig2,ax=plt.subplots(figsize=(8,4),layout='constrained')
names=['Electrics module','GPB ionics'];values=[timings[n] for n in ['Electrics','Myocard Ionics']]
bars=ax.barh(names,values,color=['#2379b5','#e28e39'])
for bar,value in zip(bars,values):ax.text(value+6,bar.get_y()+bar.get_height()/2,f'{value:.2f} s ({value/sum(values)*100:.1f}%)',va='center')
ax.set(xlabel='Compute time [s]',xlim=(0,max(values)*1.35),title='Reported module compute times (excludes initialization/output)')
fig2.savefig(dest/'compute_times.png',dpi=160)
print(json.dumps(report,indent=2))
