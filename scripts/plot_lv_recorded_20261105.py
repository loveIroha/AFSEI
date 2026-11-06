"""Raw complete 0-2s LV results; accepted active loads, never reconstructed ODE."""
from pathlib import Path
import csv,json,xml.etree.ElementTree as ET
import numpy as np,meshio,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
R=Path('/mnt/large2/gjh/AFSEI/results/lv_ramp05_fixed135_active_20261105/full')
out=R/'plots/completed_20261105';out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((R/'history.csv').open()))
events=[json.loads(s) for s in (R/'events.jsonl').read_text().splitlines() if s.strip()]
t=np.array([float(r['time_s']) for r in rows]);V=np.array([float(r['volume_ml']) for r in rows]);P=np.array([float(r['endocardial_load_mmhg']) for r in rows])
starts=[e for e in events if e['phase']=='IVC']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':210})
metadata=[]
fig,ax=plt.subplots(figsize=(10,7.4));fig.subplots_adjust(left=.12,right=.97,bottom=.22,top=.84)
pre=t<=.5;ax.plot(V[pre],P[pre],color='.55',lw=1.6,label='Initial loading: 0-500 ms')
for k,start in enumerate(starts):
 end=starts[k+1]['time_s'] if k+1<len(starts) else t[-1]
 mask=(t>=start['time_s']-1e-12)&(t<=end+1e-12)
 close=next(e for e in events if e['phase']=='IVR' and e['beat']==start['beat'])
 edv=start['volume_ml'];syst=(t>=start['time_s']-1e-12)&(t<=close['time_s']+1e-12)
 esv=float(V[syst].min());ef=100*(edv-esv)/edv;color=plt.get_cmap('tab10')(k)
 ax.plot(V[mask],P[mask],lw=2,color=color,label=f'Beat {k+1}: EDV {edv:.2f} mL | EF {ef:.2f}%')
 ax.scatter(V[mask][0],P[mask][0],s=45,color=color,zorder=5)
 ax.scatter(V[mask][-1],P[mask][-1],s=55,facecolors='white',edgecolors=[color],zorder=6)
 metadata.append(dict(beat=k+1,edv_at_stimulus_ml=edv,esv_ml=esv,ef_percent=ef,
  opening_ms=1000*next(e['time_s'] for e in events if e['phase']=='EJECTION' and e['beat']==k+1),
  closure_ms=1000*close['time_s'],net_ejected_ml=close['cumulative_out_ml']-start['cumulative_out_ml'],
  peak_pressure_mmhg=float(P[mask].max())))
ax.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=10)
fig.suptitle('LV pressure-volume trajectory: all recorded times (0-2 s)',y=.97,fontsize=16)
fig.text(.12,.06,'Raw trajectories: no oscillation trimming, pressure clipping, smoothing, or artificial closure.\n'
 'Beat 1 has premature outlet closure/unloading at 500.3 ms and nearly zero ejection.\n'
 'Beat 2 is still refilling when the requested 2 s run ends. Pressure = endocardial coupling load.',fontsize=9,color='.35',linespacing=1.7)
for ext in ('png','pdf'):fig.savefig(out/f'pv_all_times.{ext}')
plt.close(fig)

stats=list(csv.DictReader((R/'active_tension_layers.csv').open()))
st=np.array([float(r['time_s'])*1000 for r in stats])
names=['endo_side','middle','epi_side'];labels=['Endocardial-side layer','Midwall layer','Epicardial-side layer'];colors=['#c34e32','#1b8d78','#3166ac']
peak_stats=[]
for name in names:
 vals=np.array([float(r[name+'_Ta_mean_kpa']) for r in stats]);i=vals.argmax()
 peak_stats.append(dict(layer=name,peak_mean_Ta_kpa=float(vals[i]),peak_time_ms=float(st[i]),
  peak_local_Ta_kpa=max(float(r[name+'_Ta_max_kpa']) for r in stats)))
closure2=1000*next(e['time_s'] for e in events if e['phase']=='IVR' and e['beat']==2)
fig,axes=plt.subplots(2,1,figsize=(11,8.4));fig.subplots_adjust(left=.10,right=.97,bottom=.13,top=.88,hspace=.38)
for ax in axes:
 for name,label,color in zip(names,labels,colors):
  vals=np.array([float(r[name+'_Ta_mean_kpa']) for r in stats]);ax.plot(st,vals,lw=1.9,label=label,color=color)
 ax.axvspan(closure2,closure2+80,color='.8',alpha=.35,label='80 ms artificial unloading')
 ax.axvline(500,color='.5',ls='--',lw=1);ax.axvline(1350,color='.5',ls='--',lw=1)
 ax.set(xlabel='Time (ms)',ylabel='Mean active tension Ta (kPa)');ax.grid(alpha=.2)
axes[0].set(xlim=(0,2000),title='All recorded times; first beat produces negligible active tension')
axes[1].set(xlim=(1330,1750),title='Second stimulus and subsequent unloading')
axes[0].legend(loc='upper left',fontsize=9)
fig.suptitle('Actual applied active tension across myocardial layers',y=.96,fontsize=17)
fig.text(.10,.025,'Layers: thirds of the reference Laplace transmural coordinate (endo = 0, epi = 1).\n'
 'Curves are reference-volume-weighted means of accepted quadrature Ta, including artificial unloading; not cavity pressure.',fontsize=9.5,color='.35',linespacing=1.6)
for ext in ('png','pdf'):fig.savefig(out/f'active_tension_layers.{ext}')
plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5.6));fig.subplots_adjust(left=.10,right=.97,bottom=.22,top=.86)
for name,label,color in zip(names,labels,colors):
 ax.plot(st,[float(r[name+'_active_cauchy_mean_kpa']) for r in stats],lw=1.8,label=label,color=color)
ax.set(xlabel='Time (ms)',ylabel='Mean active fiber-direction Cauchy stress (kPa)',xlim=(0,2000));ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
fig.suptitle('Applied active fiber stress, including mechanical stretch factors',y=.97,fontsize=15)
fig.text(.10,.055,'This stress includes the PK1 stretch multiplier and push-forward to Cauchy stress.\n'
 'It is distinct from the Land tension Ta and from a nodal force.',fontsize=9,color='.35',linespacing=1.7)
for ext in ('png','pdf'):fig.savefig(out/f'active_cauchy_stress_layers.{ext}')
plt.close(fig)

# Actual saved nodal forces, streamed by frame (never load the entire Ta history).
with h5py.File(R/'active_load_history.h5','r') as f:
 ft=f['time_s'][:]*1000;phi=f['transmural_phi'][:]
 masks=[phi<1/3,(phi>=1/3)&(phi<2/3),phi>=2/3]
 rms=[]
 for i in range(len(ft)):
  force=f['active_force_dyn'][i];norm2=(force*force).sum(1)
  rms.append([float(np.sqrt(norm2[m].mean())*.01) for m in masks]) # dyn -> mN
rms=np.asarray(rms)
fig,ax=plt.subplots(figsize=(10,5.5));fig.subplots_adjust(left=.10,right=.97,bottom=.23,top=.86)
for j,(label,color) in enumerate(zip(labels,colors)):ax.plot(ft,rms[:,j],lw=1.8,label=label,color=color)
ax.set(xlabel='Time (ms)',ylabel='RMS active nodal-force magnitude (mN)',xlim=(0,2000));ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=9)
fig.suptitle('Saved active nodal forces across myocardial layers',y=.96,fontsize=16)
fig.text(.10,.05,'RMS over the nodes in each material-side layer; depends on mesh resolution.\n'
 'This is not a surface traction or a net resultant (internal nodal forces can cancel).',fontsize=9,color='.35',linespacing=1.7)
for ext in ('png','pdf'):fig.savefig(out/f'active_nodal_force_layers.{ext}')
plt.close(fig)
with (out/'nodal_force_layer_rms.csv').open('w',newline='') as stream:
 w=csv.writer(stream);w.writerow(['time_ms',*names]);w.writerows(zip(ft,*rms.T))
(out/'summary.json').write_text(json.dumps(dict(cycles=metadata,peak_active_tension=peak_stats,
 status=json.loads((R/'summary.json').read_text())['status'],runtime_hours=json.loads((R/'summary.json').read_text())['run_seconds']/3600,
 all_time_pv_unmodified=True,active_tension_statistics_interval_ms=1,force_fields_interval_ms=10),indent=2))
# Read actual FE geometry/voltage at nine existing frames. No simulated fields are inferred.
with np.load(R/'checkpoint.npz') as z:
 X=z['X'].copy();faces=z['faces'].copy()
records=ET.parse(R/'wall.pvd').findall('.//DataSet')
targets=[1.35,1.36,1.38,1.43,1.50,1.6197,1.66,1.85,2.0]
frames=[]
for target in targets:
 rec=min(records,key=lambda e:abs(float(e.get('timestep'))-target));m=meshio.read(R/rec.get('file'))
 frames.append(dict(time=float(rec.get('timestep')),x=m.points,voltage=m.point_data['voltage_mV'],source=rec.get('file')))
base=frames[0]['x'];center=base.mean(0);cut=float(np.median(X[:,1]))
# Clip the SAME material boundary triangles in all frames. Intersections use
# barycentric interpolation, not displacement amplification or new simulation.
selected=[];coeff=[];cut_edges=[]
for face in faces:
 signed=X[face,1]-cut;poly=[];cross=[];eye=np.eye(3)
 for j in range(3):
  a=(j-1)%3;b=j;ina=signed[a]>=0;inb=signed[b]>=0
  if ina!=inb:
   q=eye[a]+signed[a]/(signed[a]-signed[b])*(eye[b]-eye[a]);poly.append(q);cross.append(q)
  if inb:poly.append(eye[b])
 for j in range(1,len(poly)-1):selected.append(face);coeff.append([poly[0],poly[j],poly[j+1]])
 if len(cross)==2:cut_edges.append(np.array(cross)@base[face])
selected=np.array(selected);coeff=np.array(coeff)
limits=np.concatenate([f['x'] for f in frames]);lo=(limits.min(0)-center)*10-3;hi=(limits.max(0)-center)*10+3
extent=hi-lo;norm=Normalize(-85,45);cmap=plt.get_cmap('turbo')
fig=plt.figure(figsize=(15,14));fig.subplots_adjust(left=.02,right=.91,bottom=.075,top=.90,hspace=.15,wspace=.03)
frame_metadata=[]
for j,f in enumerate(frames):
 xx=(f['x']-center)*10;tri=np.einsum('tab,tbi->tai',coeff,xx[selected]);values=np.einsum('tab,tb->ta',coeff,f['voltage'][selected]).mean(1)
 ax=fig.add_subplot(3,3,j+1,projection='3d')
 surface=Poly3DCollection(tri,facecolors=cmap(norm(values)),edgecolors='none',linewidths=0);surface.set_rasterized(True);ax.add_collection3d(surface)
 ax.add_collection3d(Line3DCollection((np.array(cut_edges)-center)*10,colors='.5',linewidths=.5,linestyles='--',alpha=.5))
 delta=float(np.linalg.norm(f['x']-base,axis=1).max()*10)
 ax.set_title(f"{f['time']*1000:.1f} ms | max |du| = {delta:.2f} mm",fontsize=10,pad=3)
 ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect(extent)
 ax.set_proj_type('ortho');ax.view_init(elev=18,azim=-80);ax.tick_params(labelsize=6,pad=0)
 ax.set_xlabel('x (mm)',fontsize=7,labelpad=0);ax.set_ylabel('y (mm)',fontsize=7,labelpad=0);ax.set_zlabel('z (mm)',fontsize=7,labelpad=0)
 for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.pane.set_alpha(0)
 ax.grid(False)
 frame_metadata.append(dict(time_ms=f['time']*1000,relative_to_second_stimulus_ms=(f['time']-frames[0]['time'])*1000,
  max_displacement_from_beat_start_mm=delta,voltage_min_mv=float(f['voltage'].min()),voltage_max_mv=float(f['voltage'].max()),source=f['source']))
bar=fig.add_axes([.935,.20,.012,.6]);fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,label='Membrane voltage (mV)')
fig.suptitle('Second LV beat: electrical propagation and displacement',fontsize=18,y=.97)
fig.text(.48,.935,'Actual deformed FE half-wall | displacement scale = 1 | common camera and voltage range',ha='center',fontsize=11)
fig.text(.48,.025,'Cutaway reveals the endocardial surface. Dashed contours: geometry at the second stimulus.\n'
 'du is measured from the second-stimulus configuration; no displacement amplification. First stimulus of this beat: 1350 ms.',ha='center',fontsize=10,linespacing=1.6)
for ext in ('png','pdf'):fig.savefig(out/f'lv_beat2_voltage_displacement_nine.{ext}',dpi=180)
plt.close(fig)
(out/'lv_nine_frame_metadata.json').write_text(json.dumps(dict(frames=frame_metadata,cut_reference_y_cm=cut,displacement_scale=1),indent=2))
print(json.dumps(dict(pv_cycles=metadata,nine_frames=frame_metadata,output=str(out)),indent=2))
