"""Display-only PV trimming and real second-beat voltage/deformation views."""
from pathlib import Path
import csv,json,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection

R=Path('/mnt/large2/gjh/AFSEI/results/lv_source_wk_deformation_two_cycles_20261104_175339')
out=R/'plots/presentation_20261105';out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((R/'history.csv').open()))
events=[json.loads(s) for s in (R/'events.jsonl').read_text().splitlines() if s.strip()]
t=np.array([float(r['time_s']) for r in rows]);V=np.array([float(r['volume_ml']) for r in rows]);P=np.array([float(r['endocardial_load_mmhg']) for r in rows])
starts=[e for e in events if e['phase']=='IVC']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})

def first_return(fill_start,end,target):
 ids=np.flatnonzero((t>=fill_start-1e-12)&(t<=end+1e-12))
 hit=np.flatnonzero(V[ids]>=target)
 if not len(hit):raise ValueError('no actual first return to closure volume')
 j=int(hit[0]);b=int(ids[j]);a=int(ids[max(0,j-1)])
 f=(target-V[a])/(V[b]-V[a]) if a!=b else 0.
 time=t[a]+f*(t[b]-t[a]);pressure=P[a]+f*(P[b]-P[a])
 use=ids[t[ids]<time-1e-12]
 return np.r_[V[use],target],np.r_[P[use],pressure],float(time)

fig,ax=plt.subplots(figsize=(10,7.6));fig.subplots_adjust(left=.12,right=.96,bottom=.24,top=.83)
preV,preP,preend=first_return(0.,starts[0]['time_s'],starts[0]['volume_ml'])
ax.plot(preV,preP,color='.6',ls='--',lw=1.4,label='Initial loading (trimmed)')
metadata=[]
for k,start in enumerate(starts):
 beat=start['beat'];edv=start['volume_ml']
 close=next(e for e in events if e['beat']==beat and e['phase']=='IVR')
 fill=next(e for e in events if e['beat']==beat and e['phase']=='FILLING')
 end=starts[k+1]['time_s'] if k+1<len(starts) else t[-1]
 measured=(t>=start['time_s']-1e-12)&(t<=fill['time_s']+1e-12)
 systole=(t>=start['time_s']-1e-12)&(t<=close['time_s']+1e-12)
 esv=float(V[systole].min());ef=100*(edv-esv)/edv
 fillV,fillP,cut=first_return(fill['time_s'],end,edv)
 vv=np.r_[V[measured],fillV[1:]];pp=np.r_[P[measured],fillP[1:]]
 color=plt.get_cmap('tab10')(k)
 ax.plot(vv,pp,color=color,lw=2.4,label=f'Beat {beat}: EDV {edv:.2f} mL | EF {ef:.2f}%')
 ax.plot([edv,edv],[pp[-1],start['endocardial_load_mmhg']],color=color,lw=1.7,ls='--')
 for phase in ['EJECTION','IVR']:
  ids=np.flatnonzero(measured&np.array([r['phase']==phase for r in rows]))
  a=ids[int(.40*len(ids))];b=ids[int(.48*len(ids))]
  ax.annotate('',xy=(V[b],P[b]),xytext=(V[a],P[a]),arrowprops=dict(arrowstyle='-|>',lw=1.5,color=color,mutation_scale=15))
 a=int(.35*len(fillV));b=int(.45*len(fillV))
 ax.annotate('',xy=(fillV[b],fillP[b]),xytext=(fillV[a],fillP[a]),arrowprops=dict(arrowstyle='-|>',lw=1.5,color=color,mutation_scale=15))
 metadata.append(dict(beat=beat,edv_closure_ml=edv,esv_ml=esv,ef_percent=ef,
  stimulus_ms=1000*start['time_s'],filling_start_ms=1000*fill['time_s'],filling_first_return_cut_ms=1000*cut,
  excluded_later_filling_until_ms=1000*end,cut_pressure_mmhg=float(pp[-1]),
  graphical_vertical_closure_to_mmhg=start['endocardial_load_mmhg']))
ax.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)',xlim=(63,140),ylim=(-3,140));ax.grid(alpha=.2)
ax.legend(loc='upper right',fontsize=10)
fig.suptitle('LV pressure-volume cycles: filling oscillations trimmed',y=.97,fontsize=17)
fig.text(.54,.91,'Both contractions retained | deformation-to-conductivity feedback enabled',ha='center',fontsize=11,color='.3')
fig.text(.12,.045,'EDV = actual contraction-start volume, the maximum retained closure volume.\n'
 'Filling stops at its first return to that volume; subsequent oscillations are omitted.\n'
 'Colored dashed lines are graphical closure at constant volume, not simulated segments.\n'
 'Pressure-axis detail shown here. All original times and negative transients remain in the raw-data companion.',fontsize=9,color='.35',linespacing=1.6)
for ext in ('png','pdf'):fig.savefig(out/f'pv_trimmed_cycles.{ext}')
plt.close(fig)
(out/'pv_trimming_definition.json').write_text(json.dumps(dict(initial_preload_cut_ms=1000*preend,cycles=metadata,
 definition='display-only first-return filling crop; no alteration of raw data'),indent=2))
fig,ax=plt.subplots(figsize=(9.5,7),layout='constrained')
segments=np.stack((np.c_[V[:-1],P[:-1]],np.c_[V[1:],P[1:]]),axis=1)
line=LineCollection(segments,array=t[:-1]*1000,cmap='viridis',linewidth=1.5);ax.add_collection(line)
ax.autoscale();ax.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)',title='Unmodified PV trajectory: all recorded times');ax.grid(alpha=.2)
fig.colorbar(line,ax=ax,label='Time (ms)')
for ext in ('png','pdf'):fig.savefig(out/f'pv_all_times_raw.{ext}')
plt.close(fig)

# Read actual FE geometry/voltage at nine existing frames. No simulated fields are inferred.
with np.load(R/'checkpoint.npz') as z:
 X=z['X'].copy();faces=z['faces'].copy()
records=ET.parse(R/'wall.pvd').findall('.//DataSet')
targets=[1.3555,1.3655,1.3855,1.4355,1.51,1.6269,1.6669,1.9,2.2069]
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
 'du is measured from the second-stimulus configuration; no displacement amplification. First stimulus of this beat: 1355.5 ms.',ha='center',fontsize=10,linespacing=1.6)
for ext in ('png','pdf'):fig.savefig(out/f'lv_beat2_voltage_displacement_nine.{ext}',dpi=180)
plt.close(fig)
(out/'lv_nine_frame_metadata.json').write_text(json.dumps(dict(frames=frame_metadata,cut_reference_y_cm=cut,displacement_scale=1),indent=2))
print(json.dumps(dict(pv_cycles=metadata,nine_frames=frame_metadata,output=str(out)),indent=2))
