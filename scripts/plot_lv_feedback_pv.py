"""Second recorded PV cycle only; separate zoom/full-range views, no synthetic closure."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/large2/gjh/AFSEI/results/lv_source_wk_deformation_two_cycles_20261104_175339')
out=R/'plots';out.mkdir(exist_ok=True)
rows=list(csv.DictReader((R/'history.csv').open()))
events=[json.loads(s) for s in (R/'events.jsonl').read_text().splitlines() if s.strip()]
t=np.array([float(r['time_s']) for r in rows]);V=np.array([float(r['volume_ml']) for r in rows]);P=np.array([float(r['endocardial_load_mmhg']) for r in rows])
start=next(e for e in events if e['phase']=='IVC' and e['beat']==2)
close=next(e for e in events if e['phase']=='IVR' and e['beat']==2)
fill=next(e for e in events if e['phase']=='FILLING' and e['beat']==1)
diastole=(t>=fill['time_s']-1e-12)&(t<=start['time_s']+1e-12)
systole=(t>=start['time_s']-1e-12)&(t<=close['time_s']+1e-12)
edv=float(V[diastole].max());esv=float(V[systole].min());ef=100*(edv-esv)/edv
use=t>=start['time_s']-1e-12;tt=t[use];vv=V[use];pp=P[use]
metrics=dict(status=json.loads((R/'summary.json').read_text())['status'],cycle=2,
 start_time_ms=float(tt[0]*1000),end_time_ms=float(tt[-1]*1000),edv_max_ml=edv,esv_min_ml=esv,ef_percent=ef,
 edv_definition='maximum of preceding filling interval; not the stimulus-time volume',
 stimulus_time_volume_ml=float(vv[0]),end_volume_ml=float(vv[-1]),
 endpoint_volume_difference_ml=float(vv[-1]-vv[0]),
 endpoint_volume_difference_percent=float(100*(vv[-1]-vv[0])/vv[0]),
 peak_coupling_pressure_mmhg=float(pp.max()),minimum_coupling_pressure_mmhg=float(pp.min()),
 net_ejected_volume_0d_ml=close['cumulative_out_ml']-start['cumulative_out_ml'],
 artificially_closed=False,pressure_column='endocardial_load_mmhg',first_cycle_omitted=True,
 total_runtime_hours=json.loads((R/'summary.json').read_text())['run_seconds']/3600)
(out/'pv_beat2_metrics.json').write_text(json.dumps(metrics,indent=2))
with (out/'pv_beat2_data.csv').open('w',newline='') as stream:
 writer=csv.writer(stream);writer.writerow(['time_ms','cavity_volume_ml','coupling_pressure_mmhg'])
 writer.writerows(zip(tt*1000,vv,pp))
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
for zoom in [True,False]:
 fig,ax=plt.subplots(figsize=(9.2,7.2));fig.subplots_adjust(left=.12,right=.96,bottom=.22,top=.83)
 color='#d66b1f';ax.plot(vv,pp,color=color,lw=2.5,label=f'Beat 2 | EF {ef:.2f}%')
 ax.scatter(vv[0],pp[0],s=75,color=color,zorder=6,label='Stimulus start')
 ax.scatter(vv[-1],pp[-1],s=75,facecolors='white',edgecolors=color,lw=1.8,zorder=7,label='Recorded end')
 # Temporal arrows along each actual physiological phase, outside startup transient.
 for phase in ['IVC','EJECTION','IVR','FILLING']:
  ids=np.array([i for i,r in enumerate(rows) if float(r['time_s'])>=start['time_s']-1e-12 and r['phase']==phase])
  if len(ids)<4:continue
  a=ids[int(.42*len(ids))];b=ids[min(len(ids)-1,int(.50*len(ids)))];
  if zoom and min(P[a],P[b])<0:continue
  ax.annotate('',xy=(V[b],P[b]),xytext=(V[a],P[a]),arrowprops=dict(arrowstyle='-|>',color=color,lw=1.7,mutation_scale=15))
 ax.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)',xlim=(esv-5,vv.max()+5))
 ax.set_ylim((-5,pp.max()*1.12) if zoom else (min(-5,pp.min()-8),pp.max()*1.12))
 ax.grid(alpha=.22);ax.legend(loc='upper right',framealpha=.95,fontsize=10)
 fig.suptitle('LV pressure-volume trajectory: second beat',y=.97,fontsize=18)
 fig.text(.54,.91,'Deformation-to-conductivity feedback enabled | source Windkessel + MAC/IB',ha='center',fontsize=10.5,color='.3')
 note='Pressure-axis zoom; the full-range companion includes the negative onset transient.' if zoom else 'Full recorded pressure range, including the negative transient at contraction onset.'
 fig.text(.12,.035,f'EDV(max) = {edv:.2f} mL | ESV(min) = {esv:.2f} mL | EF = {ef:.2f}%\n'
  f'Start/end volumes = {vv[0]:.2f}/{vv[-1]:.2f} mL. No artificial loop-closing segment.\n'
  'Pressure denotes the endocardial coupling load.\n'+note,fontsize=9,color='.35',linespacing=1.7)
 name='pv_beat2' if zoom else 'pv_beat2_full_range'
 for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}')
 plt.close(fig)
print(json.dumps(metrics,indent=2))
