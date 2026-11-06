"""Single-panel PV detail; dashed segments denote graphical closure only."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('result');args=parser.parse_args()
root=Path(args.result);out=root/'plots'
rows=list(csv.DictReader((root/'history.csv').open()))
metrics=json.loads((out/'pv_metrics.json').read_text())
t=np.array([float(r['time_s']) for r in rows]);v=np.array([float(r['volume_ml']) for r in rows])
p=np.array([float(r['endocardial_load_mmhg']) for r in rows])
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
    'savefig.dpi':240,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9.5,7.5))
fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.80)
fig.suptitle('Left-ventricular pressure-volume trajectories',y=.97,fontsize=17)
fig.text(.54,.915,'Original LV mesh | source Windkessel + MAC/IB FSI',ha='center',fontsize=11,color='.35')
pre=t<=metrics['beats'][0]['stimulus_ms']/1000+1e-12
ax.plot(v[pre],p[pre],color='.65',ls='--',lw=1.4,label='Initial preload')
closures=[]
for k,beat in enumerate(metrics['beats']):
    mask=(t>=beat['stimulus_ms']/1000-1e-12)&(t<=beat['recorded_end_ms']/1000+1e-12)
    vv,pp=v[mask],p[mask];color=plt.get_cmap('tab10')(k)
    ax.plot(vv,pp,color=color,lw=2.5,label=f"Beat {k+1}: EF {beat['ef_percent']:.1f}%")
    ax.plot([vv[-1],vv[0]],[pp[-1],pp[0]],ls='--',color=color,lw=2,zorder=5)
    closures.append(dict(beat=k+1,from_volume_ml=float(vv[-1]),to_volume_ml=float(vv[0]),
        from_pressure_mmhg=float(pp[-1]),to_pressure_mmhg=float(pp[0]),graphical_only=True))
    # Indicate temporal direction along the measured positive-pressure branch.
    for a,b in ((.51,.52),(.68,.69),(.87,.88)):
        positive=np.flatnonzero(pp>15.)
        if len(positive):
            i=positive[min(len(positive)-1,int(a*len(positive)))];j=positive[min(len(positive)-1,int(b*len(positive)))]
            ax.annotate('',xy=(vv[j],pp[j]),xytext=(vv[i],pp[i]),
                arrowprops=dict(arrowstyle='-|>',color=color,lw=1.7,mutation_scale=15))
    ax.scatter(vv[0],pp[0],color=color,s=42,zorder=6)
ax.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)',
    xlim=(min(v)-4,max(v)+4),ylim=(-5,float(p.max())*1.1))
ax.grid(alpha=.22)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.15),ncol=3,fontsize=10,frameon=False)
fig.text(.12,.075,'Dashed colored segments close the recorded trajectories for visualization only.\n'
    'EDV: maximum preceding filling volume; ESV: minimum contraction volume.\n'
    'Pressure = endocardial coupling load. Displayed pressure range matches the previous left panel.',
    fontsize=9.5,color='.35',linespacing=1.6)
fig.savefig(out/'pv_two_beats_closed.png');fig.savefig(out/'pv_two_beats_closed.pdf');plt.close(fig)
(out/'pv_graphical_closure.json').write_text(json.dumps(closures,indent=2))
print(str(out/'pv_two_beats_closed.png'))
