"""PV trajectories from accepted source-LV history; no artificial closure."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser();parser.add_argument('result');args=parser.parse_args()
root=Path(args.result);out=root/'plots';out.mkdir(exist_ok=True)
rows=list(csv.DictReader((root/'history.csv').open()))
events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines() if s.strip()]
t=np.array([float(r['time_s']) for r in rows]);v=np.array([float(r['volume_ml']) for r in rows])
p=np.array([float(r['endocardial_load_mmhg']) for r in rows])
starts=[e for e in events if e['phase']=='IVC']
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':220,'pdf.fonttype':42})
fig,(ax,full)=plt.subplots(1,2,figsize=(14,7.5),gridspec_kw={'width_ratios':[1.65,1]})
fig.subplots_adjust(bottom=.25,top=.81,left=.075,right=.97,wspace=.25)
fig.suptitle('Left-ventricular pressure-volume trajectories',y=.98,fontsize=19)
pre=t<=starts[0]['time_s'];ax.plot(v[pre],p[pre],color='0.65',ls='--',lw=1.5,label='Initial preload',zorder=1)
full.plot(v[pre],p[pre],color='0.65',ls='--',lw=1.2)
stats=[]
for k,start in enumerate(starts):
    end=starts[k+1]['time_s'] if k+1<len(starts) else t[-1]
    mask=(t>=start['time_s']-1e-12)&(t<=end+1e-12)
    tt,vv,pp=t[mask],v[mask],p[mask]
    closure=next(e for e in events if e['phase']=='IVR' and e['beat']==start['beat'])
    previous_fills=[e for e in events if e['phase']=='FILLING' and e['time_s']<start['time_s']-1e-12]
    fill_start=previous_fills[-1]['time_s'] if previous_fills else 0.
    diastole=(t>=fill_start-1e-12)&(t<=start['time_s']+1e-12)
    contraction=(t>=start['time_s']-1e-12)&(t<=closure['time_s']+1e-12)
    edv_index=np.flatnonzero(diastole)[np.argmax(v[diastole])]
    esv_index=np.flatnonzero(contraction)[np.argmin(v[contraction])]
    edv=float(v[edv_index]);esv=float(v[esv_index]);ef=100*(edv-esv)/edv
    color=plt.get_cmap('tab10')(k)
    ax.plot(vv,pp,color=color,lw=2.5,label=f'Beat {k+1}: EF {ef:.1f}%')
    full.plot(vv,pp,color=color,lw=1.8)
    ax.scatter(vv[0],pp[0],s=55,color=color,zorder=5)
    ax.scatter(vv[-1],pp[-1],s=65,facecolors='white',edgecolors=[color],lw=1.6,zorder=6)
    ax.scatter(edv,p[edv_index],s=65,marker='D',facecolors='white',edgecolors=[color],lw=1.5,zorder=7)
    # Arrows on actual trajectory, selected by arc length after axis scaling.
    arc=np.r_[0,np.cumsum(np.hypot(np.diff(vv)/80,np.diff(pp)/140))]
    for frac in (.12,.40,.64,.86):
        center=np.searchsorted(arc,frac*arc[-1]);left=np.searchsorted(arc,max(0,arc[center]-.018))
        right=min(len(vv)-1,np.searchsorted(arc,arc[center]+.018))
        if right>left:
            ax.annotate('',xy=(vv[right],pp[right]),xytext=(vv[left],pp[left]),
                arrowprops=dict(arrowstyle='-|>',color=color,lw=1.8,mutation_scale=14))
    stats.append(dict(beat=k+1,stimulus_ms=1000*start['time_s'],
        edv_ml=edv,esv_ml=esv,volume_excursion_ml=edv-esv,ef_percent=ef,
        ejected_volume_0d_ml=closure['cumulative_out_ml']-start['cumulative_out_ml'],
        edv_definition='maximum during preceding diastolic filling',
        esv_definition='minimum during contraction through outlet closure',
        edv_time_ms=float(1000*t[edv_index]),esv_time_ms=float(1000*t[esv_index]),
        diastolic_window_start_ms=1000*fill_start,stimulus_volume_ml=start['volume_ml'],
        peak_coupling_pressure_mmhg=float(pp.max()),
        minimum_coupling_pressure_mmhg=float(pp.min()),
        ejection_end_ms=1000*closure['time_s'],recorded_end_ms=1000*end,
        recorded_end_volume_ml=float(vv[-1]),
        following_filling_complete=bool(k+1<len(starts))))
ax.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)')
ax.grid(alpha=.22);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=3,fontsize=10,frameon=False)
ax.set_xlim(min(v)-4,max(v)+5);ax.set_ylim(-5,float(p.max())*1.1)
ax.text(.04,.50,'PV loop detail\n(pressure axis zoom)',transform=ax.transAxes,color='0.4',fontsize=10)
full.set(xlabel='LV cavity volume (mL)',ylabel='LV coupling pressure (mmHg)',title='Full recorded pressure range')
full.grid(alpha=.22);full.set_ylim(min(-3,float(p.min())-5),float(p.max())*1.1)
fig.text(.52,.925,'Original LV mesh | source Windkessel + MAC/IB FSI',ha='center',fontsize=12,color='0.3')
fig.text(.075,.065,'Pressure = endocardial coupling load; not the mean MAC fluid pressure.\n'
    f'EDV = preceding filling maximum; ESV = contraction minimum. Diamonds = EDV. Last time = {1000*t[-1]:.1f} ms.\n'
    'Beat 2 ends during final filling; no smoothing or artificial loop-closing segment.\n'
    'Left: PV loop detail. Right: all recorded values, including negative pressure transients at contraction onset.',
    fontsize=10,color='0.35',linespacing=1.6)
fig.savefig(out/'pv_two_beats.png');fig.savefig(out/'pv_two_beats.pdf');plt.close(fig)
report=dict(result=str(root),pressure_column='endocardial_load_mmhg',last_time_ms=float(1000*t[-1]),
            beats=stats,artificial_closure=False)
(out/'pv_metrics.json').write_text(json.dumps(report,indent=2))
with (out/'pv_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
print(json.dumps(report,indent=2))
