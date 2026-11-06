"""Plot a fixed-time snapshot of an ongoing accepted-event experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
p.add_argument('--through-ms',type=float,required=True);a=p.parse_args()
rows=[]
for line in (a.directory/'steps.jsonl').read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    if r['time_ms']<=a.through_ms+1e-8:rows.append(r)
t=np.array([r['time_ms'] for r in rows]);setup=np.array([r['nonlinear']['preconditioner_setup_seconds'] for r in rows])
cost=np.array([r['step_seconds'] for r in rows])
fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True,constrained_layout=True)
for values,label in (([r['balanced']['pressure_core']['mean_mmhg'] for r in rows],'LV core'),
    ([r['balanced']['boundary_pressure_mmhg'] for r in rows],'Closed AV upstream'),
    ([r['balanced']['arterial_pc_mmhg'] for r in rows],'Downstream arterial capacitor')):
    ax[0].plot(t,values,lw=2,label=label)
ax[0].set(ylabel='Pressure [mmHg]',title='AV remains closed: upstream pressure below downstream pressure');ax[0].legend(fontsize=9)
ax[1].plot(t,[r['tension_max_cgs']/1e4 for r in rows],color='C3',lw=2)
ax[1].set(ylabel='Maximum active tension [kPa]',title='Active tension is still rising; no cycle peak established')
ax[2].bar(t,cost-setup,width=.085,label='Other accepted-step work')
ax[2].bar(t,setup,bottom=cost-setup,width=.085,color='C1',label='Cap Schur preconditioner setup')
ax[2].set(ylabel='Wall time [s/step]',xlabel='Physical time [ms]',title=f'Preconditioner setup: {setup.sum()/cost.sum()*100:.1f}% of recorded step time')
ax[2].legend(fontsize=9)
for v in ax:v.grid(alpha=.25)
fig.suptitle(f'Gao cut connection: snapshot through {t[-1]:.1f} ms, {len(rows)} accepted steps',fontsize=13)
out=a.directory/f'progress_{a.through_ms:.1f}ms.png';fig.savefig(out,dpi=150);plt.close(fig)
print(str(out))
