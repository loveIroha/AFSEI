"""Render saved audit JSON only; no geometry rebuild or operator solve."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
f=json.loads((a.directory/'audit.json').read_text());j=json.loads((a.directory/'jet_geometry.json').read_text())
g=f['geometry'];profile=j['profile'];z=[r['z_cm'] for r in profile]
labels={'standard_fe_and_support':'FE + support','paired_cap_reaction':'Cap pair',
        'pressure_mode_correction':'Mode correction','prescribed_cut_value':'Cut value',
        'saved_history_transport_remainder':'History remainder'}
fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
ax[0,0].plot(z,[r['pressure_mean_mmhg'] for r in profile],label='AV-column mean')
ax[0,0].axhline(f['core_diagnostic']['recomputed_endpoint_core_mmhg'],color='C3',ls='--',label='LV core mean')
ax[0,0].set(ylabel='Pressure [mmHg]',xlabel='Axial coordinate [cm]',title='Pressure drop and downstream recovery')
for key,label in (('mean_cm_s','Mean'),('maximum_cm_s','Maximum'),('minimum_cm_s','Minimum')):
    ax[0,1].plot(z,[r[key] for r in profile],label=label)
ax[0,1].set(ylabel='Axial velocity [cm/s]',xlabel='Axial coordinate [cm]',title='Saved jet: opening / tract area = 0.651')
for x in ax[0]:
    x.axvspan(g['basal_cap_min_cm'],g['basal_cap_max_cm'],color='gray',alpha=.15,label='Basal cap band')
    x.axvline(g['diagnostic_split_cm'],color='gray',ls=':',label='Support envelope end')
    x.legend(fontsize=8)
names=list(f['force_decomposition']);values=[f['force_decomposition'][k]['total_mmhg'] for k in names]
ax[1,0].barh([labels[k] for k in names]+['Net'],values+[f['reconstructed_ib_pressure_mmhg']],
             color=['C0','C1','C2','k'])
ax[1,0].axvline(0,color='gray',lw=1)
ax[1,0].set(xlabel='AV-column integrated contribution [mmHg]',title='Base force pair: large signed cancellation')
names=list(f['pressure_components'])
ax[1,1].barh([labels[k] for k in names],[f['pressure_components'][k]['core_minus_neck_mmhg'] for k in names])
ax[1,1].axvline(0,color='gray',lw=1)
ax[1,1].set(xlabel='Core minus fixed neck [mmHg]',title='Frozen-operator pressure responses (signed)')
for x in ax.flat:x.grid(alpha=.2)
fig.suptitle(f'AFSEI {f["time_ms"]:.1f} ms force / opening audit; no physical advance',fontsize=14)
fig.savefig(a.directory/'audit_summary.png',dpi=150);plt.close(fig)
print(str(a.directory/'audit_summary.png'))
