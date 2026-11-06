"""Read existing checkpoints only: spatial distribution of local J defects."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/mnt/large2/gjh/AFSEI/results/volume_kappa_sensitivity_20261025')
d=json.loads((root/'design.json').read_text())
init=np.load(root/'candidate/initial_state.npz');X=init['reference'];cells=init['cells'];coarse=init['ep_cells']
v=X[cells];D=(v[:,1:]-v[:,:1]).transpose(0,2,1);det=np.linalg.det(D);volume=abs(det)/6
z=(v[:,:,2].mean(1)-.4)/2
# Fine cells retain original-parent row order, eight children per parent.
if len(cells)!=len(coarse)*8:raise AssertionError('expected nested factor2 parent order')
inner_parent=((coarse//32)%3).min(1)==0;inner=np.repeat(inner_parent,8)
regions={'end_10pct_each':(z<.1)|(z>.9),'middle_80pct':(z>=.1)&(z<=.9),
         'inner_reference_half':inner,'outer_reference_half':~inner}
paths={'baseline_300ms':Path(d['baseline'])/'continued_300ms/final_state.npz','candidate_300ms':root/'candidate/final_state.npz'}
results={}
for name,path in paths.items():
    a=np.load(path);x=a['x'];dc=(x[cells[:,1:]]-x[cells[:,:1]]).transpose(0,2,1)
    J=np.linalg.det(dc)/det;error=J-1;E=volume*error**2;total=E.sum()
    rows={}
    for region,mask in regions.items():
        w=volume[mask];e=error[mask]
        rows[region]=dict(reference_volume_fraction=float(w.sum()/volume.sum()),
            J_rms_percent=float(100*np.sqrt(np.sum(w*e**2)/w.sum())),
            squared_J_error_share_percent=float(100*E[mask].sum()/total),
            region_fraction_abs_Jerror_gt10pct=float(w[abs(e)>.1].sum()/w.sum()))
    results[name]=dict(regions=rows,J_rms_percent=float(100*np.sqrt(total/volume.sum())),
        expansion_contribution_percent=float(100*np.sum(volume*np.maximum(error,0))/volume.sum()),
        compression_contribution_percent=float(-100*np.sum(volume*np.minimum(error,0))/volume.sum()))
report=dict(scope='reference-coordinate region attribution at300ms; no time advancement or new experiment',
            definitions={'end_zone':'first and last10% in reference z; combined20% reference volume',
                         'radial_halves':'inherited inner/outer original reference radial layer; not physical face-distance bands'},results=results)
(root/'regions.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained');labels=['End zones','Middle80%'];keys=['end_10pct_each','middle_80pct']
pos=np.arange(2)
for delta,(name,color,label) in zip([-.18,.18],[('baseline_300ms','#777777','53.85 kPa'),('candidate_300ms','#176d9c','73.08 kPa')]):
    r=results[name]['regions']
    for ax,metric in zip(axes,['J_rms_percent','squared_J_error_share_percent']):
        bars=ax.bar(pos+delta,[r[k][metric] for k in keys],width=.36,color=color,label=label)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
for ax,label in zip(axes,['Local J RMS (%)','Share of integrated squared J error (%)']):
    ax.set_xticks(pos,labels);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
fig.suptitle('Remaining local-volume error at300ms: reference-region attribution')
for ext in ('png','pdf'):fig.savefig(root/f'regions.{ext}',dpi=190)
plt.close(fig);print(json.dumps(report,indent=2),flush=True)
