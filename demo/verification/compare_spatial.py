"""Compare the existing coarse run with completed MAC-only refinements."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/large2/gjh/AFSEI/results')
OUT=ROOT/'reliability_campaign_20261025'

def main():
    cases={'32x32x64':ROOT/'single_tube_full_20261024/analysis',
           '48x48x96':OUT/'mac48','64x64x128':OUT/'mac64_budget16m'}
    initial=json.loads((cases['32x32x64']/'history.json').read_text())[0]
    histories={};summary={};geometries={}
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for label,p in cases.items():
        hs=json.loads((p/'history.json').read_text());hs=[h for h in hs if h['time_s']<=.300000001]
        histories[label]=hs
        report={'status':'completed','time_s':.3} if label.startswith('32x') else json.loads((p/'report.json').read_text())
        summary[label]=dict(status=report['status'],last_recorded_time_s=hs[-1]['time_s'],
            requested_end_reached=report['status']=='completed' and report['time_s']>=.3-1e-12,
            maximum_wall_volume_loss_pct=max(100*(1-h['wall_volume_cm3']/initial['wall_volume_cm3']) for h in hs),
            minimum_equivalent_diameter_mm=min(6*h['minimum_radius_ratio'] for h in hs),
            minimum_sampled_J=min(h['minimum_detF'] for h in hs),
            elapsed_seconds=report.get('elapsed_seconds'),
            seconds_per_step=report.get('mean_seconds_per_step_including_output'))
        pvd=(p/'solid.pvd') if label.startswith('32x') else p/'vtk/solid.pvd'
        geom=[]
        for e in ET.parse(pvd).findall('.//DataSet'):
            t=float(e.attrib['timestep'])
            if t>.300000001:continue
            x=meshio.read(pvd.parent/e.attrib['file']).points.reshape(41,3,32,3)
            center=x[:,0].mean(1);s=np.linspace(0,1,41)[:,None]
            bend=center-((1-s)*center[0]+s*center[-1])
            geom.append(dict(time_s=t,bend_mm=float(np.linalg.norm(bend[:,:2],axis=1).max()*10)))
        geometries[label]=geom
        t=np.array([h['time_s'] for h in hs])*1000
        axs[0,0].plot(t,[6*h['minimum_radius_ratio'] for h in hs],label=label)
        axs[0,1].plot(t,[100*(1-h['wall_volume_cm3']/initial['wall_volume_cm3']) for h in hs],label=label)
        axs[1,0].semilogy([g['time_s']*1000 for g in geom],[max(g['bend_mm'],1e-14) for g in geom],label=label)
        axs[1,1].plot(t,[h['maximum_applied_tension_dyn_cm2']/1e4 for h in hs],label=label)
    common_end=min(h[-1]['time_s'] for h in histories.values())
    common={k:[h for h in hs if h['time_s']<=common_end+1e-10] for k,hs in histories.items()}
    report=dict(scope='MAC grid only; same solid/EP meshes, dt, original material and stimuli; all started from rest',
                cases=summary,common_end_s=common_end,automatic_convergence_claim=False,
                caveat='Spatial refinement also increases adaptive IB quadrature points to preserve its stated density; symmetry-breaking directions may depend on small perturbations.',
                common_histories=common,geometry=geometries)
    (OUT/'spatial_comparison.json').write_text(json.dumps(report,indent=2))
    for ax,y in zip(axs.flat,['Minimum equivalent diameter (mm)','Wall-volume loss from rest (%)','Bending from end chord (mm)','Maximum active tension (kPa)']):
        ax.set(xlabel='Time (ms)',ylabel=y);ax.grid(alpha=.2);ax.legend()
    axs[1,0].set_ylim(1e-12,10)
    fig.suptitle('MAC spatial sensitivity: original single-layer tube')
    fig.tight_layout();fig.savefig(OUT/'spatial_comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))
    return report

if __name__=='__main__':main()
