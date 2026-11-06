"""Report domain-size sensitivity separately from solid-only refinement."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/mnt/large2/gjh/AFSEI/results');OUT=ROOT/'domain_space_campaign_20261025'

def main():
    cases={
      'Original box / Solid L0':(ROOT/'single_tube_full_20261024/analysis',),
      'Original box / Solid L1':(ROOT/'solid_space_campaign_20261025/solid2_mac32/prefix_60ms',ROOT/'solid_space_campaign_20261025/solid2_mac32/continued_300ms'),
      'Larger box / Solid L0':(OUT/'coarse_box2/prefix_60ms',OUT/'coarse_box2/continued_300ms'),
      'Larger box / Solid L1':(OUT/'fine_box2/prefix_60ms',OUT/'fine_box2/continued_300ms')}
    all_h={};reports={};bends={}
    for label,folders in cases.items():
        records={};frames={}
        for p in folders:
            for h in json.loads((p/'history.json').read_text()):
                if h['time_s']<=.300000001:records[round(h['time_s'],10)]=h
            pvd=p/'solid.pvd' if p.name=='analysis' else p/'vtk/solid.pvd'
            for e in ET.parse(pvd).findall('.//DataSet'):
                t=float(e.attrib['timestep'])
                if t<=.300000001:frames[round(t,10)]=pvd.parent/e.attrib['file']
        if 'Solid L0' in label:rings=np.arange(41*3*32).reshape(41,3,32)[:,0]
        else:
            with np.load(folders[-1]/'final_state.npz') as z:rings=z['inner_rings'].copy()
        curve=[]
        for t,p in sorted(frames.items()):
            center=meshio.read(p).points[rings].mean(1);s=np.linspace(0,1,len(center))[:,None]
            bend=center-((1-s)*center[0]+s*center[-1])
            curve.append(dict(time_s=t,bend_mm=float(np.linalg.norm(bend[:,:2],axis=1).max()*10)))
        all_h[label]=[records[t] for t in sorted(records)];bends[label]=curve
        rp=folders[-1]/'report.json'
        reports[label]=json.loads(rp.read_text()) if rp.exists() else dict(status='completed',time_s=.3)
    common_end=min(h[-1]['time_s'] for h in all_h.values());summary={}
    for label,hs in all_h.items():
        common=[h for h in hs if h['time_s']<=common_end+1e-10];v0=hs[0]['wall_volume_cm3']
        summary[label]=dict(status=reports[label]['status'],accepted_end_s=reports[label]['time_s'],
          common_end_s=common_end,common_wall_loss_pct=100*(1-common[-1]['wall_volume_cm3']/v0),
          common_diameter_mm=6*common[-1]['minimum_radius_ratio'],
          last_sample_s=hs[-1]['time_s'],last_sample_wall_loss_pct=100*(1-hs[-1]['wall_volume_cm3']/v0))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for label,hs in all_h.items():
        t=[h['time_s']*1000 for h in hs];v0=hs[0]['wall_volume_cm3'];style='--' if label.startswith('Original') else '-'
        color='tab:blue' if 'Solid L0' in label else 'tab:orange'
        axs[0,0].plot(t,[6*h['minimum_radius_ratio'] for h in hs],style,color=color,label=label)
        axs[0,1].plot(t,[100*(1-h['wall_volume_cm3']/v0) for h in hs],style,color=color,label=label)
        axs[1,0].semilogy([h['time_s']*1000 for h in bends[label]],[max(h['bend_mm'],1e-14) for h in bends[label]],style,color=color,label=label)
        axs[1,1].plot(t,[h['maximum_applied_tension_dyn_cm2']/1e4 for h in hs],style,color=color,label=label)
    for ax,y in zip(axs.flat,['Minimum equivalent diameter (mm)','Wall-volume loss from rest (%)','Bending from end chord (mm)','Maximum active tension (kPa)']):
        ax.set(xlabel='Time (ms)',ylabel=y);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[1,0].set_ylim(1e-12,100)
    fig.suptitle('Domain-size study at fixed h = 0.4375 mm; original fine-solid run stopped at 153.45 ms')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    (OUT/'comparison.json').write_text(json.dumps(dict(summary=summary,histories=all_h,bending=bends,
        scope='Outer boundary positions changed; do not merge with the solid-only refinement conclusion.'),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
