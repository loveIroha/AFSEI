"""Compare four mesh pairs; no EP/ODE/FSI advancement."""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,meshio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/large2/gjh/AFSEI/results')
OUT=ROOT/'solid_space_campaign_20261025'
def main():
    cases={
      'Solid L0 / MAC32':(ROOT/'single_tube_full_20261024/analysis',),
      'Solid L0 / MAC64':(ROOT/'reliability_campaign_20261025/mac64_budget16m',),
      'Solid L1 / MAC32':(OUT/'solid2_mac32/prefix_60ms',OUT/'solid2_mac32/continued_300ms'),
      'Solid L1 / MAC64':(OUT/'solid2_mac64/prefix_60ms',OUT/'solid2_mac64/continued_300ms')}
    all_histories={};statuses={}
    for label,folders in cases.items():
        histories={}
        for folder in folders:
            for h in json.loads((folder/'history.json').read_text()):
                if h['time_s']<=.300000001:histories[round(h['time_s'],10)]=h
        all_histories[label]=[histories[t] for t in sorted(histories)]
        last=folders[-1]/'report.json'
        r=json.loads(last.read_text()) if last.exists() else dict(status='completed',time_s=.3)
        statuses[label]={k:r.get(k) for k in ['status','time_s','accepted_steps','error']}
    end=min(h[-1]['time_s'] for h in all_histories.values())
    result={};fig,axs=plt.subplots(2,2,figsize=(12,8))
    for label,history in all_histories.items():
        hs=[h for h in history if h['time_s']<=end+1e-10]
        v0=hs[0]['wall_volume_cm3'];t=[h['time_s']*1000 for h in hs]
        result[label]=dict(final_time_s=hs[-1]['time_s'],initial_wall_volume_cm3=v0,
            maximum_wall_volume_loss_pct=max(100*(1-h['wall_volume_cm3']/v0) for h in hs),
            final_wall_volume_loss_pct=100*(1-hs[-1]['wall_volume_cm3']/v0),
            minimum_equivalent_diameter_mm=min(6*h['minimum_radius_ratio'] for h in hs),
            final_lumen_ml=hs[-1]['lumen_volume_ml'],history=hs)
        axs[0,0].plot(t,[6*h['minimum_radius_ratio'] for h in hs],label=label)
        axs[0,1].plot(t,[100*(1-h['wall_volume_cm3']/v0) for h in hs],label=label)
        axs[1,0].plot(t,[h['lumen_volume_ml'] for h in hs],label=label)
        axs[1,1].plot(t,[h['maximum_applied_tension_dyn_cm2']/1e4 for h in hs],label=label)
    for ax,y in zip(axs.flat,['Minimum equivalent diameter (mm)','Wall-volume loss from rest (%)','Lumen volume (mL)','Maximum active tension (kPa)']):
        ax.set(xlabel='Time (ms)',ylabel=y);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f'Solid P1 refinement: common 0-{end*1000:.0f} ms window; guarded stops reported separately')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    (OUT/'comparison.json').write_text(json.dumps(dict(common_end_s=end,cases=result,
        trajectory_status=statuses,all_available_histories=all_histories,
        note='Fine-solid trajectories did not reach 300 ms; common-window comparison is not full-cycle convergence.'),indent=2))
    print(json.dumps({k:{p:v for p,v in data.items() if p!='history'} for k,data in result.items()},indent=2))

if __name__=='__main__':main()
