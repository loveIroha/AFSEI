"""Summarize the named single-tube controls and axial EP resolution study."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root/'analysis';out.mkdir(exist_ok=True)
    names=['ep_fixed','ep_axial_refined','ep_axial_reference','no_stimulus_10ms','passive_60ms','coupled_60ms']
    data={name:json.loads((a.root/name/'validation.json').read_text()) for name in names}
    reports={name:json.loads((a.root/name/'report.json').read_text()) for name in names}
    history={name:json.loads((a.root/name/'history.json').read_text()) for name in names}
    speed=[data[n]['propagation']['axial_conduction_speed_cm_s'] for n in names[:3]]
    missing=[];frame_counts={}
    for n in names:
        for file in (a.root/n/'vtk').glob('*.pvd'):
            entries=ET.parse(file).findall('.//DataSet');frame_counts[f'{n}/{file.name}']=len(entries)
            missing.extend(str(file.parent/e.attrib['file']) for e in entries if not (file.parent/e.attrib['file']).exists())
    coupled=reports['coupled_60ms'];initial=history['coupled_60ms'][0]
    result=dict(cases={n:{k:v for k,v in data[n].items() if k not in ('propagation','mechanics')} for n in names},
        conduction_speed_cm_s=speed,
        successive_speed_relative_difference=[abs(speed[i]-speed[i+1])/speed[i+1] for i in range(2)],
        coupled_final=data['coupled_60ms']['mechanics'],
        lumen_volume_reduction_percent=100*(1-coupled['final']['lumen_volume_ml']/initial['lumen_volume_ml']),
        coupled_mean_seconds_per_step=coupled['mean_seconds_per_step_including_output'],
        coupled_setup_seconds=coupled['setup_seconds'],
        coupled_max_sampled_nonlinear_residual_ratio=max(r['nonlinear_residual']/max(r['nonlinear_tolerance'],1e-30) for r in history['coupled_60ms']),
        coupled_max_sampled_divergence_s_inv=max(r['max_divergence_s_inv'] for r in history['coupled_60ms']),
        controls=dict(no_stimulus_activated_fraction=data['no_stimulus_10ms']['propagation']['activated_fraction'],
            no_stimulus_max_displacement_cm=data['no_stimulus_10ms']['mechanics']['maximum_displacement_cm'],
            passive_activated_fraction=data['passive_60ms']['propagation']['activated_fraction'],
            passive_max_displacement_cm=data['passive_60ms']['mechanics']['maximum_displacement_cm']),
        pvd_frame_counts=frame_counts,missing_pvd_targets=missing,
        interpretation='Short-time functional validation only; no full-cycle, mesh-converged, literature-replication or matched-performance claim')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for n,label in zip(names[:3],['0.5 mm','0.25 mm','0.125 mm']):
        rows=data[n]['propagation']['sections']
        axs[0,0].plot([r['z_cm'] for r in rows],[r['activation_mean_ms'] for r in rows],label=label)
    axs[0,0].set(xlabel='Reference axial position (cm)',ylabel='Activation time (ms)',title='EP axial resolution sensitivity');axs[0,0].legend()
    for n,label in [('ep_fixed','Fixed-wall EP/Land'),('coupled_60ms','Coupled, actual hoop stretch'),('passive_60ms','Active stress disabled')]:
        h=[r for r in history[n] if r['time_s']<=.06000001]
        axs[0,1].plot([r['time_s']*1000 for r in h],[r['maximum_applied_tension_dyn_cm2']/1e4 for r in h],label=label)
    axs[0,1].set(xlabel='Time (ms)',ylabel='Applied tension maximum (kPa)',title='No force amplification');axs[0,1].legend(fontsize=8)
    for n,label in [('coupled_60ms','EP-driven FSI'),('passive_60ms','Active stress disabled'),('no_stimulus_10ms','No stimulus')]:
        h=history[n];axs[1,0].plot([r['time_s']*1000 for r in h],[100*(1-r['minimum_radius_ratio']**2) for r in h],label=label)
    axs[1,0].set(xlabel='Time (ms)',ylabel='Maximum section area reduction (%)',title='Contraction and controls');axs[1,0].legend(fontsize=8)
    values=data['coupled_60ms']['mechanics']['section_area_ratio'];z=np.linspace(0,2,len(values))
    axs[1,1].plot(z,100*(1-np.array(values)));axs[1,1].set(xlabel='Reference axial position (cm)',ylabel='Section area reduction (%)',title='Spatial contraction at 60 ms')
    for ax in axs.flat:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'single_tube_validation.png',dpi=180);fig.savefig(out/'single_tube_validation.svg');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
