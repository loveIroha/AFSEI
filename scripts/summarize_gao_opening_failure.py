"""Offline summary/figure of the stopped opening experiment; no simulation."""
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    root=parser.parse_args().directory
    report=json.loads((root/'report.json').read_text());rows=report['records']
    t=np.array([r['time_ms'] for r in rows]);seconds=np.array([r['step_seconds'] for r in rows])
    lv=np.array([r['balanced']['volume_ml'] for r in rows]);tract=np.array([r['balanced']['tract_volume_ml'] for r in rows])
    with np.load(root/'checkpoint.npz',allow_pickle=False) as a:meta=json.loads(str(a['metadata']))
    lv0=rows[0]['cardiac']['edv_ml'];total0=rows[0]['cardiac']['total_blood_volume_ml']
    tract0=total0-lv0
    jumps=[]
    for row in rows:
        history=row['nonlinear']['history']
        for a,b in zip(history,history[1:]):
            if a['iteration']==b['iteration']:
                jumps.append(dict(time_ms=row['time_ms'],iteration=a['iteration'],before=a['residual_norm'],
                    reevaluated=b['residual_norm'],absolute_change=abs(b['residual_norm']-a['residual_norm'])))
    summary=dict(status=report['status'],failure=report['failure'],accepted_steps=len(rows),
        time_interval_ms=[report['start_time_ms'],float(t[-1])],elapsed_seconds=report['elapsed_seconds'],
        last_saved_checkpoint_time_ms=meta['time']*1000,
        accepted_seconds_sum=float(seconds.sum()),mean_step_seconds=float(seconds.mean()),
        mean_step_excluding_first_seconds=float(seconds[1:].mean()),
        last_ten_seconds=float(seconds[-10:].mean()),
        mean_newton=float(np.mean([r['nonlinear']['iterations'] for r in rows])),
        mean_fluid_solves=float(np.mean([r['nonlinear']['fluid_solves'] for r in rows])),
        tangent_refresh_fallback_steps=sum(r['nonlinear']['tangent_refresh_fallback'] for r in rows),
        adaptive_rejections_before_terminal_failure=sum(len(r['rejected_attempts']) for r in rows),
        last_core_pressure_mmhg=rows[-1]['balanced']['pressure_core']['mean_mmhg'],
        last_closed_av_upstream_mmhg=rows[-1]['balanced']['boundary_pressure_mmhg'],
        last_arterial_pc_mmhg=rows[-1]['balanced']['arterial_pc_mmhg'],
        last_tension_max_kpa=rows[-1]['tension_max_cgs']/1e4,
        lv_change_from_preload_ml=float(lv[-1]-lv0),tract_change_from_preload_ml=float(tract[-1]-tract0),
        last_lv_ml=float(lv[-1]),last_tract_ml=float(tract[-1]),
        total_blood_volume_range_ml=[float((lv+tract).min()),float((lv+tract).max())],
        max_mass_error_ml_s=max(abs(r['balanced']['mass_error_ml_s']) for r in rows),
        max_cumulative_mass_error_ml=max(abs(r['cardiac']['cumulative_mass_error_ml']) for r in rows),
        max_transfer_power_relative_error=max(r['balanced']['transfer_power_relative_error'] for r in rows),
        max_ep_residual_to_tolerance=max(r['ep']['max_residual_to_tolerance'] for r in rows),
        residual_reevaluation_jumps=jumps,
        limitation='No saved failed-iterate residual vectors; repeatability cause needs an operator-level diagnostic, not a complete rerun.')
    (root/'failure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for value,label in (([r['balanced']['pressure_core']['mean_mmhg'] for r in rows],'LV core'),
                        ([r['balanced']['boundary_pressure_mmhg'] for r in rows],'Closed AV upstream'),
                        ([r['balanced']['arterial_pc_mmhg'] for r in rows],'Arterial capacitor (disconnected by closed AV)')):
        ax[0,0].plot(t,value,label=label,lw=2)
    ax[0,0].set(ylabel='Pressure [mmHg]',xlabel='Time [ms]',title='No physical valve opening occurred')
    ax[0,0].legend(fontsize=8)
    ax[0,1].plot(t,[r['tension_max_cgs']/1e4 for r in rows],color='C3',lw=2)
    ax[0,1].set(ylabel='Maximum active tension [kPa]',xlabel='Time [ms]',title='Early activation; not the cycle peak')
    for value,label in ((lv-lv0,'LV'),(tract-tract0,'Fixed tracts'),(lv+tract-total0,'LV + tracts')):
        ax[1,0].plot(t,value,label=label,lw=2)
    ax[1,0].set(ylabel='Volume change from preload [mL]',xlabel='Time [ms]',title='Closed total blood volume is conserved')
    ax[1,0].legend()
    for row in rows[-3:]:
        h=row['nonlinear']['history']
        ax[1,1].semilogy([v['iteration'] for v in h],[v['residual_norm'] for v in h],'-o',ms=3,label=f"{row['time_ms']:.1f} ms")
    ax[1,1].axhline(rows[-1]['nonlinear']['tolerance'],color='k',ls='--',label='Newton tolerance')
    ax[1,1].set(ylabel='Mixed nonlinear residual norm',xlabel='Newton iteration',title='Last accepted steps; failed iterate not retained')
    ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.25)
    fig.suptitle('Gao physical-cut validation stopped after 513.5 ms: no ejection data',fontsize=14)
    fig.savefig(root/'failure_diagnostics.png',dpi=160);fig.savefig(root/'failure_diagnostics.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='residual_reevaluation_jumps'},indent=2))


if __name__=='__main__':main()
