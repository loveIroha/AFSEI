"""Read-only completed opening-window analysis and figure; no GPU work."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_gao_event_window import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();p=args.directory
    report=json.loads((p/'report.json').read_text());rows=report['records']
    audit=summarize(p)
    if not audit['requested_validation_passed']:
        raise ValueError('the requested actual-opening window has not passed')
    with np.load(p/'initial_checkpoint.npz',allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
    result=dict(audit=audit,circulation=metadata['balanced'],
                elapsed_seconds=report['elapsed_seconds'],advanced_restart=report['advanced_restart_roundtrip'],
                performance={},snapshots=[])
    for name,rr in (('all',rows),('closed',[r for r in rows if r['balanced']['mode']=='closed']),
                    ('ejection',[r for r in rows if r['balanced']['mode']=='rcr'])):
        cost=sum(r['step_seconds'] for r in rr)
        setup=sum(r['nonlinear']['preconditioner_setup_seconds'] for r in rr)
        result['performance'][name]=dict(steps=len(rr),mean_s=cost/len(rr),
            median_s=statistics.median(r['step_seconds'] for r in rr),
            mean_after_first_s=statistics.mean(r['step_seconds'] for r in rr[1:]) if len(rr)>1 else None,
            mean_without_setup_s=(cost-setup)/len(rr),setup_s=setup,setup_fraction=setup/cost,
            rebuilds=sum(r['nonlinear']['preconditioner_rebuilt'] for r in rr),
            mean_newton=statistics.mean(r['nonlinear']['iterations'] for r in rr),
            mean_fluid_calls=statistics.mean(r['nonlinear']['fluid_solves'] for r in rr),
            tangent_refreshes=sum(r['nonlinear']['tangent_refresh_fallback'] for r in rr))
    event=next(r for r in rows if 'PHYSICAL_AV_OPENING' in r['cardiac']['events'])
    first_open=next(r for r in rows if r['balanced']['mode']=='rcr')
    for r in (rows[0],event,first_open,rows[-1]):
        b=r['balanced'];result['snapshots'].append(dict(time_ms=r['time_ms'],phase=r['phase'],
            solved_mode=b['mode'],lv_ml=b['volume_ml'],tract_ml=b['tract_volume_ml'],
            total_ml=r['cardiac']['total_blood_volume_ml'],core_mmhg=b['pressure_core']['mean_mmhg'],
            boundary_mmhg=b['boundary_pressure_mmhg'],pc_mmhg=b['arterial_pc_mmhg'],
            qout_ml_s=b.get('physical_outlet_flow_ml_s',0.),
            cumulative_out_ml=r['cardiac']['cumulative_out_ml']))
    a,b=event,rows[-1]
    result['opening_window_volume_budget']=dict(
        lv_change_ml=b['balanced']['volume_ml']-a['balanced']['volume_ml'],
        tract_change_ml=b['balanced']['tract_volume_ml']-a['balanced']['tract_volume_ml'],
        total_change_ml=b['cardiac']['total_blood_volume_ml']-a['cardiac']['total_blood_volume_ml'],
        ejected_ml=b['cardiac']['cumulative_out_ml']-a['cardiac']['cumulative_out_ml'])
    (p/'opening_analysis.json').write_text(json.dumps(result,indent=2)+'\n')

    t=np.array([r['time_ms'] for r in rows]);opening=event['time_ms']
    core=np.array([r['balanced']['pressure_core']['mean_mmhg'] for r in rows])
    cut=np.array([r['balanced']['boundary_pressure_mmhg'] for r in rows])
    pc=np.array([r['balanced']['arterial_pc_mmhg'] for r in rows])
    cost=np.array([r['step_seconds'] for r in rows])
    setup=np.array([r['nonlinear']['preconditioner_setup_seconds'] for r in rows])
    fig,ax=plt.subplots(3,2,figsize=(12,10),sharex=True,constrained_layout=True)
    ax[0,0].plot(t,core,label='LV core');ax[0,0].plot(t,cut,label='AV upstream / prescribed cut')
    ax[0,0].plot(t,pc,label='Arterial capacitor');ax[0,0].set(ylabel='Pressure [mmHg]',title='Pressure and actual opening')
    ax[0,0].legend(fontsize=8)
    ax[0,1].plot(t,[r['balanced'].get('physical_outlet_flow_ml_s',0.) for r in rows])
    ax[0,1].set(ylabel='Outflow [mL/s]',title='Actual physical-cut flow')
    ax[1,0].plot(t,[r['tension_max_cgs']/1e4 for r in rows],color='C3')
    ax[1,0].set(ylabel='Maximum tension [kPa]',title='Active tension: early systole')
    for field,label in (('volume_ml','LV'),('tract_volume_ml','Tracts')):
        value=np.array([r['balanced'][field] for r in rows])
        ax[1,1].plot(t,value-value[0],label=label)
    ax[1,1].plot(t,[r['cardiac']['total_blood_volume_ml']-rows[0]['cardiac']['total_blood_volume_ml'] for r in rows],label='LV + tracts')
    ax[1,1].set(ylabel='Volume change [mL]',title=f'Changes relative to first record ({t[0]:.1f} ms)');ax[1,1].legend(fontsize=8)
    ax[2,0].plot(t,core-cut,color='C4');ax[2,0].set(ylabel='Core minus boundary [mmHg]',title='Different upstream pressure before/after opening')
    ax[2,1].bar(t,cost-setup,width=.08,label='Other step work')
    ax[2,1].bar(t,setup,bottom=cost-setup,width=.08,color='C1',label='Preconditioner setup')
    ax[2,1].set(ylabel='Wall time [s/step]',title=f'Setup: {setup.sum()/cost.sum()*100:.1f}% of recorded step time');ax[2,1].legend(fontsize=8)
    for x in ax.flat:
        x.axvline(opening,color='k',ls='--',lw=.9);x.grid(alpha=.2)
    for x in ax[-1]:x.set_xlabel('Physical time [ms]')
    fig.suptitle(f'AFSEI physical-cut validation passed: AV opens at {opening:.1f} ms; +2 ms completed',fontsize=13)
    fig.savefig(p/'opening_validation.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
