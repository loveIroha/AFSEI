"""Compare accepted same-checkpoint LV connection trials without fitting curves."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def quantities(report):
    rows=report['records']
    b=[r['balanced'] for r in rows]
    return dict(
        time_ms=np.array([r['time_ms'] for r in rows]),
        volume_ml=np.array([r['volume_ml'] for r in b]),
        core_mmhg=np.array([r['pressure_core']['mean_mmhg'] for r in b]),
        outlet_mmhg=np.array([r['boundary_pressure_mmhg'] for r in b]),
        q_ml_s=np.array([r.get('physical_outlet_flow_ml_s',r['measured_relative_q_ml_s']) for r in b]),
        tract_rate_ml_s=np.array([r.get('tract_volume_rate_ml_s',0.) for r in b]),
        mass_error_ml_s=np.array([r['mass_error_ml_s'] for r in b]),
        step_seconds=np.array([r['step_seconds'] for r in rows]),
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    reports={name:json.loads((args.directory/name/'report.json').read_text())
             for name in ('legacy','physical_cut')}
    old,new=reports.values()
    if old['source_sha256']!=new['source_sha256']:
        raise ValueError('comparison sources differ')
    if any(r['status']!='passed' for r in reports.values()):
        raise ValueError('both diagnostic runs must pass before comparison')
    data={k:quantities(r) for k,r in reports.items()}
    if not np.allclose(data['legacy']['time_ms'],data['physical_cut']['time_ms'],rtol=0,atol=1e-9):
        raise ValueError('accepted time grids differ')
    summary={'source':old['source'],'source_sha256':old['source_sha256'],
             'source_time_ms':old['source_time_ms'],'dt_ms':old['dt_ms'],
             'steps':old['accepted_steps'],'cases':{}}
    for name,r in reports.items():
        d=data[name];gap=d['core_mmhg']-d['outlet_mmhg']
        duration=(d['time_ms'][-1]-r['source_time_ms'])/1000
        volume_drop=r['initial_volume_ml']-d['volume_ml'][-1]
        outflow_integral=float(d['q_ml_s'].sum()*r['dt_ms']/1000)
        tract_storage=float(d['tract_rate_ml_s'].sum()*r['dt_ms']/1000)
        summary['cases'][name]=dict(
            end_time_ms=float(d['time_ms'][-1]),initial_volume_ml=r['initial_volume_ml'],
            final_volume_ml=float(d['volume_ml'][-1]),volume_drop_ml=float(volume_drop),
            mean_volume_outflow_ml_s=float(volume_drop/duration),
            final_core_mmhg=float(d['core_mmhg'][-1]),final_outlet_mmhg=float(d['outlet_mmhg'][-1]),
            final_pressure_gap_mmhg=float(gap[-1]),mean_pressure_gap_mmhg=float(gap.mean()),
            final_outflow_ml_s=float(d['q_ml_s'][-1]),
            integrated_outflow_ml=outflow_integral,tract_volume_change_ml=tract_storage,
            integrated_mass_error_ml=float(-volume_drop+tract_storage+outflow_integral),
            max_mass_error_ml_s=float(abs(d['mass_error_ml_s']).max()),
            max_rcr_balance_ml_s=max(abs(x['balanced']['rcr_balance_ml_s']) for x in r['records']),
            elapsed_seconds=r['elapsed_seconds'],
            first_step_seconds=float(d['step_seconds'][0]),
            mean_later_step_seconds=float(d['step_seconds'][1:].mean()),
            mean_newton_iterations=float(np.mean([x['nonlinear']['iterations'] for x in r['records']])),
            mean_fluid_solves=float(np.mean([x['nonlinear']['fluid_solves'] for x in r['records']])),
            mean_final_pressure_cycles=float(np.mean([x['flow']['pressure']['cycles'] for x in r['records']])),
            pressure_backend=r['records'][-1]['balanced']['pressure_backend'],
            migration=r['migration'])
    a,b=summary['cases'].values()
    summary['new_minus_old']={k:b[k]-a[k] for k in (
        'final_core_mmhg','final_outlet_mmhg','final_pressure_gap_mmhg',
        'volume_drop_ml','final_outflow_ml_s')}
    summary['limitations']=[
        'Only a short fixed ejection phase; no EF/EDV/ESV inference.',
        'Physical-cut changes fixed-tract geometry and pressure transfer, not only an RCR parameter.',
        'New geometry requires velocity projection; initial transient is included.',
        'Old sheet pressure and new physical-cut pressure have different meanings.',
        'Boundary flow includes tract storage in the new case; LV volume rate is the common measure.',
        'Different GPUs and initialization cost: this is not a controlled performance benchmark.',
        'Source has the historical checkpoint EP/ODE settings, unchanged between paired cases.'
    ]
    (args.directory/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels={'legacy':'Legacy basal sheet','physical_cut':'Physical cut + fixed tracts'}
    colors={'legacy':'#2563a6','physical_cut':'#ce6b22'}
    for name,d in data.items():
        t=d['time_ms']
        axes[0,0].plot(t,d['core_mmhg'],label=labels[name],color=colors[name])
        axes[0,1].plot(t,d['core_mmhg']-d['outlet_mmhg'],label=labels[name],color=colors[name])
        axes[1,0].plot(t,d['q_ml_s'],label=labels[name],color=colors[name])
        # BE uses end-point flux on each interval. This also makes the
        # LV volume-rate comparison independent of extra tract storage.
        volumes=np.r_[reports[name]['initial_volume_ml'],d['volume_ml']]
        axes[1,0].plot(t,-np.diff(volumes)/(reports[name]['dt_ms']/1000),
                       linestyle='--',color=colors[name],label=labels[name]+' (-dV_LV/dt)')
        axes[1,1].plot(np.r_[reports[name]['source_time_ms'],t],volumes[0]-volumes,label=labels[name],color=colors[name])
    for ax,title,ylabel in zip(axes.flat,
        ['Core pressure','Core minus boundary pressure','Outflow and LV volume rate','LV volume reduction'],
        ['mmHg','mmHg','mL/s','mL']):
        ax.set(title=title,xlabel='Time (ms)',ylabel=ylabel)
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Same-checkpoint connection test: 664.4-666.4 ms\n'
                 'Includes new-geometry migration transient; not a full-cycle comparison',fontsize=12)
    fig.savefig(args.directory/'connection_comparison.png',dpi=170)
    fig.savefig(args.directory/'connection_comparison.pdf')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
