"""Summarize near-preload probes including failed acceptance diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--cases',nargs='+',default=['legacy','physical_cut'])
    p.add_argument('--prefix',default='')
    args=p.parse_args();root=args.directory
    reports={name:json.loads((root/name/'report.json').read_text()) for name in args.cases}
    if len({r['source_sha256'] for r in reports.values()})!=1:raise ValueError('different sources')
    first=next(iter(reports.values()))
    summary=dict(source=first['source'],source_time_ms=first['source_time_ms'],
        purpose='pressure initialization compatibility; no EF/EDV/ESV measurement',cases={})
    for name,r in reports.items():
        last=r['records'][-1] if r['records'] else None
        summary['cases'][name]=dict(status=r['status'],steps=r['accepted_steps'],
            failure=r.get('failure'),elapsed_seconds=r['elapsed_seconds'],
            hydrostatic_probe=r.get('hydrostatic_probe'),static_load_audit=r.get('static_load_audit'),
            migration=r.get('migration'),static_correction=r.get('static_correction'),
            initial_volume_ml=r['initial_volume_ml'],ep_gpb_land_advanced=r['ep_gpb_land_advanced'])
        if last:
            b=last['balanced']
            summary['cases'][name].update(
                final_core_mmhg=b['pressure_core']['mean_mmhg'],
                volume_change_ml=b['volume_ml']-r['initial_volume_ml'],
                final_inflow_ml_s=b['inlet_flow_ml_s'],
                rim_mean_axial_change_mm=last['rim_mean_axial_change_cm']*10,
                tract_volume_change_ml=last['tract_volume_ml']-r['records'][0]['balanced'].get(
                    'tract_volume_old_ml',last['tract_volume_ml']-last['tract_volume_rate_ml_s']*r['dt_ms']/1000),
                max_tension_cgs=max(x['tension_max_cgs'] for x in r['records']),
                max_mass_error_ml_s=max(abs(x['balanced']['mass_error_ml_s']) for x in r['records']))
    (root/f'{args.prefix}comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name,r in reports.items():
        if not r['records']:continue
        t=np.array([x['time_ms']-r['source_time_ms'] for x in r['records']]);b=[x['balanced'] for x in r['records']]
        label={'legacy':'Legacy sheet','physical_cut':'Physical cut (old preload)',
               'uncorrected_passive':'Old preload, passive',
               'corrected_passive':'Corrected preload, passive'}.get(name,name)
        axes[0,0].plot(t,[x['pressure_core']['mean_mmhg'] for x in b],label=label)
        axes[0,1].plot(t,[x['volume_ml']-r['initial_volume_ml'] for x in b],label=label)
        axes[1,0].plot(t,[x['rim_mean_axial_change_cm']*10 for x in r['records']],label=label)
        axes[1,1].plot(t,[x['inlet_flow_ml_s'] for x in b],label=label)
    for ax,title,unit in zip(axes.flat,['Core pressure','LV volume change','Basal axial motion','Reservoir inflow'],
                             ['mmHg','mL','mm','mL/s']):
        ax.set(title=title,xlabel='Elapsed time (ms)',ylabel=unit);ax.grid(alpha=.25);ax.legend(fontsize=8)
        ax.ticklabel_format(useOffset=False,axis='both')
    fig.suptitle('Near-preload compatibility: fixed 8 mmHg reservoir\n'
                 'Original basal support retained; passive/EP status recorded per case')
    fig.savefig(root/f'{args.prefix}preload_comparison.png',dpi=170)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
