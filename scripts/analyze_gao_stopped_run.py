"""CPU-only review of the stopped ejection and an existing historical baseline.

This script never loads a GPU solver, advances a state or restarts a job.
"""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path);p.add_argument('baseline',type=Path)
    args=p.parse_args();case=args.case
    r=json.loads((case/'report.json').read_text());rows=r['records']
    a=json.loads((case/'interval_analysis.json').read_text())
    b=json.loads((args.baseline/'report.json').read_text())
    cfg=json.loads((case/'configuration.json').read_text())
    assert r['status']=='failed' and rows
    out=case/'stopped_review';out.mkdir(exist_ok=False)
    n=len(rows);cost=sum(s['step_seconds'] for s in rows)
    setup=sum(s['nonlinear']['preconditioner_setup_seconds'] for s in rows)
    transfer=sum(s['balanced']['transfer_setup_seconds'] for s in rows)
    base_step=b['elapsed_seconds']/b['accepted_steps']
    edv=rows[-1]['cardiac']['edv_ml'];v=rows[-1]['balanced']['volume_ml']
    histories=[h['linear'] for s in rows for h in s['nonlinear']['history'] if 'linear' in h]
    review=dict(status=r['status'],first_cycle_completed=False,first_outlet_closure=a['first_outlet_closure'],
        start_time_ms=r['start_time_ms'],last_accepted_time_ms=rows[-1]['time_ms'],
        accepted_steps=n,advanced_ms=rows[-1]['time_ms']-r['start_time_ms'],
        edv_ml=edv,last_lv_ml=v,volume_reduction_vs_edv_percent=100*(edv-v)/edv,
        final_esv_ml=None,final_ef_percent=None,
        failure=r['failure'],failed_attempts=r['failed_step_attempts'],
        failure_traceback=r['traceback'],last_accepted_checkpoint_saved=r.get('last_accepted_checkpoint_saved'),
        geometry_audit=json.loads((case/'failure_geometry_audit.json').read_text()),
        elapsed_seconds=r['elapsed_seconds'],accepted_solver_seconds=cost,
        outside_accepted_steps_seconds=r['elapsed_seconds']-cost,
        measured_components=dict(preconditioner_setup_seconds=setup,transfer_setup_seconds=transfer,
            remaining_accepted_step_seconds=cost-setup-transfer),
        mean_step_s=cost/n,mean_step_without_preconditioner_s=(cost-setup)/n,
        theoretical_step_without_setup_or_transfer_s=(cost-setup-transfer)/n,
        median_step_s=statistics.median(s['step_seconds'] for s in rows),
        last_ten_step_s=statistics.mean(s['step_seconds'] for s in rows[-10:]),
        preconditioner_rebuilds=sum(s['nonlinear']['preconditioner_rebuilt'] for s in rows),
        total_fluid_responses=sum(s['nonlinear']['fluid_solves'] for s in rows),
        total_jacobian_actions=sum(s['nonlinear']['jacobian_actions'] for s in rows),
        total_tangent_assemblies=sum(s['nonlinear']['tangent_assemblies'] for s in rows),
        total_gmres_iterations=sum(h['iterations'] for h in histories),
        mean_pressure_iterations_last_residual_evaluation=statistics.mean(s['flow']['pressure']['iterations'] for s in rows),
        stage_timings_recorded=any('timings' in s for s in rows),
        phase_bins=[],baseline=dict(path=str(args.baseline),elapsed_seconds=b['elapsed_seconds'],
            accepted_steps=b['accepted_steps'],physical_duration_s=b['reached_time_s'],
            mean_step_s=base_step,fluid_shape=b['fluid_shape'],dt_s=b['dt_s'],
            pressure_backend=b['pressure_backend'],pressure_solver=b['config']['pressure_solver'],
            nonlinear_solver=b['nonlinear_solver'],time_scheme=b['time_scheme'],
            caveat='Historical all-phase prescribed-load run versus current ejection-only EP/RCR/cut-tract run; not a matched-model or matched-accuracy speedup experiment.'),
        current=dict(fluid_shape=cfg['fluid_shape'],
            pressure_solver=cfg['cut_boundary_execution']['pressure_solver_tolerances'],
            pressure_backend=rows[-1]['flow']['pressure']['backend'],
            full_fsi_outer_iterations_added=rows[-1]['balanced']['full_fsi_outer_iterations_added']),
        observed_mean_step_ratio_to_historical=cost/n/base_step,
        pressure_drop_comparison=a['pressure_drop_comparison'],
        accepted_prefix_checks=dict(connection=a['port_audit']['physical_connection_checks_passed'],
            momentum=a['all_momentum_samples_consistent'],max_mass_error_ml_s=a['port_audit']['max_mass_error_ml_s']))
    for lo,hi in ((577.9,620.),(620.,700.),(700.,790.)):
        selected=[s for s in rows if lo<s['time_ms']<=hi]
        seconds=sum(s['step_seconds'] for s in selected)
        review['phase_bins'].append(dict(start_ms=lo,end_ms=min(hi,rows[-1]['time_ms']),steps=len(selected),
            mean_step_s=seconds/len(selected),
            mean_preconditioner_s=sum(s['nonlinear']['preconditioner_setup_seconds'] for s in selected)/len(selected),
            mean_transfer_s=sum(s['balanced']['transfer_setup_seconds'] for s in selected)/len(selected),
            mean_fluid_responses=statistics.mean(s['nonlinear']['fluid_solves'] for s in selected),
            mean_newton_iterations=statistics.mean(s['nonlinear']['iterations'] for s in selected)))
    (out/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    # Copy final, already computed trajectory artifacts; do not rerun analysis.
    for name in ('systolic_pv.png','ejection_interval.png','interval_analysis.json'):
        (out/name).write_bytes((case/name).read_bytes())
    source=json.loads((Path(cfg['request']['resume']).parent/'report.json').read_text())['records'][-1]
    volumes=[source['balanced']['volume_ml']]+[s['balanced']['volume_ml'] for s in rows]
    pressures=[source['balanced']['pressure_core']['mean_mmhg']]+[s['balanced']['pressure_core']['mean_mmhg'] for s in rows]
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ax[0].plot(volumes,pressures,color='#2672b2')
    ax[0].scatter([volumes[-1]],[pressures[-1]],color='crimson',marker='x',s=65,label='Stopped; AV not closed')
    ax[0].set(xlabel='LV volume [mL]',ylabel='Core pressure [mmHg]',title='Accepted ejection prefix; final ESV pending')
    ax[0].legend(fontsize=8)
    bins=review['phase_bins'];labels=[f"{q['start_ms']:g}-{q['end_ms']:g} ms" for q in bins]
    left=[0.]*len(bins)
    for key,label,color in (('mean_preconditioner_s','Preconditioner setup','#e18a2e'),
        ('mean_transfer_s','Transfer setup','#58a86b'),('remaining','Other accepted-step work (not profiled)','#2672b2')):
        values=[q['mean_step_s']-q['mean_preconditioner_s']-q['mean_transfer_s'] if key=='remaining' else q[key] for q in bins]
        ax[1].barh(labels,values,left=left,label=label,color=color)
        left=[x+y for x,y in zip(left,values)]
    for i,q in enumerate(bins):
        ax[1].text(q['mean_step_s']+.15,i,f"{q['mean_step_s']:.2f} s; {q['mean_fluid_responses']:.1f} fluid calls",va='center',fontsize=8)
    ax[1].set(xlabel='Mean wall time per accepted step [s]',title='Measured cost within the same run',xlim=(0,max(left)+6))
    ax[1].legend(fontsize=8,loc='lower right')
    for axis in ax:axis.grid(alpha=.2)
    fig.suptitle('Stopped at 789.95 ms: virtual-port coverage guard; no outlet-closure event')
    fig.savefig(out/'result_and_cost.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in review.items() if k not in ('geometry_audit','failure_traceback','pressure_drop_comparison')},indent=2))


if __name__=='__main__':main()
