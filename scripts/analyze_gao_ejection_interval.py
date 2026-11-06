"""Read-only pressure, volume and momentum audit for ejection continuations."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--output-directory',type=Path,help='Optional separate directory for analysis artifacts')
    args=p.parse_args();root=args.directory
    output=args.output_directory or root;output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'configuration.json').read_text())['request']
    record_path=root/'steps.jsonl';rows=[]
    for line in record_path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break
    if not rows:raise ValueError('no accepted steps yet')
    audit=summarize(root)
    audit.pop('requested_validation_passed',None)
    source=Path(cfg['resume']).parent
    ref=json.loads((source/'report.json').read_text())['records'][-1]
    with np.load(root/'initial_checkpoint.npz',allow_pickle=False) as a:
        meta=json.loads(str(a['metadata']))
    if abs(ref['time_ms']-meta['time']*1000)>1e-7:
        raise ValueError('source report does not match the actual initial checkpoint')
    initial_lv=ref['balanced']['volume_ml'];initial_tract=ref['balanced']['tract_volume_ml']
    t=np.array([r['time_ms'] for r in rows]);dt=np.array([r['accepted_dt_s'] for r in rows])
    lv=np.array([r['balanced']['volume_ml'] for r in rows])
    tract=np.array([r['balanced']['tract_volume_ml'] for r in rows])
    q=np.array([r['balanced'].get('physical_outlet_flow_ml_s',0.) for r in rows])
    lv_rate=np.diff(np.r_[initial_lv,lv])/dt
    tract_rate=np.diff(np.r_[initial_tract,tract])/dt
    out=np.array([r['cardiac']['cumulative_out_ml']-ref['cardiac']['cumulative_out_ml'] for r in rows])
    samples=[r['cut_momentum_budget'] for r in rows if 'terms_mmhg' in r.get('cut_momentum_budget',{})]
    errors=[dict(time_ms=r['time_ms'],error=r['cut_momentum_diagnostic_error']) for r in rows
            if 'cut_momentum_diagnostic_error' in r]
    negative=np.flatnonzero(lv_rate < -1e-5);below=np.flatnonzero(lv < initial_lv-1e-8)
    setup=sum(r['nonlinear']['preconditioner_setup_seconds'] for r in rows)
    cost=sum(r['step_seconds'] for r in rows);observer=sum(s['diagnostic_seconds'] for s in samples)
    result=dict(port_audit=audit,start_time_ms=meta['time']*1000,end_time_ms=float(t[-1]),
        initial_lv_ml=initial_lv,final_lv_ml=float(lv[-1]),lv_change_ml=float(lv[-1]-initial_lv),
        tract_change_ml=float(tract[-1]-initial_tract),ejected_ml=float(out[-1]),
        volume_balance_error_ml=float(lv[-1]-initial_lv+tract[-1]-initial_tract+out[-1]),
        first_negative_lv_rate_ms=None if not len(negative) else float(t[negative[0]]),
        first_below_initial_lv_ms=None if not len(below) else float(t[below[0]]),
        latest_lv_rate_ml_s=float(lv_rate[-1]),latest_tract_rate_ml_s=float(tract_rate[-1]),
        momentum_samples=len(samples),momentum_errors=errors,
        all_momentum_samples_consistent=bool(samples) and not errors and all(s['consistent_with_scheme'] for s in samples),
        maximum_momentum_closure_mmhg=None if not samples else max(abs(s['closure_error_mmhg']) for s in samples),
        latest_momentum=None if not samples else samples[-1],
        window_peaks=dict(
            core_pressure=dict(time_ms=max(rows,key=lambda r:r['balanced']['pressure_core']['mean_mmhg'])['time_ms'],
                value_mmhg=max(r['balanced']['pressure_core']['mean_mmhg'] for r in rows)),
            outlet_flow=dict(time_ms=max(rows,key=lambda r:r['balanced'].get('physical_outlet_flow_ml_s',0.))['time_ms'],
                value_ml_s=float(q.max())),
            active_tension=dict(time_ms=max(rows,key=lambda r:r['tension_max_cgs'])['time_ms'],
                value_kpa=max(r['tension_max_cgs'] for r in rows)/1e4)),
        performance=dict(mean_solver_step_s=cost/len(rows),median_solver_step_s=statistics.median(r['step_seconds'] for r in rows),
            mean_without_preconditioner_setup_s=(cost-setup)/len(rows),setup_fraction=setup/cost,
            full_rebuilds=sum(r['nonlinear']['preconditioner_rebuilt'] for r in rows),
            mean_newton_iterations=statistics.mean(r['nonlinear']['iterations'] for r in rows),
            mean_fluid_calls=statistics.mean(r['nonlinear']['fluid_solves'] for r in rows),
            observer_seconds=observer,mean_step_plus_observer_s=(cost+observer)/len(rows)))
    requested_end=meta['time']*1000+cfg['duration_ms']
    event_window=audit.get('event_window')
    result['stop_condition_kind']='accepted-event' if cfg.get('stop_after_event') else 'duration'
    result['duration_upper_bound_end_ms']=requested_end
    result['interval_completed']=bool(audit['status']=='passed' and
        (bool(event_window and event_window.get('completed')) if cfg.get('stop_after_event')
         else abs(t[-1]-requested_end)<1e-6))
    result['interval_and_diagnostics_passed']=bool(result['interval_completed']
        and audit['physical_connection_checks_passed'] is True and result['all_momentum_samples_consistent'])
    open_rows=[r for r in rows if r['balanced']['mode']=='rcr'
               and r['balanced'].get('prescribed_cut_pressure_mmhg') is not None]
    def drop_record(row):
        b=row['balanced']
        return dict(time_ms=row['time_ms'],core_mmhg=b['pressure_core']['mean_mmhg'],
            cut_mmhg=b['prescribed_cut_pressure_mmhg'],
            gap_mmhg=b['pressure_core']['mean_mmhg']-b['prescribed_cut_pressure_mmhg'],
            qout_ml_s=b['physical_outlet_flow_ml_s'])
    if open_rows:
        peak_core=max(open_rows,key=lambda r:r['balanced']['pressure_core']['mean_mmhg'])
        peak_flow=max(open_rows,key=lambda r:r['balanced']['physical_outlet_flow_ml_s'])
        peak_gap=max(open_rows,key=lambda r:r['balanced']['pressure_core']['mean_mmhg']-r['balanced']['prescribed_cut_pressure_mmhg'])
        result['pressure_drop_comparison']=dict(initial=drop_record(ref),peak_core=drop_record(peak_core),
            peak_flow=drop_record(peak_flow),peak_gap=drop_record(peak_gap),last_open=drop_record(open_rows[-1]),
            post_core_peak_span_ms=open_rows[-1]['time_ms']-peak_core['time_ms'],
            core_drop_from_window_peak_mmhg=peak_core['balanced']['pressure_core']['mean_mmhg']-open_rows[-1]['balanced']['pressure_core']['mean_mmhg'])
    closure=[r for r in rows if 'OUTLET_CLOSURE_TA_SNAPSHOT' in r['cardiac']['events']]
    result['first_outlet_closure']=(None if not closure else dict(time_ms=closure[0]['time_ms'],
        edv_ml=closure[0]['cardiac']['edv_ml'],esv_ml=closure[0]['cardiac']['esv_ml'],
        ef_percent=100*(closure[0]['cardiac']['edv_ml']-closure[0]['cardiac']['esv_ml'])/closure[0]['cardiac']['edv_ml']))
    (output/'interval_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(2,2,figsize=(12,8),sharex=True,constrained_layout=True)
    for key,label in (('pressure_core','LV core'),('boundary_pressure_mmhg','Cut'),('arterial_pc_mmhg','Arterial capacitor')):
        values=[r['balanced'][key]['mean_mmhg'] if key=='pressure_core' else r['balanced'][key] for r in rows]
        ax[0,0].plot(t,values,label=label)
    ax[0,0].set(ylabel='Pressure [mmHg]',title='Pressure (closed cut: upstream trace)');ax[0,0].legend(fontsize=8)
    for value,label in ((lv-initial_lv,'LV'),(tract-initial_tract,'Tracts'),(lv-initial_lv+tract-initial_tract,'LV + tracts'),(-out,'Minus measured outflow integral')):
        ax[0,1].plot(t,value,label=label)
    ax[0,1].set(ylabel='Volume change [mL]',title='Independent volume / actual cut-flow budget');ax[0,1].legend(fontsize=8)
    for value,label in ((q,'Actual Qout'),(-lv_rate,'-dV_LV/dt'),(-lv_rate-tract_rate,'-d(V_LV + V_tract)/dt')):
        ax[1,0].plot(t,value,label=label)
    ax[1,0].set(ylabel='Flow / volume rate [mL/s]',title='LV shrinkage versus tract storage');ax[1,0].legend(fontsize=8)
    if samples:
        ts=[s['time_ms'] for s in samples]
        for key in samples[0]['terms_mmhg']:
            ax[1,1].plot(ts,[s['terms_mmhg'][key] for s in samples],label=key)
        ax[1,1].plot(ts,[s['measured_drop_mmhg'] for s in samples],'k--',lw=2,label='Measured neck to cut drop')
    ax[1,1].set(ylabel='Pressure contribution [mmHg]',title='Original BE/projection momentum identity');ax[1,1].legend(fontsize=8)
    for x in ax.flat:x.grid(alpha=.2)
    for x in ax[-1]:x.set_xlabel('Physical time [ms]')
    fig.suptitle(f'AFSEI continuation: {meta["time"]*1000:.1f} to {t[-1]:.1f} ms; status={audit["status"]}')
    fig.savefig(output/'ejection_interval.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    ax[0].plot(np.r_[initial_lv,lv],np.r_[ref['balanced']['pressure_core']['mean_mmhg'],
        [r['balanced']['pressure_core']['mean_mmhg'] for r in rows]])
    ax[0].scatter([initial_lv,lv[-1]],[ref['balanced']['pressure_core']['mean_mmhg'],rows[-1]['balanced']['pressure_core']['mean_mmhg']],s=25)
    ax[0].set(xlabel='LV volume [mL]',ylabel='Core pressure [mmHg]',title='Ejection segment; not a closed PV loop')
    if open_rows:
        ax[1].plot([r['time_ms'] for r in open_rows],[drop_record(r)['gap_mmhg'] for r in open_rows],label='Core - cut')
    if samples:
        ax[1].plot([r['time_ms'] for r in samples],[r['terms_mmhg']['unsteady'] for r in samples],label='Tract unsteady contribution')
        ax[1].plot([r['time_ms'] for r in samples],[r['measured_drop_mmhg'] for r in samples],label='Neck - cut')
    ax[1].set(xlabel='Physical time [ms]',ylabel='Pressure difference [mmHg]',title='Pressure gap versus acceleration contribution');ax[1].legend(fontsize=8)
    for axis in ax:axis.grid(alpha=.2)
    fig.suptitle('Original physical trajectory; no pressure offsets or artificial loop closure')
    fig.savefig(output/'systolic_pv.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
