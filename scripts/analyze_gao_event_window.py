"""Read-only progress and physical-cut/RCR audit of an accepted-event run."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(path):
    state_file=path/'report.json'
    if not state_file.exists():state_file=path/'progress.json'
    report=json.loads(state_file.read_text()) if state_file.exists() else {}
    rows=[]
    stream=path/'steps.jsonl'
    if stream.exists():
        lines=stream.read_text().splitlines()
        for i,line in enumerate(lines):
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
    if not rows:return dict(status=report.get('status','initializing'),accepted_steps=0)
    with np.load(path/'initial_checkpoint.npz',allow_pickle=False) as a:
        meta=json.loads(str(a['metadata']))
    parameters=meta['balanced']['windkessel'];rc=parameters['rc_mmhg_s_per_ml']
    rp=parameters['rp_mmhg_s_per_ml'];cap=parameters['compliance_ml_per_mmhg'];pv=parameters['venous_mmhg']
    events=[dict(time_ms=r['time_ms'],name=e) for r in rows for e in r['cardiac']['events']]
    open_rows=[r for r in rows if r['balanced']['mode']=='rcr']
    discrepancies=[]
    for row in open_rows:
        r=row['balanced'];q=r['physical_outlet_flow_ml_s'];dt=row['accepted_dt_s']
        expected_pc=(cap*r['arterial_pc_old_mmhg']+dt*(q+pv/rp))/(cap+dt/rp)
        discrepancies.append(dict(pc_error_mmhg=r['arterial_pc_mmhg']-expected_pc,
            pressure_law_error_mmhg=r['prescribed_cut_pressure_mmhg']-r['arterial_pc_mmhg']-rc*q,
            balance_ml_s=cap*(r['arterial_pc_mmhg']-r['arterial_pc_old_mmhg'])/dt-q+(r['arterial_pc_mmhg']-pv)/rp,
            internal_pressure_sheet_mmhg=max(abs(v) for v in r['internal_pressure_sheets_mmhg']),
            core_minus_cut_mmhg=r['pressure_core']['mean_mmhg']-r['prescribed_cut_pressure_mmhg']))
    last=rows[-1];b=last['balanced'];times=np.array([r['step_seconds'] for r in rows])
    result=dict(status=report.get('status','running'),accepted_steps=len(rows),time_ms=last['time_ms'],
        phase=last['phase'],solved_mode=b['mode'],events=events,actual_ejection_steps=len(open_rows),
        lv_volume_ml=b['volume_ml'],tract_volume_ml=b['tract_volume_ml'],
        total_blood_volume_ml=last['cardiac']['total_blood_volume_ml'],
        pressure_core_mmhg=b['pressure_core']['mean_mmhg'],pressure_cut_mmhg=b['boundary_pressure_mmhg'],
        pressure_cut_kind=b['boundary_pressure_kind'],arterial_pc_mmhg=b['arterial_pc_mmhg'],
        outlet_flow_ml_s=b.get('physical_outlet_flow_ml_s',0.),tension_max_cgs=last['tension_max_cgs'],
        mean_step_seconds=float(times.mean()),last_ten_step_seconds=float(times[-10:].mean()),
        max_mass_error_ml_s=max(abs(r['balanced']['mass_error_ml_s']) for r in rows),
        max_cumulative_mass_error_ml=max(abs(r['cardiac']['cumulative_mass_error_ml']) for r in rows),
        new_adaptive_rejections=sum(len(r.get('rejected_attempts',[])) for r in rows),
        valve_branch_retries=sum(bool(r['valve_trial_discarded']) for r in rows),
        event_window=report.get('event_window'),failure=report.get('failure'),
        full_cycle_completed=report.get('full_cycle_completed',False))
    if discrepancies:
        result['ejection_audit']={k:max(abs(d[k]) for d in discrepancies) for k in discrepancies[0] if k!='core_minus_cut_mmhg'}
        result['ejection_audit']['core_minus_cut_range_mmhg']=[min(d['core_minus_cut_mmhg'] for d in discrepancies),max(d['core_minus_cut_mmhg'] for d in discrepancies)]
    else:result['ejection_audit']=None
    audit=result['ejection_audit']
    result['physical_connection_checks_passed']=(None if audit is None else
        audit['pc_error_mmhg']<=1e-7 and audit['pressure_law_error_mmhg']<=1e-7
        and audit['balance_ml_s']<=1e-8 and audit['internal_pressure_sheet_mmhg']<=1e-7
        and result['max_mass_error_ml_s']<=1e-5 and result['max_cumulative_mass_error_ml']<=1e-3
        and min(r['balanced']['physical_outlet_flow_ml_s'] for r in open_rows)>=-1e-5)
    result['requested_validation_passed']=(report.get('status')=='passed'
        and bool((report.get('event_window') or {}).get('completed'))
        and result['physical_connection_checks_passed'] is True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output_directory',type=Path)
    args=p.parse_args();result=summarize(args.output_directory)
    (args.output_directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
