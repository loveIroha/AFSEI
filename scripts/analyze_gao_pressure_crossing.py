"""Compare the repaired short segment with existing overlapping accepted data."""
import json,sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);newdir=root/'strict_pressure_crossing'
report=json.loads((newdir/'report.json').read_text());rows=report['records']
old=json.loads((root/'first_opening_20261026/report.json').read_text())
oldrows={round(r['time_ms'],8):r for r in old['records']}
pairs=[(oldrows[round(r['time_ms'],8)],r) for r in rows if round(r['time_ms'],8) in oldrows]
comparison={}
for name,key in (('volume_ml','volume_ml'),('core_pressure_mmhg','pressure_core'),('arterial_pc_mmhg','arterial_pc_mmhg')):
    def value(row):
        v=row['balanced'][key];return v['mean_mmhg'] if key=='pressure_core' else v
    comparison[name]=max((abs(value(a)-value(b)) for a,b in pairs),default=None)
comparison['tension_cgs']=max((abs(a['tension_max_cgs']-b['tension_max_cgs']) for a,b in pairs),default=None)
def mean(rs,group,key):return float(np.mean([r[group][key] for r in rs])) if rs else None
summary=dict(status=report['status'],time_interval_ms=[report['start_time_ms'],report.get('end_time_ms',rows[-1]['time_ms'])],
    crossed_previous_failure=rows[-1]['time_ms']>513.6,accepted_steps=len(rows),
    new_rejections=sum(len(r['rejected_attempts']) for r in rows),failed_attempts=report.get('failed_step_attempts',[]),
    elapsed_seconds=report['elapsed_seconds'],mean_after_first_seconds=float(np.mean([r['step_seconds'] for r in rows[1:]])),
    mean_newton=mean(rows,'nonlinear','iterations'),mean_fluid_calls=mean(rows,'nonlinear','fluid_solves'),
    max_nonlinear_residual_to_tolerance=max(r['nonlinear']['residual_norm']/r['nonlinear']['tolerance'] for r in rows),
    max_mass_error_ml_s=max(abs(r['balanced']['mass_error_ml_s']) for r in rows),
    max_cumulative_mass_error_ml=max(abs(r['cardiac']['cumulative_mass_error_ml']) for r in rows),
    restart_roundtrip=report['restart_roundtrip'],advanced_restart_roundtrip=report.get('advanced_restart_roundtrip'),
    matched_accepted_times=len(pairs),overlap_max_abs_differences=comparison,
    overlap_old_mean_newton=mean([a for a,b in pairs],'nonlinear','iterations'),
    overlap_new_mean_newton=mean([b for a,b in pairs],'nonlinear','iterations'),
    overlap_old_mean_fluid_calls=mean([a for a,b in pairs],'nonlinear','fluid_solves'),
    overlap_new_mean_fluid_calls=mean([b for a,b in pairs],'nonlinear','fluid_solves'),
    failure=report.get('failure'),full_cycle_completed=report.get('full_cycle_completed',False))
(newdir/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
