"""Read-only summary of full loading history and the short physical-cut EP/FSI run."""
import json
from pathlib import Path
import sys
import numpy as np

root=Path(sys.argv[1])
load=json.loads((root/'loading/report.json').read_text())
run=json.loads((root/'ep_fsi_2ms/report.json').read_text())
config=json.loads((root/'ep_fsi_2ms/configuration.json').read_text())
with np.load(root/'ep_fsi_2ms/checkpoint.npz',allow_pickle=False) as a:
    meta=json.loads(str(a['metadata']))
    voltage=a['voltage_v'];calcium=a['gpb'][:,37]*1000
    stretch=a['stretch'];zs=a['zs'];zw=a['zw']
rows=run['records'];last=rows[-1]
dt=np.diff([run['start_time_ms']]+[r['time_ms'] for r in rows])
step_seconds=np.array([r['step_seconds'] for r in rows])
summary=dict(
    loading=dict(status=load['status'],steps=load['loading_samples'],duration_s=load['duration_s'],
        elapsed_seconds=load['elapsed_seconds'],lv_volume_ml=load['lv_volume_ml'],
        total_blood_volume_ml=load['total_blood_volume_ml'],force_residual_dyn=load['force_residual_dyn'],
        zs_max_abs=load['zs_max_abs'],zw_max_abs=load['zw_max_abs']),
    integration=dict(status=run['status'],start_time_ms=run['start_time_ms'],end_time_ms=run['end_time_ms'],
        accepted_steps=run['accepted_steps'],rejected_attempt_count=meta['rejected_attempts'],
        accepted_dt_ms_min=float(dt.min()),accepted_dt_ms_max=float(dt.max()),
        shortened_intervals=int(np.count_nonzero(dt<.075)),ep_steps=run['ep_steps'],ode_substeps=run['ode_substeps'],
        initial_restart_roundtrip=run['restart_roundtrip'],advanced_restart_roundtrip=run['advanced_restart_roundtrip'],
        full_cycle_completed=run['full_cycle_completed'],
        observed_phases=sorted({r['phase'] for r in rows}),
        max_mass_error_ml_s=run['max_mass_error_ml_s'],max_cumulative_mass_error_ml=run['max_cumulative_mass_error_ml'],
        volume_ml=last['balanced']['volume_ml'],pressure_core_mmhg=last['balanced']['pressure_core']['mean_mmhg'],
        arterial_pc_mmhg=meta['balanced']['arterial_pc_mmhg'],tension_max_cgs=last['tension_max_cgs'],
        voltage_mv_range=[float(voltage.min()*1000),float(voltage.max()*1000)],
        calcium_um_range=[float(calcium.min()),float(calcium.max())],
        ep_nodes=len(voltage),solid_nodes=config['structural_nodes'],fluid_shape=config['fluid_shape'],
        ep_discretization=meta['config']['ep_numerics']['discretization'],
        ep_time_scheme=meta['config']['ep_numerics']['time_scheme'],
        ode_step_ms=meta['config']['dt_ode_ms'],
        elapsed_seconds=run['elapsed_seconds'],mean_accepted_step_seconds=float(step_seconds.mean()),
        mean_after_first_step_seconds=float(step_seconds[1:].mean()),first_step_seconds=float(step_seconds[0])),
    limitations=[
        'Only stimulus-onset isovolumic contraction was physically advanced.',
        'Opening, ejection, closure, 80 ms unloading, filling and the second pacing need longer physical verification.',
        'Controller-event tests do not establish full-cycle physical accuracy.',
        'No updated EF/ESV/EDV comparison or resolution of the historical ~50 mmHg systolic gap.',
        'One adaptive retry occurred; the original short-run logger did not retain its exception detail.',
        'Per-step nonlinear/flow/retry logging was added after this run; existing physical results were not repeated.',
        'Fixed-tract solver performance remains much slower than the original spectral-box connection.'
    ])
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
