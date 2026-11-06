"""Audit saved LV circulation histories without advancing EP, fluid or solid.

This audit targets the archived 20261012 run (starts at 500 ms; RCR defaults
0.05/1.05/1.3/8). Do not use these assumptions for differently configured runs.
The same-flow RCR replay is a boundary sensitivity calculation, NOT an FSI
prediction: flow, motion, valve events and active tension are held as recorded.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def audit(steps_path, solid_path):
    rows = [json.loads(s) for s in Path(steps_path).read_text().splitlines() if s.strip()]
    with Path(solid_path).open(encoding='utf-8-sig') as stream:
        solid = list(csv.DictReader(stream))
    active = [r for r in solid if r['arterial_initialized'] == '1']
    rc = [(float(r['p_AO_mmHg'])-float(r['p_c_mmHg']))/float(r['Qout_mL_s'])
          for r in active if float(r['Qout_mL_s']) > 1.]
    # Identify C and 1/Rp from the actual accepted source trajectory, Pv=8.
    matrix, rhs = [], []
    for old, new in zip(active, active[1:]):
        dt = (float(new['time_ms'])-float(old['time_ms']))/1000
        matrix.append([(float(new['p_c_mmHg'])-float(old['p_c_mmHg']))/dt,
                       float(new['p_c_mmHg'])-8.])
        rhs.append(float(new['Qout_mL_s']))
    c, conductance = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    source = dict(rc=float(np.median(rc)), rp=float(1/conductance), c=float(c), pv=8.)
    source['ode_fit_max_error_ml_s'] = float(np.max(np.abs(np.asarray(matrix) @ [c, conductance]-rhs)))
    source['rc_spread'] = float(np.ptp(rc))
    replay_pc = rows[0]['balanced']['arterial_pc_old_mmhg']
    previous_time = 500.
    replay, residuals, open_gaps = [], [], []
    for r in rows:
        dt = (r['time_ms']-previous_time)/1000
        if dt <= 0: raise ValueError('history must start after 500 ms and be strictly increasing')
        b = r['balanced']; q = r['q_out_ml_s']; pc = r['pc_mmhg']
        residuals.append(1.3*(pc-b['arterial_pc_old_mmhg'])/dt-q+(pc-8.)/1.05)
        if b['outlet_open']:
            open_gaps.append(b['boundary_pressure_mmhg']-pc-.05*q)
        replay_pc = (source['c']*replay_pc+dt*(q+8./source['rp']))/(source['c']+dt/source['rp'])
        replay.append(replay_pc+source['rc']*q)
        previous_time = r['time_ms']
    beats = []
    for beat in (1, 2):
        start = 500.+800*(beat-1); end = start+800.
        # Assign the interval ending at pacing to the preceding beat.
        indices = [i for i,r in enumerate(rows) if start < r['time_ms'] <= end+1e-7]
        group = [rows[i] for i in indices]
        fill = [r for r in group if r['cardiac']['solved_phase']=='FILLING']
        eject = [r for r in group if r['balanced']['outlet_open']]
        peak_index = max(indices, key=lambda i: rows[i]['pressure_core_mmhg'])
        peak = rows[peak_index]
        closure = sorted({r['cardiac']['closure_tick']*.05 for r in group
                          if r['cardiac']['closure_tick'] >= 0 and r['cardiac']['closure_tick']*.05 > start})
        sf = [r for r in solid if start < float(r['time_ms']) <= end and r['next_phase']=='FILLING']
        se = [r for r in solid if start < float(r['time_ms']) <= end and r['valve_open']=='1']
        sc = sorted({float(r['Ta_closure_ms']) for r in solid
                     if start < float(r['time_ms']) <= end and float(r['Ta_closure_ms'])>start})
        last = group[-1]
        beats.append(dict(beat=beat, closure_ms=closure, source_closure_ms=sc,
            filling_first_solved_endpoint_ms=fill[0]['time_ms'],
            source_filling_entry_ms=float(sf[0]['time_ms']),
            filling_tension_max_cgs=max(r['tension_max_cgs'] for r in fill),
            end_volume_ml=last['volume_ml'], end_core_pressure_mmhg=last['pressure_core_mmhg'],
            end_inlet_pressure_mmhg=last['balanced']['boundary_pressure_mmhg'],
            end_inflow_ml_s=last['q_in_ml_s'],
            peak_core_time_ms=peak['time_ms'], peak_core_mmhg=peak['pressure_core_mmhg'],
            port_at_core_peak_mmhg=peak['balanced']['boundary_pressure_mmhg'],
            capacitor_at_core_peak_mmhg=peak['pc_mmhg'],
            core_minus_port_at_peak_mmhg=peak['pressure_core_mmhg']-peak['balanced']['boundary_pressure_mmhg'],
            same_flow_source_rcr_at_core_peak_mmhg=replay[peak_index],
            ejection_first_core_mmhg=eject[0]['pressure_core_mmhg'],
            ejection_last_core_mmhg=eject[-1]['pressure_core_mmhg'],
            source_ejection_first_pressure_mmhg=float(se[0]['p_LV_mmHg']),
            source_ejection_peak_pressure_mmhg=max(float(r['p_LV_mmHg']) for r in se),
            source_ejection_last_pressure_mmhg=float(se[-1]['p_LV_mmHg'])))
    result = dict(scope='saved-history audit; no new coupled simulation',source_effective_rcr=source,
        afsei_rcr=dict(rc=.05,rp=1.05,c=1.3,pv=8.),
        reconstructed_rcr_max_residual_ml_s=max(map(abs,residuals)),
        open_port_law_max_error_mmhg=max(map(abs,open_gaps)),beats=beats,
        same_flow_replay_is_not_coupled_prediction=True)
    return result, rows, solid, replay


def plot(path, rows, solid, replay):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.array([r['time_ms'] for r in rows])
    st = np.array([float(r['time_ms']) for r in solid])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    axes[0].plot(t, [r['pressure_core_mmhg'] for r in rows], label='AFSEI cavity-core mean')
    axes[0].plot(t, [r['balanced']['boundary_pressure_mmhg'] if r['balanced']['mode']=='rcr' else np.nan for r in rows], label='AFSEI outlet multiplier')
    axes[0].plot(t, [r['pc_mmhg'] for r in rows], label='AFSEI RCR capacitor', ls=':')
    axes[0].plot(st, [float(r['p_LV_mmHg']) for r in solid], label='Solid-only cavity pressure', ls='--')
    axes[0].set_ylabel('Pressure [mmHg]')
    axes[0].legend(ncol=2, fontsize=9)
    axes[1].plot(t, [r['balanced']['boundary_pressure_mmhg'] if r['balanced']['outlet_open'] else np.nan for r in rows], label='Recorded open-outlet pressure')
    axes[1].plot(t, [p if r['balanced']['outlet_open'] else np.nan for r,p in zip(rows,replay)], label='Same-flow RCR replay with source Rc/Rp', ls='--')
    axes[1].set_ylabel('Outlet pressure [mmHg]')
    axes[1].set_title('Offline boundary sensitivity only; flow and motion are NOT recomputed', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[2].plot(t, [r['volume_ml'] for r in rows], label='AFSEI')
    axes[2].plot(st, [float(r['V_LV_mL']) for r in solid], label='Solid-only (record ends at 1850 ms)', ls='--')
    axes[2].set_ylabel('Cavity volume [mL]'); axes[2].set_xlabel('Time [ms]'); axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.25); ax.set_xlim(500,2100)
        ax.axvline(1300,color='gray',lw=.8,ls=':')
    fig.savefig(path,dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=Path,required=True)
    p.add_argument('--solid-pv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    a=p.parse_args(); result,rows,solid,replay=audit(a.steps,a.solid_pv)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if a.plot: plot(a.output/'circulation_audit.png',rows,solid,replay)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
