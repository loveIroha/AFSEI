"""Short EP/Land + original implicit MAC moving-solid/RCR outlet validation.

No full cardiac phase controller or local wall-seal claim. Fresh connection
retains the saved arterial state unless an explicitly labelled circuit probe
is requested. --resume restores every coupled physical/history setting.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_rcr_coupled import RCRAFSEIStepper,RCRState


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',help='connect a saved closed-cap/EP-FSI state')
    source.add_argument('--resume',help='restore a coupled RCR interval checkpoint')
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--patch-width-cells',type=float)
    p.add_argument('--probe-pc-mmhg',type=float,help='explicit nonphysiological circuit-condition probe')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not 1<=a.steps<=10000:
        p.error('actual CUDA and 1..10000 steps required')
    if a.resume and (a.patch_width_cells is not None or a.probe_pc_mmhg is not None):
        p.error('resume restores saved boundary/circuit settings')
    path=Path(a.resume or a.checkpoint); sha=hashlib.sha256(path.read_bytes()).hexdigest()
    driver,state,progress=checkpoint.load(path)
    if a.resume and not isinstance(state,RCRState):
        p.error('resume requires a coupled RCR checkpoint')
    if not isinstance(driver,RCRAFSEIStepper):
        driver=RCRAFSEIStepper.from_driver(driver); state=driver.connect(state)
    if a.patch_width_cells is not None:
        if state.cap_pressure is not None and a.patch_width_cells!=driver.cap_patch_width_cells:
            p.error('cannot change saved boundary-history partition')
        driver.set_execution(cap_patch_width_cells=a.patch_width_cells)
    if a.probe_pc_mmhg is not None:
        state=replace(state,arterial_pc_mmhg=a.probe_pc_mmhg)
        driver.validate_control(state)
        progress['circuit_condition_probe']=dict(initial_pc_mmhg=a.probe_pc_mmhg,
                                               source_checkpoint=str(path))
    a.output.mkdir(parents=True,exist_ok=False)
    driver.profile=True
    initial_time=state.fsi.time; initial_rejected=state.rejected_attempts
    rows=[]; started=perf_counter()
    previous_elapsed=progress.get('elapsed_seconds',0.)
    try:
        for _ in range(a.steps):
            torch.cuda.synchronize(); begin=perf_counter()
            state,info=driver.step(state,diagnostics=True)
            torch.cuda.synchronize(); elapsed=perf_counter()-begin
            rows.append(dict(step=state.fsi.step,time_s=state.fsi.time,wall_seconds=elapsed,
                rcr=info['rcr'],cap=info['cap'],nonlinear=info['nonlinear'],
                timings=info['timings'],rejected_attempts=info['rejected_attempts']))
            for name in ('ep_steps','ode_substeps'):
                progress[name]=progress.get(name,0)+info['ep']['steps' if name=='ep_steps' else name]
            if len(rows)%50==0:
                progress['elapsed_seconds']=previous_elapsed+perf_counter()-started
                checkpoint.save(a.output/'checkpoint.npz',driver,state,progress)
            r=info['rcr']
            print(f'RCR step={state.fsi.step}, t={state.fsi.time*1000:.3f}ms, '
                f'open={r["outlet_open"]}, Q={r["measured_relative_q_ml_s"]:.6g}ml/s, '
                f'Pc={r["arterial_pc_mmhg"]:.6g}mmHg, '
                f'mass={r["mass_error_ml_s"]:.3g}ml/s, wall={elapsed:.3f}s',flush=True)
    finally:
        progress['elapsed_seconds']=previous_elapsed+perf_counter()-started
        checkpoint.save(a.output/'checkpoint.npz',driver,state,progress)
        report=dict(status='completed' if len(rows)==a.steps else 'incomplete',
            scope=__doc__,source_checkpoint=str(path),source_sha256=sha,
            circuit_condition_probe='circuit_condition_probe' in progress,
            probe_pc_mmhg=progress.get('circuit_condition_probe',{}).get('initial_pc_mmhg'),
            accepted_steps=len(rows),physical_duration_s=state.fsi.time-initial_time,
            reached_time_s=state.fsi.time,rejected_attempts=state.rejected_attempts-initial_rejected,
            elapsed_seconds=perf_counter()-started,execution=driver.execution_settings(),rows=rows,
            gpu=torch.cuda.get_device_name(driver.mesh.X.device),fluid_shape=driver.config.fsi.fluid.shape,
            node_count=len(driver.mesh.X),cell_count=len(driver.mesh.cells),
            mean_seconds_after_first=sum(r['wall_seconds'] for r in rows[1:])/len(rows[1:]) if len(rows)>1 else None,
            step_timing_scope='CUDA synchronized full EP/FSI/RCR step; IO excluded; first includes cold setup',
            ep_advanced=bool(rows),myocardium_motion_advanced=bool(rows),fluid_advanced=bool(rows),
            windkessel_actual_port_flow_connected=bool(rows),
            rcr_inside_original_mac_newton=bool(rows),full_fsi_outer_iterations_added=False,
            pointwise_impermeability_established=False,full_cardiac_cycle_implemented=False,
            physical_cavity_pressure_identification_pending=True,
            three_field_integration_implemented=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        assert sha==hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps({k:v for k,v in report.items() if k not in ('scope','rows')},indent=2))


if __name__=='__main__': main()
