"""Solve a passive 8 mmHg preload and save a compatible three-field start.

The input solid geometry remains the constitutive reference. Continuation
does not advance voltage, GPB/Land gates, the pacing clock or circulation.
Optional source loading history updates mechanical stretch and distortion.
The matched initial atrial pressure is the requested cavity pressure.
"""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.config import TimeConfig
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.moving_cavity import ALEConfig,from_solid
from afsi_torch.two_way import ThreeFieldStepper
from afsi_torch.two_way import checkpoint
from afsi_torch.two_way.preload import PassivePreloader,PreloadConfig,initialize_preloaded,SourceLoadingHistory
from afsi_torch.moving_cavity.checkpoint import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    p.add_argument('--target-pressure-mmhg',type=float,default=8.)
    p.add_argument('--initial-pc-mmhg',type=float,default=85.)
    p.add_argument('--pressure-step-mmhg',type=float,default=.5)
    p.add_argument('--history-mode',choices=('source-ramp','static-zero-z'),default='source-ramp',
        help='source-ramp: retain source 100 x 5ms loading Z history; static-zero-z: phase-7 comparison')
    p.add_argument('--dt-ms',type=float,default=.1); p.add_argument('--cap-edge-cm',type=float,default=.30)
    p.add_argument('--output',required=True); p.add_argument('--write-vtk',action='store_true')
    a=p.parse_args()
    if a.target_pressure_mmhg<=0 or a.initial_pc_mmhg<a.target_pressure_mmhg: p.error('positive target and arterial Pc >= cavity pressure required')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): p.error('output directory must be new or empty')
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); started=time.perf_counter()
    cfg=AFSEIConfig.source(fsi=PaperLVConfig(source_dir=a.source_dir,time=TimeConfig(a.dt_ms/1000,a.dt_ms/1000)))
    mesh=imported_model(cfg.fsi,'cuda').mesh
    blood=ALEConfig(dt_s=a.dt_ms/1000,start_time_s=cfg.start_time_s,clock_dt_s=cfg.protocol.dt_ep_s,
        atrium_mmhg=a.target_pressure_mmhg,initial_pc_mmhg=a.initial_pc_mmhg)
    d=ThreeFieldStepper(mesh,cfg,from_solid(mesh,cap_max_edge_cm=a.cap_edge_cm),blood)
    settings=PreloadConfig(target_mmhg=a.target_pressure_mmhg,initial_step_mmhg=a.pressure_step_mmhg,
        maximum_step_mmhg=max(1.,a.pressure_step_mmhg),minimum_step_mmhg=.005 if a.history_mode=='source-ramp' else .03125,
        nominal_steps=100 if a.history_mode=='source-ramp' else 0)
    strain=SourceLoadingHistory(d,a.target_pressure_mmhg) if a.history_mode=='source-ramp' else None
    solver=PassivePreloader(d,settings); history=[]; status='completed'; failure=None; result=None; state=None; report={}
    def progress(entry):
        history.append(entry)
        (out/'loading_history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
        print(f"Preload P={entry['pressure_mmhg']:.6g}mmHg, V={entry['volume_ml']:.9g}ml, force residual={entry['residual_norm_dyn']:.3g}/{entry['tolerance_dyn']:.3g}, Newton={entry['newton_iterations']}, GMRES={entry['linear_iterations']}",flush=True)
    try:
        result=solver.solve(progress,observe_state=strain.observe if strain else None)
        state,report=initialize_preloaded(d,result,strain_history=strain.final() if strain else None)
        checkpoint.save(out/'checkpoint.npz',d,state,report)
        if a.write_vtk:
            from run_afsei_two_way import write_vtk
            write_vtk(out,d,state)
    except Exception as exc:
        status='failed'; failure=str(exc)
        if hasattr(exc,'accepted_x'):
            atomic(out/'last_converged_loading_state.npz',dict(producer='afsei-passive-loading-state',schema=1,
                pressure_mmhg=exc.history[-1]['pressure_mmhg'] if exc.history else 0.,complete_preload=False),
                dict(x_cm=exc.accepted_x.cpu().numpy(),reference_x_cm=mesh.X.cpu().numpy()))
        raise
    finally:
        summary=dict(status=status,failure=failure,target_pressure_mmhg=a.target_pressure_mmhg,
            elapsed_seconds=time.perf_counter()-started,preload_settings=asdict(settings),
            loading_steps=len(history),rejected_attempts=len(result.rejected) if result else None,
            preload_performed=status=='completed' and state is not None,initial_state=report or None,
            final_loading_state=history[-1] if history else None,
            reference_geometry_retained=True,active_stress_during_preload=False,
            ep_advanced_during_preload=False,circulation_advanced_during_preload=False,
            cone_mesh_for_verification_only=True,whole_beat_validated=False)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__': main()
