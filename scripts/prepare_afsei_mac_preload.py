"""Prepare source 8 mmHg endocardial-traction preload for the original MAC solver.

This creates a MAC checkpoint, with zero initial MAC flow/gauge pressure and
retained prescribed traction. Physical AV/MV/RCR coupling remains pending.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.config import load_config,TimeConfig
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.driver import AFSEIStepper
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_preload import solve,reuse_source_result,initialize
from afsi_torch.two_way.preload import PreloadConfig
from afsi_torch.cycle_checkpoint import atomic_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fsi-config',default=str(Path(__file__).resolve().parents[1]/'configs/afsei_mac.json'))
    p.add_argument('--source-dir')
    p.add_argument('--fluid-shape',type=int)
    p.add_argument('--nonlinear-solver',choices=('newton','anderson-newton'),
        help='original MAC solver mode saved in the new preload checkpoint')
    p.add_argument('--target-pressure-mmhg',type=float,default=8.)
    p.add_argument('--reuse-source-preload',help='checksummed source_solid_preload_bound.npz; a fitted-fluid checkpoint is rejected')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): p.error('output must be new or empty')
    settings=PreloadConfig(target_mmhg=a.target_pressure_mmhg,nominal_steps=100,minimum_step_mmhg=.005)
    fsi=load_config(a.fsi_config,PaperLVConfig)
    from dataclasses import replace
    fsi=replace(fsi,time=TimeConfig(fsi.time.dt,fsi.time.dt))
    if a.source_dir: fsi=replace(fsi,source_dir=a.source_dir)
    if a.fluid_shape is not None: fsi=replace(fsi,fluid=replace(fsi.fluid,shape=(a.fluid_shape,)*3))
    if a.nonlinear_solver is not None: fsi=replace(fsi,nonlinear_solver=a.nonlinear_solver)
    cfg=AFSEIConfig.source(fsi=fsi)
    out.mkdir(parents=True,exist_ok=True)
    started=perf_counter(); status='failed'; failure=None; report={}; result=None
    try:
        mesh=imported_model(cfg.fsi,'cuda').mesh
        driver=AFSEIStepper(mesh,cfg)
        def progress(entry):
            print(f"MAC preload P={entry['pressure_mmhg']:.4g}mmHg V={entry['volume_ml']:.9g}ml residual={entry['residual_norm_dyn']:.3g}",flush=True)
        if a.reuse_source_preload:
            result,strain=reuse_source_result(driver,a.reuse_source_preload,target_mmhg=a.target_pressure_mmhg)
        else:
            result,strain=solve(driver,settings,progress=progress)
        prepared,state,report=initialize(driver,result,strain)
        report.update(source_result_reused=bool(a.reuse_source_preload),
            source_result_sha256=hashlib.sha256(Path(a.reuse_source_preload).read_bytes()).hexdigest() if a.reuse_source_preload else None)
        atomic_json(out/'loading_history.json',result.history)
        checkpoint.save(out/'checkpoint.npz',prepared,state,
            dict(ep_steps=0,ode_substeps=0,elapsed_seconds=0.,preload=report))
        status='completed'
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        summary=dict(status=status,failure=failure,elapsed_seconds=perf_counter()-started,
            preload_settings=asdict(settings),initial_state=report,
            loading_steps=len(result.history) if result else None,
            preload_rejected_attempts=len(result.rejected) if result else None,
            fluid_discretization='cartesian-staggered-MAC',fluid_shape=fsi.fluid.shape,
            source_solid_preload_reused=bool(a.reuse_source_preload),whole_beat_validated=False)
        atomic_json(out/'summary.json',summary)
        print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__': main()
