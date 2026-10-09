"""Run EP/GPB/Land and fitted two-way LV blood/myocardium coupling.

Fresh runs start from the unloaded reference mesh at the first pacing time.
They validate the coupled integrator; they are not preloaded cardiac cycles.
Three-field checkpoints from prepare_afsei_preload.py start from passive
equilibrium. Earlier IB/one-way checkpoints cannot be resumed here.
"""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from afsi_torch.paper_lv import PaperLVConfig,imported_model
from afsi_torch.config import TimeConfig
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.protocol import ticks
from afsi_torch.moving_cavity import ALEConfig,from_solid
from afsi_torch.two_way import ThreeFieldStepper,MechanicalConfig
from afsi_torch.two_way import checkpoint


def write_vtk(output,driver,state):
    import meshio
    e=state.ep; b=state.blood; m=driver.blood_driver.mesh
    meshio.write(output/'solid_final.vtu',meshio.Mesh(e.fsi.x.cpu().numpy(),[('tetra',driver.mesh.cells.cpu().numpy())],
        point_data={'voltage_mV':e.tissue.voltage_v.cpu().numpy()*1000,'velocity_cm_s':e.fsi.velocity.cpu().numpy(),
                    'displacement_cm':(e.fsi.x-driver.mesh.X).cpu().numpy()}))
    meshio.write(output/'blood_final.vtu',meshio.Mesh(m.coordinates(b.solid_x).cpu().numpy(),[('tetra',m.cells.cpu().numpy())],
        cell_data={'pressure_mmhg':[b.pressure.cpu().numpy()/1333.22368421],
                   'velocity_cm_s':[b.velocity[m.cell_faces].mean(1).cpu().numpy()]}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    p.add_argument('--output',required=True); p.add_argument('--resume')
    p.add_argument('--duration-ms',type=float,default=.2); p.add_argument('--dt-ms',type=float,default=.1)
    p.add_argument('--initial-pc-mmhg',type=float,default=85.); p.add_argument('--atrium-mmhg',type=float,default=8.5)
    p.add_argument('--solid-density',type=float,default=1.,help='myocardium density in g/cm3; explicit dynamic model')
    p.add_argument('--cap-edge-cm',type=float,default=.30); p.add_argument('--no-convection',action='store_true')
    p.add_argument('--write-vtk',action='store_true'); p.add_argument('--checkpoint-every',type=int,default=10)
    a=p.parse_args()
    if a.duration_ms<=0 or a.checkpoint_every<1: p.error('positive duration and checkpoint interval required')
    output=Path(a.output)
    if output.exists() and any(output.iterdir()): p.error('output directory must be new or empty')
    output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); started=time.perf_counter()
    if a.resume:
        d,state,progress=checkpoint.load(a.resume)
    else:
        cfg=AFSEIConfig.source(fsi=PaperLVConfig(source_dir=a.source_dir,time=TimeConfig(a.dt_ms/1000,a.dt_ms/1000)))
        mesh=imported_model(cfg.fsi,'cuda').mesh
        fluid=ALEConfig(dt_s=a.dt_ms/1000,start_time_s=cfg.start_time_s,clock_dt_s=cfg.protocol.dt_ep_s,
            initial_pc_mmhg=a.initial_pc_mmhg,atrium_mmhg=a.atrium_mmhg,convection=not a.no_convection)
        d=ThreeFieldStepper(mesh,cfg,from_solid(mesh,cap_max_edge_cm=a.cap_edge_cm),fluid,MechanicalConfig(density_g_cm3=a.solid_density))
        state=d.initialize_three_fields(); progress=dict(initialization='unloaded reference; no preload',preload_performed=False)
    target=state.ep.tick+ticks(a.duration_ms/1000,d.protocol.dt_ep_s)
    start_step=state.ep.fsi.step; history=[]; status='completed'; failure=None; rejected=[]; force=torch.zeros_like(state.ep.fsi.x)
    checkpoint.save(output/'checkpoint.npz',d,state,progress)
    try:
        while state.ep.tick<target:
            state,diag,force=d.step_three_fields(state,target)
            d.buffer.stage(state.ep.force_load)
            row={k:v for k,v in diag.items() if k not in ('ep',)}
            row.update(ep_steps=diag['ep']['steps'],ode_substeps=diag['ep']['ode_substeps'],
                ta_applied_max_cgs=d.buffer.tension(state.ep.fsi.x).max().item(),ta_next_mode=state.ep.cycle.mode,
                ta_closure_tick=state.ep.cycle.closure_tick)
            history.append(row)
            print(f"AFSEI two-way step={state.ep.fsi.step}, t={diag['time_s']*1000:.3f}ms, V={diag['volume_ml']:.9g}ml, P={diag['pressure_mean_mmhg']:.6g}mmHg, MV/AV={diag['mv_open']}/{diag['av_open']}, solid residual={diag['solid_momentum_residual_norm_dyn']:.3g}/{diag['solid_momentum_tolerance_dyn']:.3g}",flush=True)
            if state.ep.fsi.step%a.checkpoint_every==0: checkpoint.save(output/'checkpoint.npz',d,state,progress)
    except Exception as exc:
        status='failed'; failure=str(exc); rejected=getattr(exc,'attempts',[])
        raise
    finally:
        checkpoint.save(output/'checkpoint.npz',d,state,progress)
        if history:
            with (output/'history.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
        summary=dict(status=status,failure=failure,failed_trial_details=rejected,accepted_steps=state.ep.fsi.step-start_step,
            total_steps=state.ep.fsi.step,reached_time_s=state.ep.fsi.time,elapsed_seconds=time.perf_counter()-started,
            final=history[-1] if history else None,mesh=d.blood_driver.mesh.metadata,
            max_mass_error_ml_s=max((abs(r['mass_error_ml_s']) for r in history),default=0.),
            max_interface_work_error_erg_s=max((abs(r['interface_work_error_erg_s']) for r in history),default=0.),
            max_solid_residual_to_tolerance=max((r['solid_momentum_residual_norm_dyn']/r['solid_momentum_tolerance_dyn'] for r in history),default=0.),
            ep_steps=sum(r['ep_steps'] for r in history),ode_substeps=sum(r['ode_substeps'] for r in history),
            accepted_outlet_closures=sum(r['accepted_outlet_closure'] for r in history),
            two_way_interface_feedback_implemented=True,three_field_integration_implemented=True,
            original_box_fluid_used=False,preload_performed=progress['preload_performed'],
            initialization=progress,
            whole_beat_validated=False,full_energy_balance_validated=False,
            cone_mesh_for_verification_only=True,anatomical_valves_present=False,
            solid_dynamics='lumped P1 inertia; backward Euler',solid_density_g_cm3=d.mechanics.config.density_g_cm3)
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps(summary,indent=2,allow_nan=False),flush=True)
    if a.write_vtk: write_vtk(output,d,state)
    np.savez_compressed(output/'interface_reaction.npz',solid_force_dyn=force.cpu().numpy(),solid_x_cm=state.ep.fsi.x.cpu().numpy(),feedback_applied=np.array(True))


if __name__=='__main__': main()
