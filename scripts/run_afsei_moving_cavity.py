"""Moving fitted LV flow: prescribed motion or ONE-WAY EP-FSI geometry bridge.

No two-way blood reaction feedback or preload. Source IB solid dynamics are
retained in EP mode; fitted blood is a separate kinematic/port verification.
"""
import argparse,csv,json,time,os
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from afsi_torch.mesh_io import read_solid_mesh
from afsi_torch.moving_cavity import from_solid,ALEConfig,ALEStepper
from afsi_torch.moving_cavity import checkpoint
from afsi_torch.moving_cavity.bridge import OneWayBridge
from afsi_torch.ep import checkpoint as epcheck
from afsi_torch.ep.protocol import ticks


def vtk(output,driver,state):
    import meshio
    m=driver.mesh; x=m.coordinates(state.solid_x).cpu().numpy()
    meshio.write(output/'blood_final.vtu',meshio.Mesh(x,[('tetra',m.cells.cpu().numpy())],
        cell_data={'pressure_mmhg':[state.pressure.cpu().numpy()/1333.22368421],
                   'velocity_cm_s':[state.velocity[m.cell_faces].mean(1).cpu().numpy()]}))
    boundary=m.tags>=0
    meshio.write(output/'moving_walls_ports.vtu',meshio.Mesh(x,[('triangle',m.faces[boundary].cpu().numpy())],
        cell_data={'port_tag_0_wall_1_mv_2_av':[m.tags[boundary].cpu().numpy()]}))


def main():
    p=argparse.ArgumentParser(description=__doc__,epilog='Resume restores physics/mode/motion parameters. --duration-ms is additional time.')
    p.add_argument('--source-dir',default='/mnt/large2/gjh/realistic_left_ventricle')
    p.add_argument('--output',required=True); p.add_argument('--resume'); p.add_argument('--ep-checkpoint')
    p.add_argument('--ep',action='store_true',help='fresh source-aligned EP-FSI -> ALE one-way run')
    p.add_argument('--duration-ms',type=float,default=2.); p.add_argument('--dt-ms',type=float,default=.1)
    p.add_argument('--cap-edge-cm',type=float,default=.30)
    p.add_argument('--initial-pc-mmhg',type=float,default=85.); p.add_argument('--atrium-mmhg',type=float,default=8.5)
    p.add_argument('--motion-amplitude',type=float,default=1e-5,help='prescribed relative scale amplitude; verification only')
    p.add_argument('--motion-period-ms',type=float,default=2.,help='triangular contraction/expansion verification period')
    p.add_argument('--no-convection',action='store_true',help='explicitly run Stokes-only verification')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda'); p.add_argument('--write-vtk',action='store_true')
    a=p.parse_args()
    if a.duration_ms<=0 or a.motion_period_ms<=0 or not 0<a.motion_amplitude<.01: p.error('positive duration/period and 0 < motion amplitude < .01 required')
    if a.resume and (a.ep or a.ep_checkpoint): p.error('resume already restores mode; do not also specify --ep/--ep-checkpoint')
    output=Path(a.output)
    if output.exists() and any(output.iterdir()): p.error('output must be new or empty')
    output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); started=time.perf_counter()
    bridge=None; progress={}
    if a.resume:
        with np.load(a.resume,allow_pickle=False) as f: producer=json.loads(str(f['metadata']))['producer']
        if producer=='afsei-one-way-ep-ale':
            bridge,pair,progress=checkpoint.load_pair(a.resume,a.device); driver=bridge.blood_driver; state=pair.blood
        else: driver,state,progress=checkpoint.load(a.resume,a.device)
    elif a.ep or a.ep_checkpoint:
        if a.device!='cuda': p.error('EP/GPB/Land requires CUDA')
        if a.ep_checkpoint: ep,epstate,_=epcheck.load(a.ep_checkpoint,a.device)
        else:
            from afsi_torch.paper_lv import PaperLVConfig,imported_model
            from afsi_torch.config import TimeConfig,FluidConfig
            from afsi_torch.ep.config import AFSEIConfig
            from afsi_torch.ep.driver import AFSEIStepper
            fsi=PaperLVConfig(source_dir=a.source_dir,time=TimeConfig(a.dt_ms/1000,a.dt_ms/1000),
                fluid=FluidConfig((32,)*3,(13.,)*3,rho=1.,mu=.04),nonlinear_solver='newton')
            config=AFSEIConfig.source(fsi=fsi); mesh=imported_model(config.fsi,a.device).mesh
            ep=AFSEIStepper(mesh,config); epstate=ep.initialize()
        mesh=from_solid(ep.mesh,cap_max_edge_cm=a.cap_edge_cm)
        cfg=ALEConfig(dt_s=ep.config.fsi.time.dt,start_time_s=epstate.fsi.time,clock_dt_s=ep.protocol.dt_ep_s,
            initial_pc_mmhg=a.initial_pc_mmhg,atrium_mmhg=a.atrium_mmhg,convection=not a.no_convection)
        driver=ALEStepper(mesh,cfg); bridge=OneWayBridge(ep,driver); pair=bridge.initialize(epstate); state=pair.blood
        progress=dict(mode='one-way-ep-fsi-ale')
    else:
        folder=Path(a.source_dir)
        solid=read_solid_mesh(folder/'mesh_scale.xdmf',units='cm',boundaries=folder/'boundaries.xml',tag_map={2:1,1:2,3:3},device=a.device)
        mesh=from_solid(solid,cap_max_edge_cm=a.cap_edge_cm)
        driver=ALEStepper(mesh,ALEConfig(dt_s=a.dt_ms/1000,initial_pc_mmhg=a.initial_pc_mmhg,atrium_mmhg=a.atrium_mmhg,convection=not a.no_convection))
        state=driver.initial_state()
        period=ticks(a.motion_period_ms/1000,driver.config.clock_dt_s)
        if period%2: p.error('motion period must contain an even number of clock ticks')
        progress=dict(mode='prescribed-verification',motion_amplitude=a.motion_amplitude,motion_period_ticks=period)
    if bridge is None and progress.get('mode')!='prescribed-verification': p.error('standalone checkpoint lacks motion protocol')
    target=state.tick+ticks(a.duration_ms/1000,driver.config.clock_dt_s); history=[]; start_step=state.step
    save=lambda: checkpoint.save_pair(output/'checkpoint.npz',bridge,pair,progress) if bridge else checkpoint.save(output/'checkpoint.npz',driver,state,progress)
    save(); status='completed'; failure=None; force=torch.zeros_like(state.solid_x)
    try:
        while state.tick<target:
            if bridge:
                pair,diag,force=bridge.step(pair,target_tick=target); state=pair.blood; entry=dict(diag['blood'])
                ep=bridge.ep_driver; ep.buffer.stage(pair.ep.force_load)
                entry.update(ep_tick=pair.ep.tick,ta_applied_max_cgs=ep.buffer.tension(pair.ep.fsi.x).max().item(),
                    ta_next_mode=pair.ep.cycle.mode,ta_closure_tick=pair.ep.cycle.closure_tick,rejected_attempts=diag['rejected_attempts'])
            else:
                dt=min(driver.config.dt_s,(target-state.tick)*driver.config.clock_dt_s)
                next_tick=state.tick+ticks(dt,driver.config.clock_dt_s)
                phase=(next_tick-round(driver.config.start_time_s/driver.config.clock_dt_s))%progress['motion_period_ticks']
                half=progress['motion_period_ticks']/2; fraction=phase/half if phase<=half else (2*half-phase)/half
                scale=1-progress['motion_amplitude']*fraction
                reference=driver.mesh.reference_solid; center=driver.mesh.coordinates(reference)[-1]
                new_x=center+scale*(reference-center)
                state,entry,force=driver.step(state,new_x,dt)
            history.append(entry)
            if state.step%5==0:
                print(f"ALE step={state.step}, t={entry['time_s']*1000:.3f}ms, V={entry['volume_ml']:.9g}ml, Qin={entry['q_in_ml_s']:.6g}, Qout={entry['q_out_ml_s']:.6g}ml/s, MV/AV={entry['mv_open']}/{entry['av_open']}, mass_error={entry['mass_error_ml_s']:.3g}",flush=True)
    except Exception as exc:
        status='failed'; failure=str(exc)
        raise
    finally:
        save()
        if history:
            with (output/'history.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
        summary=dict(status=status,failure=failure,mode=progress['mode'],accepted_steps=state.step-start_step,total_steps=state.step,
            reached_time_s=state.tick*driver.config.clock_dt_s,elapsed_seconds=time.perf_counter()-started,
            mesh=driver.mesh.metadata,convection=driver.config.convection,final=history[-1] if history else None,
            max_mass_error_ml_s=max((abs(d['mass_error_ml_s']) for d in history),default=0.),
            max_gcl_error_ml_s=max((d['max_gcl_error_ml_s'] for d in history),default=0.),
            max_cell_flux_residual_ml_s=max((d['max_cell_flux_residual_ml_s'] for d in history),default=0.),
            max_interface_work_error_erg_s=max((abs(d['interface_work_error_erg_s']) for d in history),default=0.),
            accepted_outlet_closures=sum(d['accepted_outlet_closure'] for d in history),
            moving_cavity_implemented=True,independent_lumped_valves_implemented=True,ep_geometry_drives_ale_blood=bool(bridge),
            accepted_valve_events_connected_to_ep=bool(bridge),two_way_interface_feedback_implemented=False,
            preload_performed=False,three_field_integration_implemented=False,
            cone_mesh_for_verification_only=True,closed_phase_absolute_pressure_requires_solid_constraint=True,
            original_ep_fsi_ib_fluid_retained_in_one_way_mode=bool(bridge))
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if a.write_vtk: vtk(output,driver,state)
    np.savez_compressed(output/'interface_reaction.npz',solid_force_dyn=force.cpu().numpy(),solid_x_cm=state.solid_x.cpu().numpy(),feedback_applied=np.array(False))
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__': main()
