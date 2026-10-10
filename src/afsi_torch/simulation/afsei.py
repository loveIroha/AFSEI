"""Standalone AFSEI short EP/FSI experiment with restart and numeric history.

Fresh runs use unpreloaded reference geometry. MAC source-traction preload
checkpoints retain the prescribed traction and loading history. The MAC box
pressure boundary is zero, and physical AV/MV/RCR ports remain
pending. Reports distinguish integrated three-field advancement from a
validated physiological full cardiac cycle.
"""
import csv
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import torch
from ..paper_lv import imported_model
from ..cycle_checkpoint import atomic_json
from ..ep.config import AFSEIConfig
from ..ep.driver import AFSEIStepper
from ..ep.protocol import ticks
from ..ep.checkpoint import save as save_checkpoint,load as load_checkpoint
from ..ep.mac_isovolumic import IsovolumicState,IsovolumicAFSEIStepper
from ..ep.mac_preload import MACEndocardialPressure
from ..ep.windkessel import MMHG_TO_DYN_CM2 as PRESSURE_TO_CGS


@torch.no_grad()
def run(*,config=None,output=None,resume=None,duration_s=.002,device='cuda',profile=False,isovolumic=False):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('AFSEI EP/FSI experiment requires an actual CUDA GPU')
    if resume and config is not None:
        raise ValueError('resume restores configuration and all physical states')
    folder = Path(output) if output else Path(resume).parent if resume else Path('results/afsei_ep_fsi')
    started = perf_counter()
    if resume:
        driver,state,progress = load_checkpoint(resume,device)
        config = driver.config
    else:
        if any((folder/name).exists() for name in ('checkpoint.npz','report.json','history.csv')):
            raise ValueError('output contains a run; select a fresh output or resume')
        config = config or AFSEIConfig.source()
        # The inherited passive model is used only as an input reader here.
        mesh = imported_model(config.fsi,device).mesh
        driver = AFSEIStepper(mesh,config)
        state = driver.initialize()
        progress = dict(ep_steps=0,ode_substeps=0,elapsed_seconds=0.)
    if isovolumic and not isinstance(state,IsovolumicState):
        if any((folder/name).exists() for name in ('checkpoint.npz','report.json','history.csv')):
            raise ValueError('enabling isovolumic mode requires a fresh output folder')
        driver = IsovolumicAFSEIStepper.from_driver(driver)
        state = driver.lock(state)
    constrained = isinstance(state,IsovolumicState)
    driver.profile = bool(profile)
    torch.cuda.synchronize(driver.mesh.X.device)
    setup_seconds = perf_counter()-started
    measured_steps = []
    duration_ticks = ticks(duration_s,config.protocol.dt_ep_s)
    if duration_ticks < 1: raise ValueError('positive duration on the EP clock required')
    target = state.tick+duration_ticks
    folder.mkdir(parents=True,exist_ok=True)
    history = []
    if resume and (folder/'history.csv').exists():
        with (folder/'history.csv').open(newline='') as stream:
            history = [r for r in csv.DictReader(stream) if int(r['ep_tick']) <= state.tick]
    elapsed_previous = progress.get('elapsed_seconds',0.)
    if 'failure' in progress:
        progress.setdefault('previous_failures',[]).append(progress.pop('failure'))
    info = {}

    def row():
        # The last solved load is distinct from the newly published strain
        # histories and possibly reset Land ODE at an accepted closure.
        driver.buffer.stage(state.force_load)
        ta = driver.buffer.tension(state.fsi.x)
        diagnostics = driver.model.diagnostics(state.fsi.x)
        if constrained:
            diagnostics.update(cavity_pressure_mmhg=state.cavity_pressure_mmhg,
                additional_endocardial_pressure_cgs=state.cavity_pressure_mmhg*PRESSURE_TO_CGS,
                isovolumic_target_ml=state.target_volume_ml,
                isovolumic_error_ml=MACEndocardialPressure(driver.model).volume(state.fsi.x).item()-state.target_volume_ml,
                arterial_pc_mmhg=state.arterial_pc_mmhg)
        ep_info = info.get('ep',{})
        return dict(step=state.fsi.step,ep_tick=state.tick,time_s=state.fsi.time,
            accepted_dt_s=info.get('accepted_dt_s',0.),activation_mode=state.cycle.mode,
            voltage_min_mv=state.tissue.voltage_v.min().item()*1000,
            voltage_max_mv=state.tissue.voltage_v.max().item()*1000,
            calcium_min_um=state.tissue.gpb[:,37].min().item()*1000,
            calcium_max_um=state.tissue.gpb[:,37].max().item()*1000,
            tension_min_cgs=ta.min().item(),tension_max_cgs=ta.max().item(),
            maximum_displacement_cm=(state.fsi.x-driver.mesh.X).abs().max().item(),
            maximum_fluid_velocity_cm_per_s=max(u.abs().max().item() for u in state.fsi.velocity),
            ep_steps=ep_info.get('steps',0),total_ep_steps=progress['ep_steps'],
            step_wall_seconds=info.get('step_wall_seconds',0.),
            **{name:info.get('nonlinear',{}).get(name,0) for name in
               ('fluid_solves','pressure_cycles','helmholtz_sweeps','residual_evaluations',
                'tangent_assemblies','jacobian_actions','mass_solves','mass_iterations','gmres_iterations')},
            **{name:info.get('timings',{}).get(name,0.) for name in
               ('gpb_land_seconds','monodomain_seconds','load_staging_seconds',
                'implicit_fsi_seconds','strain_feedback_seconds')},
            nonlinear_iterations=info.get('nonlinear',{}).get('iterations',0),
            nonlinear_residual=info.get('nonlinear',{}).get('residual_norm',0.),
            nonlinear_tolerance=info.get('nonlinear',{}).get('tolerance',0.),
            divergence_l2=info.get('divergence_l2',0.),power_error=info.get('power_error',0.),
            rejected_attempts=state.rejected_attempts,**diagnostics)

    def publish(status):
        invocation_elapsed = perf_counter()-started
        progress['elapsed_seconds'] = elapsed_previous+invocation_elapsed
        if not history or int(history[-1]['ep_tick']) != state.tick: history.append(row())
        with (folder/'history.csv').open('w',newline='') as stream:
            fields = list(dict.fromkeys(name for r in history for name in r))
            writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(history)
        save_checkpoint(folder/'checkpoint.npz',driver,state,progress)
        last = history[-1]
        report = dict(status=status,accepted_steps=state.fsi.step,reached_time_s=state.fsi.time,
            reached_ep_tick=state.tick,target_ep_tick=target,ep_steps=progress['ep_steps'],
            ode_substeps=progress['ode_substeps'],rejected_attempts=state.rejected_attempts,
            elapsed_seconds=progress['elapsed_seconds'],gpu=torch.cuda.get_device_name(driver.mesh.X.device),
            node_count=len(driver.mesh.X),cell_count=len(driver.mesh.cells),fluid_shape=driver.config.fsi.fluid.shape,
            fluid_discretization='cartesian-staggered-MAC',
            fsi_solver='original-MAC-response; augmented-Newton' if constrained else 'upstream-BEIBStepper',
            cavity_mode='isovolumic-validation' if constrained else 'prescribed-traction',
            isovolumic_constraint_implemented=constrained,isovolumic_validation_only=constrained,
            windkessel_closed_state_advanced=constrained and state.accepted_closed_steps>0,
            assumed_closed_outlet_for_0d_update=constrained,
            fsi_coupling='implicit-BE-BE; old-position IB templates',
            ep_fsi_coupling='partitioned EP/ODE then implicit FSI',
            performance=dict(scope='this invocation; setup and output excluded from step times',
                profile_enabled=bool(profile),setup_seconds=setup_seconds,
                elapsed_this_invocation_seconds=invocation_elapsed,
                elapsed_previous_seconds=elapsed_previous,
                elapsed_seconds_scope='top-level elapsed_seconds is cumulative through checkpoint ancestry; elapsed sampled before publication IO',
                measured_steps=len(measured_steps),
                first_step_seconds=measured_steps[0]['step_wall_seconds'] if measured_steps else None,
                mean_step_seconds=sum(r['step_wall_seconds'] for r in measured_steps)/len(measured_steps) if measured_steps else None,
                mean_after_first_step_seconds=sum(r['step_wall_seconds'] for r in measured_steps[1:])/len(measured_steps[1:]) if len(measured_steps)>1 else None,
                stage_seconds={name:sum(r.get(name,0.) for r in measured_steps) for name in
                    ('gpb_land_seconds','monodomain_seconds','load_staging_seconds','implicit_fsi_seconds','strain_feedback_seconds')} if profile else {},
                timing_notes='stage times include rejected trials and lazy compilation; synchronized profiling adds overhead; nonlinear counters describe the accepted attempt'),
            direction_mapping=config.direction_mapping,stimulated_nodes=int(driver.stimulus_mask.sum()),
            mechanics_profile=config.mechanics_profile,
            source_mechanics_parameters_aligned=config.mechanics_profile == 'source',
            mechanical_direction_discretization='cell-expression-equivalent DG0',
            config=asdict(config),last_state=last,solid_motion_advanced=state.fsi.step > 0,
            fluid_advanced=state.fsi.step > 0,
            ep_fsi_time_integration_implemented=True,three_field_integration_implemented=False,
            windkessel_ports_connected=False,source_full_cycle_reproduced=False,
            preload_performed=bool(progress.get('preload',{}).get('preload_performed',False)),
            preload=progress.get('preload'),
            prescribed_endocardial_pressure_mmhg=None if constrained else config.prescribed_endocardial_pressure_mmhg,
            reference_traction_mmhg=config.prescribed_endocardial_pressure_mmhg,
            physical_valve_events_detected=False,
            limitations=['MAC outer-box zero-pressure boundaries; no physical AV/MV port seal',
                         'RCR and valve-event hooks not connected to the physical fluid boundary',
                         'partitioned EP then implicit FSI; not monolithic EP-FSI Newton']
                        + (['global volume constraint verified independently; local impermeability and physical valve events pending']
                           if constrained else ['prescribed endocardial traction retained; physical flow-driven cavity pressure pending']
                           if progress.get('preload',{}).get('preload_performed') else
                           ['reference geometry at first stimulus; no 8 mmHg preload'])
                        + (['inherited mechanical coefficients; source profile not selected']
                           if config.mechanics_profile != 'source' else []),
            checkpoint='checkpoint.npz',history='history.csv',progress=progress)
        atomic_json(folder/'report.json',report)
        return report

    if not history: history.append(row())
    try:
        while state.tick < target:
            if profile: torch.cuda.synchronize(driver.mesh.X.device)
            step_started = perf_counter()
            candidate,candidate_info = driver.step(state,target_tick=target,diagnostics=True)
            if profile: torch.cuda.synchronize(driver.mesh.X.device)
            candidate_info['step_wall_seconds'] = perf_counter()-step_started
            # Atomic publication after the complete FSI + strain feedback pass.
            state,info = candidate,candidate_info
            progress['ep_steps'] += info['ep']['steps']
            progress['ode_substeps'] += info['ep']['ode_substeps']
            latest_row = row()
            history.append(latest_row)
            measured_steps.append(latest_row)
            if state.fsi.step % config.fsi.output.log_every == 0:
                latest = history[-1]
                print(f'AFSEI step={state.fsi.step}, time={state.fsi.time*1000:.3f} ms, '
                      f'EP={progress["ep_steps"]}, Ta_max={latest["tension_max_cgs"]:.6g} dyn/cm2, '
                      f'max|dx|={latest["maximum_displacement_cm"]:.3g} cm, '
                      f'max|u|={latest["maximum_fluid_velocity_cm_per_s"]:.3g} cm/s, '
                      f'step_wall={latest["step_wall_seconds"]:.3f}s',flush=True)
                if constrained:
                    print(f'  Iso Pcavity={state.cavity_pressure_mmhg:.6f} mmHg, '
                          f'Vtarget={state.target_volume_ml:.9f} ml, '
                          f'dV={latest["isovolumic_error_ml"]:.3g} ml',flush=True)
            if state.tick < target and state.fsi.step % config.fsi.output.checkpoint_every == 0: publish('running')
        return publish('completed')
    except (Exception,KeyboardInterrupt) as exc:
        progress['failure'] = dict(type=type(exc).__name__,message=str(exc),last_accepted_step=state.fsi.step,
                                   last_accepted_ep_tick=state.tick)
        if hasattr(exc,'attempts'): progress['failure']['attempts'] = exc.attempts
        publish('failed')
        raise
