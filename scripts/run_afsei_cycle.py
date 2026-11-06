"""Run complete cardiac cycles with source EP/Land and implicit Cartesian MAC FSI."""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_cycle import CardiacStepper,PHASES
from afsi_torch.ep.protocol import ticks
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.circulation_config import CardiacCirculationConfig, circulation_metadata
from afsi_torch.config import load_config
from afsi_torch.circulation.geometry import PortGeometry


def publish_json(path,data):
    temporary=path.with_suffix(path.suffix+'.working')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preload',type=Path); g.add_argument('--resume',type=Path)
    p.add_argument('--reference-checkpoint',type=Path)
    p.add_argument('--cycles',type=int,help='fresh run only; default 2')
    p.add_argument('--period-ms',type=float,help='fresh run only; default 800 ms')
    p.add_argument('--filling-hold-ms',type=float,help='fresh run only; Gao pressure tail, default 0 ms')
    p.add_argument('--duration-ms',type=float,help='optional short interval, not complete-cycle validation')
    p.add_argument('--checkpoint-every',type=int,default=100)
    p.add_argument('--ep-discretization',choices=('nodal_p1','consistent_p1'))
    p.add_argument('--ep-backend',choices=('amgx_pcg_amg','torch_pcg_jacobi'))
    p.add_argument('--ep-config',type=Path,help='MonodomainNumerics JSON; fresh runs only')
    p.add_argument('--ep-refinement-factor',type=int,choices=(1,3),default=None)
    p.add_argument('--ep-refinement-cache',type=Path)
    p.add_argument('--ode-step-ms',type=float)
    p.add_argument('--ode-recovery-config',type=Path,
                   help='Explicit failed-node SDIRK retry policy; also allowed on restart, recorded as a migration')
    p.add_argument('--fsi-config',type=Path,help='Explicit original FSI settings; fresh run only')
    p.add_argument('--spatial-config',type=Path,help='Explicit MAC/IB selection JSON; fresh runs only')
    p.add_argument('--circulation-config',type=Path,help='RCR/initial pressure JSON; fresh runs only, no solver changes')
    p.add_argument('--port-config',type=Path,help='Idealized port geometry JSON; fresh runs only')
    p.add_argument('--diagnostics-config',type=Path,help='Sparse accepted-state/history snapshots for offline analysis')
    p.add_argument('--reuse-step-tangent',action=argparse.BooleanOptionalAction,default=None,
                   help='Reuse the mechanical tangent inside one step, with fresh-Newton fallback.')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    p.error('This immersed-cap LV cycle driver is retired. Use fresh initialization with the single physical basal-port implementation once its LV boundary validation is complete.')
    if a.ep_config and (a.ep_discretization or a.ep_backend):
        p.error('select --ep-config or individual EP overrides')
    if not torch.cuda.is_available() or a.checkpoint_every<1: p.error('CUDA and positive checkpoint interval required')
    if a.resume:
        if a.ep_refinement_factor is not None or a.ep_refinement_cache or a.ode_step_ms is not None or a.fsi_config:
            p.error('resume restores EP mesh and ODE settings')
        if a.port_config: p.error('resume restores port geometry; use a fresh preload for comparison')
        if a.circulation_config: p.error('resume restores circulation state and parameters; use a fresh preload for comparison')
        if a.spatial_config: p.error('resume restores MAC/IB discretization; do not reinterpret history')
        if a.ep_discretization or a.ep_backend or a.ep_config:
            p.error('resume restores EP discretization/backend; overrides are for fresh runs only')
        if (a.cycles is not None or a.reference_checkpoint or a.period_ms is not None
                or a.filling_hold_ms is not None): p.error('resume restores cycle count and timing')
        d,state,progress=restart.load(a.resume,cardiac=True)
    else:
        if not a.reference_checkpoint: p.error('reference checkpoint is required with preload')
        cycles=a.cycles if a.cycles is not None else 2
        period_s=(a.period_ms if a.period_ms is not None else 800.)/1000
        filling_hold_s=(a.filling_hold_ms if a.filling_hold_ms is not None else 0.)/1000
        if cycles<1: p.error('positive cycle count required')
        old,source,_=checkpoint.load(a.reference_checkpoint)
        from afsi_torch.paper_lv import PaperLVConfig
        cfg=replace(old.config,prescribed_endocardial_pressure_mmhg=0.,
            fsi=load_config(a.fsi_config,PaperLVConfig) if a.fsi_config else old.config.fsi,
            ep_refinement_factor=a.ep_refinement_factor or 1,
            ep_refinement_cache=str(a.ep_refinement_cache) if a.ep_refinement_cache else None,
            dt_ode_ms=a.ode_step_ms if a.ode_step_ms is not None else old.config.dt_ode_ms,
            ep_numerics=load_config(a.ep_config,MonodomainNumerics) if a.ep_config else
                MonodomainNumerics(a.ep_discretization or 'nodal_p1',
                                   a.ep_backend or 'amgx_pcg_amg'),
            protocol=replace(old.protocol,pulse_starts_s=tuple(.5+period_s*i for i in range(cycles))))
        spatial={}
        if a.spatial_config:
            spatial=json.loads(a.spatial_config.read_text(encoding='utf-8'))
            keys={'spatial_coupling','pressure_backend','ib_response_backend'}
            if not keys<=set(spatial) or set(spatial)-keys-{'fluid_solver','cap_integration_rule'}:
                p.error('spatial JSON requires spatial_coupling, pressure_backend, ib_response_backend')
            ib=spatial.pop('ib_response_backend')
            cfg=replace(cfg,fsi=replace(cfg.fsi,ib_response_backend=ib,
                ib_csr_contraction_backend='sites' if ib=='quadrature' else cfg.fsi.ib_csr_contraction_backend))
        circulation=load_config(a.circulation_config,CardiacCirculationConfig) if a.circulation_config else CardiacCirculationConfig()
        d=CardiacStepper(old.mesh,cfg,cycles=cycles,period_s=period_s,filling_hold_s=filling_hold_s,
            fill_target_mmhg=circulation.fill_target_mmhg,
            filling_pressure_location=circulation.filling_pressure_location,
            filling_reservoir_mmhg=circulation.filling_reservoir_mmhg,
            filling_cap_reference=circulation.filling_cap_reference,
            filling_model=circulation.filling_model,gao_filling=circulation.gao_filling,**spatial)
        d.valve_resistance_mmhg_s_per_ml=circulation.valve_resistance_mmhg_s_per_ml
        if a.port_config:
            d.port_geometry=load_config(a.port_config,PortGeometry)
        del old,source
        state=d.initialize_preload(a.preload,windkessel=circulation.windkessel,
                                   arterial_pc_mmhg=circulation.initial_pc_mmhg)
        state=circulation.apply_initial_state(state)
        d.validate_control(state)
        progress=dict(preload_source=str(a.preload),ep_steps=0,ode_substeps=0,
                      phase_seen={},max_mass_error_ml_s=0.,max_cumulative_mass_error_ml=0.)
    if a.ode_recovery_config:
        from afsi_torch.ep.tissue import ODERecoveryConfig
        policy=load_config(a.ode_recovery_config,ODERecoveryConfig)
        previous_policy=asdict(d.config.ode_recovery)
        d.config=replace(d.config,ode_recovery=policy)
        d.ode.recovery=policy
        if a.resume and previous_policy!=asdict(policy):
            progress.setdefault('numerical_migrations',[]).append(dict(
                time_ms=state.fsi.time*1000,source_checkpoint=str(a.resume.resolve()),
                setting='ode_recovery',previous=previous_policy,current=asdict(policy),
                history_recomputed=False))
    d.ep.prepare(d.protocol.dt_ep_s)
    if a.reuse_step_tangent is not None: d.reuse_step_tangent=a.reuse_step_tangent
    if state.tick>=d.end_tick: p.error('checkpoint has already reached the requested cycle end')
    target=d.end_tick
    if a.duration_ms is not None:
        if a.duration_ms<=0: p.error('positive duration required')
        target=min(target,state.tick+ticks(a.duration_ms/1000,d.protocol.dt_ep_s))
    a.output.mkdir(parents=True,exist_ok=False)
    publish_json(a.output/'configuration.json',dict(config=asdict(d.config),
        ep_mesh=getattr(d.ep,'metadata',dict(ep_nodes=len(d.ep.X),ep_cells=len(d.mesh.cells))),
        structural_nodes=len(d.mesh.X),structural_cells=len(d.mesh.cells),
        fluid_grid_shape=list(d.config.fsi.fluid.shape),cycles=d.cycles,
        cardiac_execution=d.checkpoint_metadata(state)["cardiac_execution"],
        physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')))
    diagnostics=None
    if a.diagnostics_config:
        from afsi_torch.ep.cycle_diagnostics import CycleDiagnostics
        diagnostics=CycleDiagnostics(json.loads(a.diagnostics_config.read_text(encoding='utf-8')),
                                     a.output,d,state,target)
    publish_json(a.output/'circulation.json',circulation_metadata(d,state))
    publish_json(a.output/'port_geometry.json',asdict(d.port_geometry or PortGeometry()))
    d.profile=True; start=perf_counter(); step_seconds=0.; count=0; failure=None
    # Persist the initial accepted state before the first expensive step.
    restart.save(a.output/'checkpoint.npz',d,state,progress)
    columns=('time_ms','beat','phase','volume_ml','pressure_core_mmhg','pc_mmhg',
             'wall_pressure_mmhg','cap_reference_mmhg','boundary_pressure_mmhg',
             'supply_pressure_mmhg','inlet_open','inlet_reservoir_power_erg_s',
             'wall_load_power_erg_s','q_in_ml_s','q_out_ml_s','tension_max_cgs','mass_error_ml_s','step_seconds')
    with (a.output/'steps.jsonl').open('w',buffering=1) as stream,(a.output/'pv.csv').open('w',newline='',buffering=1) as pv:
        writer=csv.DictWriter(pv,fieldnames=columns); writer.writeheader()
        try:
            while state.tick<target:
                torch.cuda.synchronize(); begin=perf_counter()
                previous=state
                step_target=diagnostics.step_target(state.tick,target) if diagnostics else target
                state,info=d.step(state,target_tick=step_target)
                torch.cuda.synchronize(); seconds=perf_counter()-begin
                step_seconds+=seconds; count+=1
                r=info['balanced']; c=info['cardiac']; ta=d.buffer.tension(state.fsi.x).max().item()
                progress['ep_steps']+=info['ep']['steps']; progress['ode_substeps']+=info['ep']['ode_substeps']
                for event in info['ep']['ode_recovery_events']:
                    summary=progress.setdefault('ode_recovery',dict(events=0,recovered_node_intervals=0,
                        extra_node_substeps=0,max_refinement=0))
                    summary['events']+=1
                    summary['recovered_node_intervals']+=event['recovered_nodes']
                    summary['extra_node_substeps']+=event['extra_node_substeps']
                    summary['max_refinement']=max(summary['max_refinement'],event['max_refinement'])
                seen=progress['phase_seen'].setdefault(str(state.beat),[])
                if state.phase not in seen: seen.append(state.phase)
                progress['max_mass_error_ml_s']=max(progress['max_mass_error_ml_s'],abs(r['mass_error_ml_s']))
                progress['max_cumulative_mass_error_ml']=max(progress['max_cumulative_mass_error_ml'],abs(c['cumulative_mass_error_ml']))
                row=dict(time_ms=state.fsi.time*1000,beat=state.beat,phase=state.phase,volume_ml=r['volume_ml'],
                    pressure_core_mmhg=r['pressure_core']['mean_mmhg'],pc_mmhg=state.arterial_pc_mmhg,
                    wall_pressure_mmhg=r['wall_pressure_mmhg'],cap_reference_mmhg=r['cap_reference_mmhg'],
                    boundary_pressure_mmhg=r['boundary_pressure_mmhg'],
                    supply_pressure_mmhg=r['supply_pressure_mmhg'],inlet_open=r['inlet_open'],
                    inlet_reservoir_power_erg_s=r['inlet_reservoir_power_erg_s'],
                    wall_load_power_erg_s=r['wall_load_power_erg_s'],
                    q_in_ml_s=r['inlet_flow_ml_s'],q_out_ml_s=r['measured_relative_q_ml_s'],
                    tension_max_cgs=ta,mass_error_ml_s=r['mass_error_ml_s'],step_seconds=seconds)
                writer.writerow(row)
                stream.write(json.dumps(dict(**row,cardiac=c,nonlinear=info['nonlinear'],
                    balanced=r,ep=info['ep'],timings=info['timings'],retries=info['rejected_attempts']),allow_nan=False)+'\n')
                if diagnostics:
                    diagnostics.record(d,previous,state,progress,info)
                if count%a.checkpoint_every==0 or c['events']:
                    restart.save(a.output/'checkpoint.npz',d,state,progress)
                if count%10==0 or c['events'] or state.tick==target:
                    publish_json(a.output/'progress.json',dict(status='running',**row,progress=progress,
                        reached_tick=state.tick,target_tick=target,mean_step_seconds=step_seconds/count))
                    print(f"AFSEI beat={state.beat} {state.phase} t={row['time_ms']:.3f}ms "
                          f"V={r['volume_ml']:.6f} P={row['pressure_core_mmhg']:.5f} "
                          f"Qin={r['inlet_flow_ml_s']:.5g} Qout={r['measured_relative_q_ml_s']:.5g} "
                          f"step={seconds:.3f}s events={c['events']}",flush=True)
        except BaseException as exc:
            failure=repr(exc)
            raise
        finally:
            restart.save(a.output/'checkpoint.npz',d,state,progress)
            all_phases=all(set(PHASES)<=set(progress['phase_seen'].get(str(i),[])) for i in range(1,d.cycles+1))
            completed=state.tick==target and failure is None
            full_pass=completed and state.tick==d.end_tick and all_phases and state.phase=='FILLING'
            report=dict(status='completed' if completed else 'incomplete',failure=failure,
                accepted_steps_this_run=count,reached_time_s=state.fsi.time,end_time_s=d.end_tick*d.protocol.dt_ep_s,
                elapsed_seconds=perf_counter()-start,mean_step_seconds=step_seconds/count if count else None,
                cycles=d.cycles,phase=state.phase,beat=state.beat,progress=progress,
                full_cardiac_phase_controller=True,full_cycle_validation_passed=full_pass,
                short_interval_requested=a.duration_ms is not None,
                pointwise_wall_impermeability_established=False,
                fluid_grid='original Cartesian MAC',checkpoint=str(a.output/'checkpoint.npz'))
            if diagnostics:
                report['diagnostics']=dict(index=str(diagnostics.directory/'index.json'),
                    snapshots=len(diagnostics.index),observer_seconds=diagnostics.elapsed_seconds)
            report['ep_numerics'] = dict(discretization=d.ep.numerics.discretization,
                backend=d.ep.numerics.backend,diffusion_quadrature=d.ep.numerics.diffusion_quadrature,
                time_scheme=d.ep.numerics.time_scheme,dt_ode_ms=d.config.dt_ode_ms,
                ode_recovery=asdict(d.config.ode_recovery),
                mesh=getattr(d.ep,'metadata',dict(ep_nodes=len(d.ep.X))))
            report['circulation'] = circulation_metadata(d,state)
            report['mac_numerics']=dict(time_scheme='BE-BE',spatial_coupling=d.spatial_coupling,
                cap_integration_rule=d.cap_integration_rule,
                pressure_backend=d.pressure_backend,volume_transfer=d.volume_transfer,fluid_solver=d.fluid_solver,
                full_griffith_luo_2017_time_scheme=False)
            publish_json(a.output/'report.json',report)
            publish_json(a.output/'progress.json',report)
    print(json.dumps(report,indent=2),flush=True)
    if a.duration_ms is None and not full_pass: raise SystemExit('time reached but complete cardiac phase validation failed')


if __name__=='__main__': main()
