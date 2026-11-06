"""Opt-in physical-cut LV cycles from a complete, compatible loading history."""
import argparse
import gc
from dataclasses import asdict,replace
import json
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.cut_cycle import CutCardiacStepper
from afsi_torch.ep.cut_validation import EventWindow
from afsi_torch.ep.protocol import ticks
from afsi_torch.ep.circulation_config import CardiacCirculationConfig
from afsi_torch.config import load_config


def native(v):
    if isinstance(v,torch.Tensor):return v.item() if v.numel()==1 else v.tolist()
    if isinstance(v,dict):return {k:native(x) for k,x in v.items() if not k.startswith('_')}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v


def publish(path,data):
    temporary=path.with_suffix(path.suffix+'.working')
    temporary.write_text(json.dumps(native(data),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('configuration',type=Path)
    cfg=json.loads(p.parse_args().configuration.read_text())
    p.error('This fixed-tract/cap/pressure-mode LV driver is retired. Do not resume it; the single physical basal-port driver requires fresh boundary-compatible initialization.')
    diagnostic_every=cfg.get('cut_diagnostics_every',0)
    active_every=cfg.get('active_diagnostics_every',0)
    if type(active_every) is not int or active_every<0:raise ValueError('invalid active diagnostic cadence')
    periodic_cycles=cfg.get('record_periodic_cycles',0)
    if type(periodic_cycles) is not int or periodic_cycles<0:raise ValueError('invalid periodic cycle count')
    if type(diagnostic_every) is not int or diagnostic_every<0:
        raise ValueError('cut_diagnostics_every must be a nonnegative integer')
    if 'pressure_compact_geometry' in cfg and type(cfg['pressure_compact_geometry']) is not bool:
        raise ValueError('pressure_compact_geometry must be boolean')
    if 'tract_geometry_gpu' in cfg and type(cfg['tract_geometry_gpu']) is not bool:
        raise ValueError('tract_geometry_gpu must be boolean')
    out=Path(cfg['output']);out.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(4)
    if cfg.get('resume'):
        if cfg.get('preload_history') or cfg.get('circulation_config'):
            raise ValueError('resume restores loading, circulation and geometry; no fresh overrides')
        d,s,_=restart.load(cfg['resume'],cardiac=True)
        if not isinstance(d,CutCardiacStepper):raise ValueError('physical-cut checkpoint required')
        for key,stored in [('cut_geometry',d.cut_geometry),('filling_clock',
                asdict(d.filling_clock) if d.filling_clock else None)]:
            if key in cfg and cfg[key]!=stored:
                raise ValueError(key+' changes the model; initialize a fresh compatible run')
        if 'rcr_initialization' in cfg or 'cycles' in cfg or 'periodic_criteria' in cfg:
            raise ValueError('resume retains its circulation initialization, cycle count and criteria')
        if 'pressure_compact_geometry' in cfg:
            d.cut_pressure_compact_geometry=cfg['pressure_compact_geometry']
        if 'tract_geometry_gpu' in cfg:
            d.cut_tract_geometry_gpu=cfg['tract_geometry_gpu']
        if 'cut_pressure_solver' in cfg:d.cut_pressure_tolerances=cfg['cut_pressure_solver']
    else:
        old,source,_=restart.load(cfg['reference_checkpoint'],cardiac=True)
        execution=old.checkpoint_metadata(source)['cardiac_execution']
        config=old.config
        for key in ('filling_clock','periodic_criteria'):
            if key in cfg:execution[key]=cfg[key]
        if 'cycles' in cfg:
            count=cfg['cycles']
            if type(count) is not int or count<1:raise ValueError('positive cycle count required')
            execution['cycles']=count
            protocol=replace(old.protocol,pulse_starts_s=tuple(old.protocol.pulse_starts_s[0]
                +i*execution['period_s'] for i in range(count)))
            config=replace(old.config,protocol=protocol)
        d=CutCardiacStepper(old.mesh,config,level=old.pressure_mode_level,
            spatial_coupling=old.spatial_coupling,pressure_backend=old.pressure_backend,
            fluid_solver=old.fluid_solver,cap_integration_rule=old.cap_integration_rule,**execution)
        for name in ('compact_cap_operators','mass_graph_check_every','balanced_tangent_chunk_size',
                     'helmholtz_initial_sweeps','reuse_step_tangent','pressure_guard_cells',
                     'one_way','cap_patch_width_cells','valve_resistance_mmhg_s_per_ml',
                     'port_geometry','response_optimizations'):
            setattr(d,name,getattr(old,name))
        circulation=load_config(cfg['circulation_config'],CardiacCirculationConfig)
        d.valve_resistance_mmhg_s_per_ml=circulation.valve_resistance_mmhg_s_per_ml
        d.cut_pressure_compact_geometry=cfg.get('pressure_compact_geometry',False)
        d.cut_tract_geometry_gpu=cfg.get('tract_geometry_gpu',False)
        d.cut_pressure_tolerances=cfg.get('cut_pressure_solver')
        d.cut_geometry=cfg.get('cut_geometry',{})
        if d.valve_resistance_mmhg_s_per_ml!=0:
            raise ValueError('physical-cut RCR adapter currently requires zero additional valve resistance')
        initialization=cfg.get('rcr_initialization',{'mode':'specified'})
        pc=circulation.initial_pc_mmhg
        if initialization['mode']=='frozen_flow':
            import csv
            from afsi_torch.ep.cycle_policy import frozen_flow_initial_pc
            start=initialization['period_start_ms'];end=initialization['period_end_ms']
            records=list(csv.DictReader(Path(initialization['template']).open()))
            rows=[r for r in records if start<float(r['time_ms'])<=end+1e-8]
            if not rows or abs(float(rows[-1]['time_ms'])-end)>1e-7:
                raise ValueError('RCR initialization requires a complete recorded flow period')
            times=[start]+[float(r['time_ms']) for r in rows]
            dt=np.diff(times)/1000
            if dt.min()<=0 or dt.max()>.001:raise ValueError('flow template is incomplete or unordered')
            flows=[float(r['q_out_ml_s']) for r in rows]
            if min(flows)<-1e-5:raise ValueError('flow template contains reverse arterial flow')
            seed=frozen_flow_initial_pc(dt,[max(0.,q) for q in flows],circulation.windkessel)
            if abs(seed['period_s']-execution['period_s'])>1e-8:
                raise ValueError('flow template period differs from nominal pacing period')
            pc=seed['pc_mmhg'];initialization=dict(initialization,estimate=seed)
        elif initialization['mode']!='specified':raise ValueError('unknown RCR initialization mode')
        d.cut_initialization=dict(initialization,actual_initial_pc_mmhg=pc,periodic_state_assumed=False)
        s=d.initialize_loading_history(cfg['preload_history'],arterial_pc_mmhg=pc,windkessel=circulation.windkessel)
        del old,source
        gc.collect()
    from afsi_torch.mac.sealed_pressure import tightened_pressure_options
    tightened_pressure_options(d.config.fsi.pressure_solver,d.cut_pressure_tolerances)
    if periodic_cycles and d.periodic_monitor is None:
        raise ValueError('periodic recording requires coupled-state periodic criteria')
    publish(out/'configuration.json',dict(request=cfg,config=asdict(d.config),
        cut_boundary_execution=dict(pressure_solver_tolerances=d.cut_pressure_tolerances,
            tract_geometry_gpu=d.cut_tract_geometry_gpu,compact_geometry=d.cut_pressure_compact_geometry,
            geometry=d.cut_geometry,initialization=getattr(d,'cut_initialization',None)),
        cardiac_execution=d.checkpoint_metadata(s)['cardiac_execution'],
        preload_history=d.cut_preload_metadata,coarse_ep_nodes=len(d.ep.X),
        structural_nodes=len(d.mesh.X),fluid_shape=d.config.fsi.fluid.shape))
    report=dict(status='running',start_tick=s.tick,start_time_ms=s.fsi.time*1000,
        accepted_steps=0,ep_steps=0,ode_substeps=0,records=[],restart_roundtrip=False,
        full_cycle_completed=False)
    target=d.end_tick
    absolute_limit=None
    if cfg.get('duration_ms') is not None:
        if cfg['duration_ms']<=0:raise ValueError('positive duration required')
        absolute_limit=s.tick+ticks(cfg['duration_ms']/1000,d.protocol.dt_ep_s)
        target=min(target,absolute_limit)
    report['target_tick']=target
    limit_tick=target
    window=(EventWindow.from_config(cfg['stop_after_event'],d.protocol.dt_ep_s)
            if 'stop_after_event' in cfg else None)
    if window is not None:
        report['event_window']=dict(name=window.name,following_ticks=window.following_ticks,
                                    first_tick=None,limit_tick=limit_tick,completed=False)
    started=perf_counter()
    try:
        restart.save(out/'initial_checkpoint.npz',d,s,dict(scope='fresh complete loading history'))
        if cfg.get('verify_restart',True):
            restored,copied,_=restart.load(out/'initial_checkpoint.npz',cardiac=True)
            for a,b in ((copied.fsi.x,s.fsi.x),(copied.strain.zs,s.strain.zs),(copied.strain.zw,s.strain.zw),
                        (restored.gao_tract_anchor,d.gao_tract_anchor),(copied.cap_pressure,s.cap_pressure),
                        (copied.fsi.force,s.fsi.force)):
                if not torch.equal(a,b):raise ValueError('physical-cut initial restart changed accepted data')
            if (copied.initial_tract_volume_ml!=s.initial_tract_volume_ml
                    or restored.cut_preload_metadata!=d.cut_preload_metadata
                    or restored.cut_pressure_compact_geometry!=d.cut_pressure_compact_geometry
                    or restored.cut_tract_geometry_gpu!=d.cut_tract_geometry_gpu
                    or restored.cut_pressure_tolerances!=d.cut_pressure_tolerances):
                raise ValueError('physical-cut initial restart changed control/history provenance')
            report['restart_roundtrip']=True
            # Advance the restored object, rather than only comparing arrays.
            previous_driver=d
            d,s=restored,copied
            del previous_driver,restored,copied
            gc.collect()
        with (out/'steps.jsonl').open('w',buffering=1) as stream:
            while s.tick<target:
                step_started=perf_counter()
                step_target=target
                if cfg.get('validation_step_ms') is not None:
                    step_target=min(target,s.tick+ticks(cfg['validation_step_ms']/1000,d.protocol.dt_ep_s))
                s,info=d.step(s,target_tick=step_target,diagnostics=True)
                limit_tick=min(d.end_tick,absolute_limit) if absolute_limit is not None else d.end_tick
                target=limit_tick
                if window is not None:
                    target=window.observe(s.tick,info['cardiac']['events'],limit_tick)
                    report['target_tick']=target
                    report['event_window'].update(first_tick=window.first_tick,completed=window.completed(s.tick))
                r=info['balanced'];ep=info['ep']
                row=dict(time_ms=s.fsi.time*1000,phase=s.phase,mode=s.mode,
                    step_seconds=perf_counter()-step_started,balanced=r,cardiac=info['cardiac'],
                    ep=ep,tension_max_cgs=float(d.buffer.tension(s.fsi.x).max()),
                    accepted_dt_s=info['accepted_dt_s'],rejected_attempts=info.get('rejected_attempts',[]),
                    nonlinear=info['nonlinear'],flow=info['flow'],
                    valve_trial_discarded=info.get('cut_valve_trial_discarded',False))
                if active_every and (report['accepted_steps']==0 or (report['accepted_steps']+1)%active_every==0):
                    from afsi_torch.ep.active_diagnostics import active_metrics
                    row['active_mechanics']=active_metrics(d.model,d.buffer,s.fsi.x)
                if periodic_cycles and d.periodic_monitor.summary()['completed_periodic_cycles']>=periodic_cycles:
                    target=s.tick
                    report['periodic_recording_completed']=True
                    first=next(sample for sample in d.periodic_monitor.samples if sample['beat']==s.beat-periodic_cycles)
                    report['reporting_window']=dict(start_time_ms=first['time_s']*1000,
                        end_time_ms=s.fsi.time*1000,completed_cycles=periodic_cycles,
                        first_beat=first['beat'],end_boundary_beat=s.beat)
                if diagnostic_every and (report['accepted_steps']==0
                        or (report['accepted_steps']+1)%diagnostic_every==0 or s.tick==target):
                    from afsi_torch.ep.cut_diagnostics import observe_cut_momentum
                    try:
                        row['cut_momentum_budget']=observe_cut_momentum(d,s,info)
                    except Exception as exc:
                        # Observational failure must not discard accepted physics.
                        row['cut_momentum_diagnostic_error']=type(exc).__name__+': '+str(exc)
                stream.write(json.dumps(native(row),allow_nan=False)+'\n')
                report['records'].append(native(row));report['accepted_steps']+=1
                report['ep_steps']+=ep['steps'];report['ode_substeps']+=ep['ode_substeps']
                if (report['accepted_steps']%cfg.get('checkpoint_every',10)==0 or s.tick==target
                        or info['cardiac']['events']):
                    restart.save(out/'checkpoint.npz',d,s,dict(accepted_steps=report['accepted_steps']))
                    publish(out/'progress.json',report)
                    print(json.dumps(dict(time_ms=row['time_ms'],steps=report['accepted_steps'],
                        phase=s.phase,volume_ml=r['volume_ml'],pressure_core_mmhg=r['pressure_core']['mean_mmhg'],
                        pressure_cut_mmhg=r['boundary_pressure_mmhg'],
                        arterial_pc_mmhg=r['arterial_pc_mmhg'],
                        qout_ml_s=r.get('physical_outlet_flow_ml_s',0.),events=info['cardiac']['events'],
                        step_seconds=row['step_seconds'],mass_error_ml_s=r['mass_error_ml_s'],
                        tension_max_cgs=row['tension_max_cgs'])),flush=True)
        report.update(status='passed',end_tick=s.tick,end_time_ms=s.fsi.time*1000,
            full_cycle_completed=s.tick==d.end_tick,
            max_mass_error_ml_s=max(abs(r['balanced']['mass_error_ml_s']) for r in report['records']),
            max_cumulative_mass_error_ml=max(abs(r['cardiac']['cumulative_mass_error_ml']) for r in report['records']))
        report['final_schedule_ms']=[v*1000 for v in d.protocol.pulse_starts_s]
        report['periodicity']=d.periodic_monitor.summary() if d.periodic_monitor else None
        report['periodic_state_verified']=bool(d.periodic_monitor and d.periodic_monitor.summary()['periodic'])
        if periodic_cycles and not report.get('periodic_recording_completed'):
            report.update(status='incomplete',stop_reason='maximum beats/duration before verified periodic recording')
        report['rejected_attempt_count']=s.rejected_attempts
        restored,copied,_=restart.load(out/'checkpoint.npz',cardiac=True)
        if not torch.equal(copied.tissue.voltage_v,s.tissue.voltage_v) or not torch.equal(copied.strain.zs,s.strain.zs):
            raise ValueError('advanced restart changed EP or mechanics history')
        report['advanced_restart_roundtrip']=True
        if window is not None and not window.completed(s.tick):
            report.update(status='incomplete',stop_reason='duration limit before requested event window completed')
        del restored,copied
    except Exception as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        report['failed_step_attempts']=getattr(exc,'attempts',[])
        cause=exc
        while cause is not None:
            result=getattr(cause,'result',None)
            if result is not None:
                report['failed_nonlinear']=dict(residual_norm=result.residual_norm,
                    tolerance=result.tolerance,iterations=result.iterations,history=result.history)
                np.savez_compressed(out/'unaccepted_newton_iterate.npz',unknown=result.x.detach().cpu().numpy(),
                    description='UNACCEPTED diagnostic iterate; not a physical restart',last_accepted_tick=s.tick)
                break
            cause=cause.__cause__
        try:
            restart.save(out/'last_accepted_checkpoint.npz',d,s,dict(scope='last accepted state before failed trial'))
            report['last_accepted_checkpoint_saved']=True
        except Exception as save_error:
            report['last_accepted_checkpoint_save_error']=str(save_error)
        print(report['traceback'],flush=True)
    finally:
        report['elapsed_seconds']=perf_counter()-started
        publish(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
