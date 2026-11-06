"""Near-preload pressure compatibility probe; fixed reservoir, no cycle controller."""
import argparse
from dataclasses import fields,replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper,BalancedState
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.gao_tracts import prepare_cut_flow,cut_rows,LVTractDomain
from afsi_torch.ep.tract_storage import TractStorage
from afsi_torch.ep.mac_transport import closed_volume_gradient
from afsi_torch.mac.sealed_pressure import divergence,gradient
from afsi_torch.mac.open_boundary import face_weights
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.ep.buffered_load import ActiveLoadState


def native(v):
    if isinstance(v,torch.Tensor):return v.item() if v.numel()==1 else v.tolist()
    if isinstance(v,dict):return {k:native(x) for k,x in v.items() if not k.startswith('_')}
    if isinstance(v,(tuple,list)):return [native(x) for x in v]
    return v


def fixed_reservoir_driver(source,state,pressure):
    """Copy physics/execution settings into the interval driver explicitly."""
    d=BalancedAFSEIStepper(source.mesh,replace(source.config,max_retries=0),
        level=source.pressure_mode_level,spatial_coupling=source.spatial_coupling,
        pressure_backend=source.pressure_backend,fluid_solver=source.fluid_solver,
        cap_integration_rule=source.cap_integration_rule)
    for name in ('compact_cap_operators','mass_graph_check_every','balanced_tangent_chunk_size',
                 'helmholtz_initial_sweeps','reuse_step_tangent','pressure_guard_cells',
                 'one_way','cap_patch_width_cells','valve_resistance_mmhg_s_per_ml',
                 'port_geometry','response_optimizations'):
        setattr(d,name,getattr(source,name))
    # Both signed directions are retained in this fixed-reservoir diagnostic:
    # reverse flow is recorded rather than suppressed by a valve switch.
    d.inlet_one_way=False
    s=BalancedState(**{f.name:getattr(state,f.name) for f in fields(BalancedState)})
    s=replace(s,mode='filling',cap_mode='filling',filling_pressure_mmhg=pressure,
              wall_pressure_mmhg=0.,cap_reference_mmhg=0.,outlet_open=False)
    d.buffer.stage(s.force_load)
    return d,s


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration',type=Path)
    cfg=json.loads(parser.parse_args().configuration.read_text())
    out=Path(cfg['output']);out.mkdir(exist_ok=False,parents=True)
    (out/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
    torch.set_num_threads(4)
    source,state,_=restart.load(cfg['resume'],cardiac=True)
    pressure=float(cfg.get('pressure_mmhg',8.))
    d,s=fixed_reservoir_driver(source,state,pressure)
    del source,state
    d.gao_cut_boundary=cfg['connection']=='physical-cut'
    if cfg['connection'] not in ('physical-cut','legacy-sheet'):
        raise ValueError('invalid connection')
    d.gao_tract_anchor=s.fsi.x.clone()
    e=d._fsi_stepper(d.maximum_ticks)
    load=MACEndocardialPressure(e.model)
    x0=s.fsi.x.clone();rim=torch.unique(load.rim);dt=e.flow.dt
    report=dict(scope='near-preload fixed-8mmHg initialization compatibility; not a full cycle',
        connection=cfg['connection'],source=cfg['resume'],
        source_sha256=hashlib.sha256(Path(cfg['resume']).read_bytes()).hexdigest(),
        source_time_ms=s.fsi.time*1000,initial_volume_ml=float(load.volume(x0)),
        pressure_mmhg=pressure,dt_ms=dt*1000,source_support_beta=d.config.fsi.beta,
        source_support_changed=False,ep_gpb_land_advanced=not cfg.get('passive_hold',False),phase_controller_advanced=False,
        signed_reservoir_flow=True,accepted_steps=0,records=[])
    started=perf_counter()
    try:
        if d.gao_cut_boundary:
            prepare_cut_flow(e,s.fsi,s)
            domain=e.flow.domain
        else:
            domain=LVTractDomain(e.grid,outlet_rows(e,x0,label=2).prepared.cap,x0,mode='filling')
            domain.storage=TractStorage(outlet_rows(e,x0,label=2).prepared.cap,domain)
        storage=domain.storage
        report['geometry']=domain.metadata
        if cfg.get('correct_passive_preload') or cfg.get('passive_hold'):
            z=torch.zeros_like(s.force_load.xs)
            passive=ActiveLoadState(s.fsi.x,z,z,z,z,dt,False)
            d.buffer.stage(passive)
            s=replace(s,ep=replace(s.ep,force_load=passive))
        if cfg.get('correct_passive_preload'):
            if not d.gao_cut_boundary or not cfg.get('passive_hold'):
                raise ValueError('tract correction is only supported for a passive diagnostic')
            from afsi_torch.ep.tract_preload import correct_equilibrium
            y,correction=correct_equilibrium(d,x0,domain,pressure,progress=lambda r:print(
                json.dumps(dict(static_correction=r)),flush=True))
            report['static_correction']=correction
            report['static_correction'].update(
                volume_before_ml=float(load.volume(x0)),volume_after_ml=float(load.volume(y)),
                rim_mean_axial_change_mm=float((y[rim]-x0[rim])[:,domain.axis].mean()*10))
            x0=y.clone()
            passive=replace(s.force_load,previous_x=x0)
            d.buffer.stage(passive)
            s=replace(s,cap_geometry=x0,ep=replace(s.ep,force_load=passive,
                fsi=replace(s.fsi,x=x0,previous_x=x0,step=0,
                    velocity=e.grid.zeros(device=x0.device,dtype=x0.dtype))))
            report['initial_volume_ml']=float(load.volume(x0))
        rows=cut_rows(e,s.cap_geometry) if d.gao_cut_boundary else outlet_rows(e,s.cap_geometry,label=1)
        bp=torch.full_like(rows.area,C*pressure)
        if d.gao_cut_boundary:bp[-2:]=0.
        u=s.fsi.velocity
        if d.gao_cut_boundary:
            masked=domain.mask(u)
            correction,_=e.flow.pressure_solver.solve(-e.flow.rho/dt*divergence(masked,e.grid.spacing))
            g=gradient(correction,e.grid.spacing,domain.masks)
            u=tuple(v-dt/e.flow.rho*w for v,w in zip(masked,g))
        kinetic=lambda values:.5*e.flow.rho*sum((v*v*face_weights(e.grid,c,x0)).sum() for c,v in enumerate(values))
        report['migration']=dict(kinetic_before_erg=float(kinetic(s.fsi.velocity)),
                                 kinetic_after_erg=float(kinetic(u)))
        s=replace(s,cap_pressure=bp,ep=replace(s.ep,fsi=replace(s.fsi,velocity=u)))
        s=replace(s,ep=replace(s.ep,fsi=replace(s.fsi,force=d.force_for_state(s))))
        d.validate_control(s)

        stencil=e.transfer.prepare(x0)
        unit=getattr(stencil,'cap_unit_pressure',None)
        if unit is None:unit=torch.full_like(stencil.rows.area,C)
        # Known mathematical hydrostatic mode, independent of material force.
        density,_=e.transfer.relative_density(stencil.mode*pressure,unit*pressure,stencil)
        zero=e.grid.zeros(device=x0.device,dtype=x0.dtype)
        hu,hp,hi=e.flow.advance(zero,density)
        hv,_=e.transfer.interpolate(hu,stencil,check_finite=False)
        report['hydrostatic_probe']=dict(
            pressure_mmhg=pressure,max_fluid_speed_cm_s=max(float(v.abs().max()) for v in hu),
            max_nodal_speed_cm_s=float(hv.norm(dim=-1).max()),
            rms_pressure_error_mmhg=float((hp-pressure*stencil.potential).square().mean().sqrt()/C),
            lv_volume_rate_ml_s=float((closed_volume_gradient(x0,load.faces,load.rim)*hv).sum()),
            tract_volume_rate_ml_s=float((storage.discrete(x0,x0)*hv).sum()),
            cut_outward_ml_s=(native(e.flow.flux(hu)) if d.gao_cut_boundary else None),
            max_divergence_s_inv=float(divergence(hu,e.grid.spacing).abs().max()),flow=hi)
        solid=e.model.force(x0,s.fsi.time)
        glv=closed_volume_gradient(x0,load.faces,load.rim)
        gt=storage.discrete(x0,x0)
        report['static_load_audit']=dict(material_force_norm_dyn=solid.norm(),
            closed_lv_pressure_residual_dyn=(solid+C*pressure*glv).norm(),
            lv_plus_tract_pressure_residual_dyn=(solid+C*pressure*(glv+gt)).norm(),
            extra_tract_pressure_force_norm_dyn=(C*pressure*gt).norm())

        for i in range(cfg.get('steps',5)):
            old=s;step_started=perf_counter()
            if cfg.get('passive_hold'):
                d._boundary_context=s
                try:
                    fsi,info=e.physical_step(s.fsi,s.fsi.time+dt,True)
                finally:
                    d._boundary_context=None
                pressure_state,geometry=info.pop('_cap_state')
                s=replace(s,cap_pressure=pressure_state,cap_geometry=geometry,
                    arterial_pc_mmhg=info['balanced']['arterial_pc_mmhg'],
                    ep=replace(s.ep,fsi=fsi))
            else:
                s,info=d.step(s,target_tick=s.tick+d.maximum_ticks,diagnostics=True)
            v=(s.fsi.x-old.fsi.x)/dt
            axial=(s.fsi.x[rim]-x0[rim])[:,domain.axis]
            record=dict(time_ms=s.fsi.time*1000,step_seconds=perf_counter()-step_started,
                balanced=info['balanced'],nonlinear=info['nonlinear'],
                tension_max_cgs=float(d.buffer.tension(s.fsi.x).max()),
                rim_mean_axial_change_cm=axial.mean(),rim_max_abs_axial_change_cm=axial.abs().max(),
                rim_mean_axial_velocity_cm_s=v[rim,domain.axis].mean(),
                tract_volume_ml=storage.volume(s.fsi.x),
                tract_volume_rate_ml_s=(storage.volume(s.fsi.x)-storage.volume(old.fsi.x))/dt)
            report['records'].append(native(record));report['accepted_steps']+=1
            (out/'progress.json').write_text(json.dumps(native(report),indent=2,allow_nan=False)+'\n')
            print(json.dumps(native(record),allow_nan=False),flush=True)
        report['status']='passed'
        np.savez_compressed(out/'accepted_diagnostic_fields.npz',
            x=s.fsi.x.cpu().numpy(),initial_x=x0.cpu().numpy(),
            previous_x=s.fsi.previous_x.cpu().numpy(),pressure=s.fsi.pressure.cpu().numpy(),
            cap_pressure=s.cap_pressure.cpu().numpy(),rim=rim.cpu().numpy(),
            **{f'velocity_{c}':v.cpu().numpy() for c,v in enumerate(s.fsi.velocity)})
        report['diagnostic_fields_are_restart']=False
    except Exception as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if hasattr(exc,'result'):
            report['nonlinear_failure']=dict(residual_norm=exc.result.residual_norm,
                tolerance=exc.result.tolerance,history=exc.result.history)
        problem=getattr(e,'_last_cut_problem',None)
        if problem is not None and problem.last_data is not None:
            y,_,v,force,uu,pp,*_=problem.last_data
            np.savez_compressed(out/'discarded_trial_fields.npz',x=y.cpu().numpy(),
                initial_x=x0.cpu().numpy(),pressure=pp.cpu().numpy(),rim=rim.cpu().numpy(),
                **{f'velocity_{c}':a.cpu().numpy() for c,a in enumerate(uu)})
            report['failed_trial']=dict(volume_ml=float(v),
                tract_volume_change_ml=domain.storage.volume(y)-domain.storage.volume(problem.state.x),
                rim_mean_axial_change_mm=float((y[rim]-problem.state.x[rim])[:,domain.axis].mean()*10))
        print(report['traceback'],flush=True)
    finally:
        report['elapsed_seconds']=perf_counter()-started
        (out/'report.json').write_text(json.dumps(native(report),indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'],steps=report['accepted_steps'],output=str(out))),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
