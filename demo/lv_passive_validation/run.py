"""Same-state passive LV diagnostics; no EP advancement or production restart."""
import argparse
from dataclasses import replace
import hashlib
import json
import runpy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_balanced_dynamic import BalancedEndpoint, BalancedDynamicProblem
from afsi_torch.ep.mac_standard_ports import StandardPortTransfer, StandardPortProblem
from afsi_torch.ep.mac_cap_coupled import closed_rows
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_pressure_audit import cavity_masks, pressure_statistics
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.grid import divergence
from afsi_torch.mac.open_boundary import face_weights
from afsi_torch.nonlinear import newton
from afsi_torch.config import TimeConfig


def read_config(path):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    required={'demo','checkpoint','preload','initial_state','spatial_coupling','pressure_backend',
              'ib_response_backend','boundary_mode','pressure_mmhg','dt_s','steps','log_every',
              'budget_limit_ml','equilibrium_pressure_tolerance_mmhg','equilibrium_flow_tolerance_ml_s'}
    optional={'pressure_rtol','newton_linear_fraction','fluid_solver','cap_integration_rule',
              'wall_pressure_mmhg','cap_reference_mmhg'}
    if not required<=set(c) or set(c)-required-optional or c['demo']!='lv-passive-validation':
        raise ValueError('unexpected or missing validation configuration fields')
    if c['initial_state'] not in ('preload','final') or c['spatial_coupling'] not in ('legacy-balanced','griffith-luo-unified'):
        raise ValueError('invalid initial state or spatial coupling')
    if c['pressure_backend'] not in ('multigrid','spectral') or c['ib_response_backend'] not in ('quadrature','csr'):
        raise ValueError('invalid pressure/IB backend')
    if c['boundary_mode'] not in ('closed','filling'):
        raise ValueError('passive validation supports closed or filling boundaries')
    for k in ('steps','log_every'):
        if type(c[k]) is not int or c[k]<1: raise ValueError('positive integer '+k+' required')
    for k in ('dt_s','budget_limit_ml','equilibrium_pressure_tolerance_mmhg','equilibrium_flow_tolerance_ml_s'):
        if not np.isfinite(c[k]) or c[k]<=0: raise ValueError('positive finite '+k+' required')
    if not np.isfinite(c['pressure_mmhg']): raise ValueError('finite reservoir pressure required')
    if not np.isfinite(c.get('wall_pressure_mmhg',0.)): raise ValueError('finite wall pressure required')
    if not np.isfinite(c.get('cap_reference_mmhg',0.)): raise ValueError('finite cap reference required')
    for k in {'pressure_rtol','newton_linear_fraction'} & set(c):
        if not np.isfinite(c[k]) or not 0<c[k]<1: raise ValueError('invalid '+k)
    if c.get('fluid_solver','be-projection') not in ('be-projection','be-stokes-fgmres'):
        raise ValueError('invalid fluid solver')
    if c.get('fluid_solver')=='be-stokes-fgmres' and (c['spatial_coupling']!='griffith-luo-unified' or c['pressure_backend']!='multigrid'):
        raise ValueError('BE Stokes requires standard IB / multigrid')
    if c.get('cap_integration_rule','centroid') not in ('centroid','gauss3','gauss7','gauss12'):
        raise ValueError('invalid cap integration rule')
    return c


def publish(path,data):
    tmp=path.with_suffix('.working')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


@torch.no_grad()
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--steps',type=int)
    p.add_argument('--check-config',action='store_true')
    p.add_argument('--diagnostic-resume',type=Path,help='Continue only a passive diagnostic snapshot, retaining its history')
    p.add_argument('--audit-budget',action='store_true',help='Decompose only newly accepted steps; no change to equations')
    p.add_argument('--audit-preload',action='store_true',help='Frozen preload force audit only; no time advancement')
    p.add_argument('--audit-surface',type=Path,help='Read saved budget pair and velocity; no time advancement')
    p.add_argument('--surface-prior',type=Path,help='Reuse completed surface audit; refine only cap quadrature')
    p.add_argument('--audit-cap-rules',type=Path,help='Compare cap quadrature on saved fields without advancing time')
    p.add_argument('--cap-reference',type=Path,help='Converged saved-field surface audit for cap-rule comparison')
    p.add_argument('--cap-rules-prior',type=Path,help='Reuse existing cap-rule measurements and evaluate missing rules only')
    p.add_argument('--cap-rule-trial',action='store_true',help='Explicit diagnostic branch changing only cap quadrature at a saved state')
    a=p.parse_args(argv); c=read_config(a.config)
    if a.steps is not None:
        if a.steps<1: p.error('positive steps required')
        c['steps']=a.steps
    if a.check_config: print(json.dumps(c,indent=2)); return
    if a.surface_prior and not a.audit_surface:
        p.error('--surface-prior requires --audit-surface')
    if a.cap_rule_trial and (not a.diagnostic_resume or c['spatial_coupling']!='griffith-luo-unified'):
        p.error('cap-rule trial requires a standard IB diagnostic snapshot')
    if bool(a.audit_cap_rules)!=bool(a.cap_reference):
        p.error('--audit-cap-rules and --cap-reference must be supplied together')
    if a.cap_rules_prior and not a.audit_cap_rules:
        p.error('--cap-rules-prior requires --audit-cap-rules')
    if a.audit_preload and (c['initial_state']!='preload' or c['spatial_coupling']!='griffith-luo-unified' or a.diagnostic_resume):
        p.error('frozen preload audit requires a fresh standard IB preload configuration')
    if a.audit_surface:
        if a.audit_preload or a.audit_budget or a.diagnostic_resume:
            p.error('surface audit is a separate read-only operation')
        previous_config=json.loads((a.audit_surface/'config.resolved.json').read_text())
        if any(c.get(k)!=previous_config.get(k) for k in set(c)|set(previous_config)
               if k not in {'steps','log_every','pressure_rtol','newton_linear_fraction'}):
            p.error('surface audit configuration must match the saved experiment')
    if not torch.cuda.is_available(): p.error('GPU validation requires CUDA')
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    publish(a.output/'config.resolved.json',c)
    start=perf_counter()
    d,state,_=restart.load(c['checkpoint'],cardiac=True)
    # Reuse accepted mechanical fields but explicitly disable active force.
    # No ODE/PDE call, stimulus, artificial relaxation or hidden reset is made.
    d.buffer.stage(replace(state.force_load,use_land=False,anchor=None,ramp_factor=0.,dt_s=c['dt_s']))
    d.pressure_backend=c['pressure_backend']
    d.spatial_coupling=c['spatial_coupling']
    d.fluid_solver=c.get('fluid_solver','be-projection')
    d.cap_integration_rule=c.get('cap_integration_rule','centroid')
    cfg=replace(d.config.fsi,time=TimeConfig(c['dt_s'],c['dt_s']),
                ib_response_backend=c['ib_response_backend'],
                ib_csr_contraction_backend='sites' if c['ib_response_backend']=='quadrature' else d.config.fsi.ib_csr_contraction_backend)
    if 'pressure_rtol' in c:
        cfg=replace(cfg,pressure_solver=replace(cfg.pressure_solver,rtol=c['pressure_rtol']))
    if 'newton_linear_fraction' in c:
        cfg=replace(cfg,nonlinear=replace(cfg.nonlinear,linear_tolerance_fraction=c['newton_linear_fraction']))
    # Release endpoint caches owned by the checkpoint loader before the new
    # explicitly configured experiment (especially its large CSR transfer).
    d._steppers.clear()
    e=BalancedEndpoint(d,cfg)
    aligned=c['spatial_coupling']=='griffith-luo-unified'
    if aligned: e.transfer=StandardPortTransfer(e,e.transfer.base)
    cls=StandardPortProblem if aligned else BalancedDynamicProblem
    old=state.fsi
    pre=np.load(c['preload'])
    pre_x=torch.as_tensor(pre['x'],device=old.x.device)
    load=MACEndocardialPressure(d.model)
    target_volume=load.volume(pre_x).item()
    if c['initial_state']=='preload':
        old=replace(old,x=pre_x,previous_x=None,previous_dt=c['dt_s'],
                    velocity=e.grid.zeros(device=old.x.device,dtype=old.x.dtype),
                    pressure=torch.as_tensor(pre['pressure'],device=old.x.device))
    control=replace(state,mode=c['boundary_mode'],cap_mode=None,filling_pressure_mmhg=c['pressure_mmhg'],
                    wall_pressure_mmhg=c.get('wall_pressure_mmhg',0.),
                    cap_reference_mmhg=c.get('cap_reference_mmhg',0.))
    initial_volume=load.volume(old.x).item()
    volume0=initial_volume; cumulative_out=0.; records=[]; failure=None
    initial_x=old.x.clone(); cap_pressure=None; core=None; core_anchor=None
    if a.diagnostic_resume:
        prior=a.diagnostic_resume.resolve()
        previous_config=json.loads((prior/'config.resolved.json').read_text())
        # Only solver accuracy and requested duration may change. All physical
        # inputs, geometry, time step and spatial operators must be identical.
        for key in set(c)|set(previous_config):
            if key=='cap_integration_rule':
                if c.get(key,'centroid')!=previous_config.get(key,'centroid') and not a.cap_rule_trial:
                    raise ValueError('changing cap integration requires explicit --cap-rule-trial')
                continue
            if key not in {'steps','log_every','pressure_rtol','newton_linear_fraction'} and c.get(key)!=previous_config.get(key):
                raise ValueError('diagnostic continuation changed '+key)
        records=[json.loads(line) for line in (prior/'history.jsonl').read_text().splitlines()]
        previous_report=json.loads((prior/'report.json').read_text())
        if not records: raise ValueError('no accepted diagnostic state to continue')
        snap=np.load(prior/'diagnostic_state.npz')
        like=lambda key:torch.as_tensor(snap[key],device=old.x.device)
        n=records[-1]['step']; elapsed=n*c['dt_s']
        if c['steps']<=n:
            p.error('continuation must request a step beyond the saved history')
        old=replace(old,x=like('x'),velocity=tuple(like(f'velocity_{j}') for j in range(3)),
                    pressure=like('pressure'),time=old.time+elapsed,step=old.step+n,
                    previous_x=None,previous_dt=c['dt_s'])
        initial_volume=previous_report['initial_volume_ml']; volume0=records[-1]['volume_ml']
        cumulative_out=records[-1]['cumulative_budget_defect_ml']-volume0+initial_volume
        publish(a.output/'continuation.json',dict(source=str(prior),accepted_steps_retained=n,
            previous_failure=previous_report['failure'],previous_x_only_initial_guess_reset=True,
            cap_rule_changed_at_step=n+1 if a.cap_rule_trial else None,
            previous_cap_rule=previous_config.get('cap_integration_rule','centroid'),
            new_cap_rule=d.cap_integration_rule))
    methods=dict(time='BE-BE',fluid='MAC; implicit viscosity; semi-Lagrangian convection',fluid_solver=d.fluid_solver,
        pressure_backend=e.flow.pressure_solver.backend,spatial_coupling=c['spatial_coupling'],
        transfer_class=type(e.transfer.base).__name__,pressure_mode_correction=not aligned,
        cap_integration_rule=d.cap_integration_rule,
        complete_2017_time_and_stokes_scheme=False,ep_advanced=False,active_tension_cgs=0.,
        upstream='AFSI_GPU cb2d30b / adaptive_transfer.py from 9c39e32',
        prescribed_wall_pressure_mmhg=c.get('wall_pressure_mmhg',0.),reservoir_pressure_mmhg=c['pressure_mmhg'],
        cap_reference_mmhg=c.get('cap_reference_mmhg',0.),
        warning='Validation-only trajectory. Budget failure is recorded, never hidden or promoted to a production checkpoint.')
    publish(a.output/'methods.json',methods)
    if a.audit_preload or a.audit_budget or a.audit_surface or a.audit_cap_rules:
        audit_helpers=runpy.run_path(str(Path(__file__).with_name('budget_audit.py')))
    if a.audit_cap_rules:
        report=audit_helpers['audit_cap_rules'](e,a.audit_cap_rules,a.cap_reference,a.cap_rules_prior)
        report.update(elapsed_seconds=perf_counter()-start,methods=methods)
        publish(a.output/'cap_rules_audit.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.audit_surface:
        report=audit_helpers['audit_saved_surface'](e,a.audit_surface,a.surface_prior)
        report.update(elapsed_seconds=perf_counter()-start,methods=methods)
        publish(a.output/'surface_audit.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return
    if a.audit_preload:
        report=audit_helpers['audit_static_preload'](e,old.x,old.time,c['pressure_mmhg'])
        report.update(elapsed_seconds=perf_counter()-start,methods=methods)
        publish(a.output/'static_force_audit.json',report)
        print(json.dumps(report,indent=2),flush=True)
        return
    setup=perf_counter()-start
    with (a.output/'history.jsonl').open('w',buffering=1) as stream:
        for row in records: stream.write(json.dumps(row)+'\n')
        try:
            for k in range(len(records)+1,c['steps']+1):
                begin=perf_counter()
                stencil=e.transfer.prepare(old.x)
                problem=cls(e,old,stencil,e.flow.advect(old.velocity),control)
                if cap_pressure is None or len(cap_pressure)!=problem.rows.count:
                    cap_pressure=torch.full_like(problem.rows.area,c['pressure_mmhg']*C)
                y=torch.zeros_like(old.x) if old.previous_x is None else (old.x-old.previous_x)*(c['dt_s']/(old.previous_dt or c['dt_s']))
                z=torch.cat((y.flatten(),cap_pressure/problem.cap_scale))
                r0=problem.residual(z); opt=e.config.nonlinear
                tol=max(opt.atol,opt.rtol*r0[:old.x.numel()].norm().item())
                sol=newton(problem.residual,z,validate=problem.validate,
                    linearization_factory=lambda v:problem.linearization(v,reuse_tangent=False),
                    preconditioner_factory=problem.preconditioner,
                    options=replace(opt,rtol=0.,atol=tol),krylov_workspace=e.krylov_workspace)
                residual=problem.accepted(sol.x)
                x,_,vol,force,u,pressure,flow,nodal,density,_,_=problem.last_data
                e.check_support(x)
                if residual.norm().item()>tol: raise RuntimeError('true nonlinear residual failed')
                courant=c['dt_s']*sum(v.abs().max()/h for v,h in zip(u,e.grid.spacing)).item()
                if courant>e.config.max_courant: raise RuntimeError('configured CFL limit exceeded')
                relative=problem.rows.flow(u)-problem.rows.material_flow((x-old.x)/c['dt_s'])
                q=relative[-1].item() if c['boundary_mode']=='filling' else 0.
                nclosed=problem.rows.count-(c['boundary_mode']=='filling')
                closed_flux=relative[:nclosed].norm().item()
                if closed_flux>1e-5: raise RuntimeError('true closed-cap constraint failed')
                cap_pressure=problem.last_cap_pressure.clone()
                if c['boundary_mode']=='filling' and abs(cap_pressure[-1].item()/C-c['pressure_mmhg'])>1e-7:
                    raise RuntimeError('reservoir pressure law failed')
                if core is None or (x-core_anchor).abs().max().item()>.25*min(e.grid.spacing):
                    _,core=cavity_masks(x,d.model.endo_faces,e.grid,guard_cells=d.pressure_guard_cells)
                    core_anchor=x.clone()
                pc=pressure_statistics(pressure,core)
                cumulative_out+=q*c['dt_s']
                budget=vol.item()-initial_volume+cumulative_out
                fp=sum((a*b*face_weights(e.grid,j,x)).sum() for j,(a,b) in enumerate(zip(u,density)))
                # Density includes -B^T p. Add its power back before checking S/J.
                bp_power=sum((a*b*face_weights(e.grid,j,x)).sum() for j,(a,b) in enumerate(zip(u,problem.rows.lift(cap_pressure))))
                sp=(force*nodal).sum()
                power_error=abs((fp+bp_power-sp).item())/max(abs(fp.item())+abs(bp_power.item())+abs(sp.item()),1.)
                if power_error>1e-7: raise RuntimeError('transfer power-adjoint check failed')
                row=dict(step=k,elapsed_physical_ms=k*c['dt_s']*1000,volume_ml=vol.item(),
                    cap_integration_rule=d.cap_integration_rule,
                    pressure_core_mmhg=pc['mean_mmhg'],boundary_pressure_mmhg=cap_pressure[-1].item()/C,
                    q_in_ml_s=-q,mass_defect_ml_s=(vol.item()-volume0)/c['dt_s']+q,
                    cumulative_budget_defect_ml=budget,closed_flux_norm_ml_s=closed_flux,
                    divergence_l2=(e.grid.volume*divergence(u,e.grid.spacing).square().sum()).sqrt().item(),
                    max_velocity_cm_s=max(v.abs().max().item() for v in u),
                    max_displacement_from_initial_cm=(x-initial_x).abs().max().item(),
                    power_relative_error=power_error,nonlinear_iterations=sol.iterations,
                    nonlinear_residual=residual.norm().item(),nonlinear_tolerance=tol,
                    step_seconds=perf_counter()-begin)
                if a.audit_budget:
                    row['budget_decomposition']=audit_helpers['decompose_budget'](old.x,x,d.model.endo_faces,
                        d.model.rim,c['dt_s'],nodal,u,problem.rows)
                    if k==c['steps']:
                        np.savez_compressed(a.output/'budget_last_pair.npz',
                            previous_x=old.x.cpu().numpy(),x=x.cpu().numpy(),nodal_velocity=nodal.cpu().numpy())
                old=replace(old,step=old.step+1,time=old.time+c['dt_s'],x=x,velocity=u,pressure=pressure,
                            force=force,previous_x=old.x.clone(),previous_dt=c['dt_s'])
                volume0=vol.item(); records.append(row); stream.write(json.dumps(row)+'\n')
                if k%c['log_every']==0 or k==1 or k==c['steps']:
                    publish(a.output/'progress.json',dict(status='running',**row))
                    print(f"{c['spatial_coupling']} step={k} t={row['elapsed_physical_ms']:.3f}ms V={vol.item():.6f} P={pc['mean_mmhg']:.5f} Qin={-q:.5f} budget={budget:.3g} s/step={row['step_seconds']:.3f}",flush=True)
                if abs(budget)>c['budget_limit_ml']:
                    raise RuntimeError('independent cavity budget exceeded configured diagnostic stop limit')
        except Exception as exc:
            failure=repr(exc)
        finally:
            last=records[-1] if records else None
            reached=len(records)==c['steps'] and failure is None
            equilibrium=bool(reached and last and
                abs(last['pressure_core_mmhg']-c['pressure_mmhg'])<c['equilibrium_pressure_tolerance_mmhg'] and
                abs(last['q_in_ml_s'])<c['equilibrium_flow_tolerance_ml_s'] and
                last['max_velocity_cm_s']<(1e-6 if c['boundary_mode']=='closed' else .1) and
                (c['boundary_mode']!='closed' or (abs(last['volume_ml']-initial_volume)<1e-3 and
                    last['max_displacement_from_initial_cm']<1e-9)))
            report=dict(status='completed' if reached else 'failed',failure=failure,methods=methods,
                accepted_diagnostic_steps=len(records),setup_seconds=setup,elapsed_seconds=perf_counter()-start,
                initial_volume_ml=initial_volume,reference_preload_volume_ml=target_volume,last=last,
                equilibrium_reached=equilibrium,full_cycle_validated=False,
                volume_difference_from_preload_ml=None if last is None else last['volume_ml']-target_volume,
                maximum_budget_defect_ml=max((abs(r['cumulative_budget_defect_ml']) for r in records),default=0.),
                input_checkpoint_sha256=hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest())
            publish(a.output/'report.json',report); publish(a.output/'progress.json',report)
            np.savez_compressed(a.output/'diagnostic_state.npz',x=old.x.cpu().numpy(),pressure=old.pressure.cpu().numpy(),
                **{f'velocity_{j}':v.cpu().numpy() for j,v in enumerate(old.velocity)})
    print(json.dumps(report,indent=2),flush=True)
    if failure: raise SystemExit(1)


if __name__=='__main__': main()
