"""Read-only decomposition of the actual frozen-geometry BE port budget."""
import torch
import json
import numpy as np
from pathlib import Path
import copy
from time import perf_counter
from afsi_torch.ep.mac_transport import ClosedVolumeTransport, closed_volume_covectors
from afsi_torch.ep.mac_balanced_preload import EndocardialAndCapPressure
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_pressure_compatibility import weighted_norm
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.grid import divergence
from afsi_torch.mac.open_boundary import gradient
from afsi_torch.ep.mac_mass_budget import ClosedCavityMassAudit
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.mac.open_boundary import face_weights


@torch.no_grad()
def decompose_budget(previous_x, x, faces, rim, dt, nodal, velocity, rows):
    """B = dV/dt + sum(Bu-Cv); positive port flow is outward.

    v=(x-x_old)/dt, w=Ju, g=old exact closed volume gradient, c=old
    material cap covector, gbar=exact swept gradient. The four terms are
    (gbar-g)v + (g-c)(v-w) + (g_cap-c)w + g_wall*w + sum(Bu).
    This algebra does not impose a budget or change a field.
    """
    transport = ClosedVolumeTransport.prepare(previous_x, x, faces, dt, rim=rim)
    old = closed_volume_covectors(previous_x, faces, rim)
    swept = transport.covectors.sum(0)
    g = old.sum(0)
    c = rows.reaction(torch.ones_like(rows.area))
    v = (x-previous_x)/dt
    fluid_cap = rows.flow(velocity).sum()
    cap_motion = rows.material_flow(v).sum()
    geometric = transport.audit(nodal)
    terms = dict(
        geometry_time_ml_s=((swept-g)*v).sum(),
        motion_residual_ml_s=((g-c)*(v-nodal)).sum(),
        cap_quadrature_ml_s=((old[1]-c)*nodal).sum(),
        spatial_wall_port_ml_s=(old[0]*nodal).sum()+fluid_cap,
    )
    measured = geometric['secant_volume_rate_ml_s']+fluid_cap-cap_motion
    residual = measured-sum(terms.values())
    numbers = torch.stack((*terms.values(), measured, residual, fluid_cap, cap_motion)).tolist()
    report = dict(zip(terms, numbers[:4]))
    report.update(total_budget_ml_s=numbers[4], decomposition_error_ml_s=numbers[5],
        fluid_cap_outward_ml_s=numbers[6], material_cap_outward_ml_s=numbers[7],
        projected_wall_outward_ml_s=(old[0]*nodal).sum().item(),
        geometry=geometric, changes_simulation=False,
        interpretation='spatial_wall_port is an integrated compatibility defect, not a pointwise leak measurement')
    if abs(report['decomposition_error_ml_s'])>1e-7:
        raise RuntimeError('independent budget decomposition identity failed')
    return report


@torch.no_grad()
def audit_static_preload(endpoint, x, time, pressure_mmhg):
    """Frozen configuration; no FSI time step, preload solve, or force correction."""
    e = endpoint
    stencil = e.transfer.prepare(x)
    rows = stencil.rows
    material = e.solid.force(x, time)
    source = EndocardialAndCapPressure(e.model)
    prescribed = source.force(x, pressure_mmhg)
    balance = material+prescribed
    bp = torch.full_like(rows.area, pressure_mmhg*C)
    cap = rows.reaction(bp)
    e.transfer.reset_warm_start()
    density, _, _ = e.transfer.relative_parts(material+cap, bp, stencil, rows=rows)
    # Copy before further transfer calls: implementation may reuse scratch.
    density = tuple(v.clone() for v in density)
    potential, info = e.flow.pressure_solver.solve(-divergence(density, e.grid.spacing))
    g = gradient(potential, e.grid.spacing)
    remainder = tuple(f-v for f,v in zip(density, g))
    norm = weighted_norm(e.grid, density)
    residual = weighted_norm(e.grid, remainder)
    cap_exact = pressure_mmhg*C*source.cap_covector(x)
    # Replacing material by its exactly balanced value is ONLY an audit
    # comparison; neither field is advanced or stored as a production state.
    e.transfer.reset_warm_start()
    ideal, _, _ = e.transfer.relative_parts(-prescribed+cap, bp, stencil, rows=rows)
    force_scale = prescribed.norm().item()
    out = dict(scope='frozen standard IB preload load / MAC gradient compatibility',
        state_advanced=False, production_force_modified=False,
        pressure_mmhg=pressure_mmhg,
        volume_ml=MACEndocardialPressure(e.model).volume(x).item(),
        solid_force_balance_l2_dyn=balance.norm().item(),
        source_pressure_force_l2_dyn=force_scale,
        solid_force_balance_relative=balance.norm().item()/max(force_scale,1e-30),
        cap_reaction_vs_geometric_l2_dyn=(cap-cap_exact).norm().item(),
        force_density_weighted_l2=norm.item(),
        non_gradient_density_weighted_l2=residual.item(),
        non_gradient_density_fraction=(residual/norm.clamp_min(1e-30)).item(),
        density_change_if_solid_exactly_equilibrated_relative=(
            weighted_norm(e.grid, tuple(a-b for a,b in zip(density,ideal)))/norm.clamp_min(1e-30)).item(),
        hodge_pressure_solve=info,
        note='Nonzero solenoidal force cannot be eliminated by changing pressure alone at this fixed geometry. '
             'This is the instantaneous MAC force balance, not the split-BE manufactured balance H*G*p.')
    e.transfer.reset_warm_start()
    return out


@torch.no_grad()
def audit_saved_surface(endpoint, source, prior=None):
    """Sample the saved step on its actual frozen interaction geometry."""
    source=Path(source)
    pair=np.load(source/'budget_last_pair.npz')
    state=np.load(source/'diagnostic_state.npz')
    like=endpoint.model.mesh.X
    tensor=lambda value:torch.as_tensor(value,device=like.device,dtype=like.dtype)
    old=tensor(pair['previous_x']); nodal=tensor(pair['nodal_velocity'])
    if not np.array_equal(pair['x'],state['x']):
        raise ValueError('saved budget pair does not match diagnostic endpoint')
    velocity=tuple(tensor(state[f'velocity_{j}']) for j in range(3))
    report=json.loads((source/'report.json').read_text())
    decomposition=report['last']['budget_decomposition']
    audit=ClosedCavityMassAudit(old,endpoint.model.endo_faces,endpoint.grid,rim=endpoint.model.rim)
    triangles=audit.triangles(old); material=audit.triangles(nodal)
    levels=[]
    if prior:
        previous=json.loads(Path(prior).read_text())
        if Path(previous['source']).resolve()!=source.resolve() or previous['step']!=report['last']['step']:
            raise ValueError('surface refinement must reuse the same saved step')
        levels=previous['levels']
    for level in ((levels[-1]['level']+1,levels[-1]['level']+2) if prior else (2,3)):
        if prior:
            # The wall already converged to 1.4e-5 ml/s. Refine only the
            # long basal fan; do not repeat the existing wall integrations.
            wall=levels[-1]['wall']
        else:
            wall,_=audit.region(triangles[0],material[0],velocity,level)
        cap,_=audit.region(triangles[1],material[1],velocity,level+2)
        # Exact split of spatial wall/port defect, up to quadrature error:
        # (FE wall - kernel wall) + (kernel wall + kernel cap)
        # + (actual cap operator - independently integrated kernel cap).
        terms=dict(
            fe_projection_wall_ml_s=decomposition['projected_wall_outward_ml_s']-wall['fluid_flux_ml_s'],
            kernel_closed_surface_ml_s=wall['fluid_flux_ml_s']+cap['fluid_flux_ml_s'],
            port_vs_surface_cap_ml_s=decomposition['fluid_cap_outward_ml_s']-cap['fluid_flux_ml_s'])
        levels.append(dict(level=level,wall=wall,cap=cap,terms=terms,
            rt0_closed_flux_ml_s=wall['rt0_fluid_flux_ml_s']+cap['rt0_fluid_flux_ml_s'],
            decomposition_error_ml_s=sum(terms.values())-decomposition['spatial_wall_port_ml_s']))
    changes={key:abs(levels[-1]['terms'][key]-levels[-2]['terms'][key]) for key in levels[-1]['terms']}
    return dict(scope='saved step; frozen old geometry; read-only Peskin and RT0 surface quadrature',
        source=str(source),step=report['last']['step'],state_advanced=False,levels=levels,
        reused_surface_report=None if prior is None else str(prior),
        last_level_change_ml_s=changes,
        term_quadrature_converged_1e_minus3=all(v<1e-3 for v in changes.values()),
        interpretation='FE projection term compares the volume-L2-projected velocity trace with the '
            'regularized Euler velocity trace. Kernel term measures closed flux of the regularized '
            'field. RT0 is diagnostic only and has its own quadrature error; none is a new wall constraint.')


@torch.no_grad()
def audit_cap_rules(endpoint,source,reference,prior=None):
    """Compare integration rules with identical labels and pressure patches."""
    source=Path(source)
    pair=np.load(source/'budget_last_pair.npz')
    snap=np.load(source/'diagnostic_state.npz')
    if not np.array_equal(pair['x'],snap['x']): raise ValueError('saved step mismatch')
    ref=json.loads(Path(reference).read_text())
    if Path(ref['source']).resolve()!=source.resolve() or not ref['term_quadrature_converged_1e_minus3']:
        raise ValueError('matching converged surface reference required')
    exact=ref['levels'][-1]['cap']['fluid_flux_ml_s']
    like=endpoint.model.mesh.X
    tensor=lambda value:torch.as_tensor(value,device=like.device,dtype=like.dtype)
    x=tensor(pair['previous_x']); w=tensor(pair['nodal_velocity'])
    u=tuple(tensor(snap[f'velocity_{c}']) for c in range(3))
    config=json.loads((source/'config.resolved.json').read_text())
    if config['boundary_mode']!='filling': raise ValueError('this saved-field comparison requires a filling step')
    rows0=None; results={}
    if prior:
        previous=json.loads(Path(prior).read_text())
        if (Path(previous['source']).resolve()!=source.resolve() or
                Path(previous['reference']).resolve()!=Path(reference).resolve()):
            raise ValueError('prior cap-rule measurements must use the same fields and reference')
        results=previous['results']
    for rule in ('centroid','gauss3','gauss7','gauss12'):
        if rule in results and rule!='centroid': continue
        shadow=copy.copy(endpoint); shadow.owner=copy.copy(endpoint.owner)
        shadow.owner.cap_integration_rule=rule
        if hasattr(shadow,'_outlet_cap'): del shadow._outlet_cap
        if x.is_cuda: torch.cuda.synchronize(x.device)
        started=perf_counter()
        rows=outlet_rows(shadow,x,label=1)
        if x.is_cuda: torch.cuda.synchronize(x.device)
        seconds=perf_counter()-started
        if rows0 is None: rows0=rows
        if rule in results: continue  # Rebuild only partition metadata; reuse its measurement.
        if rows.count!=rows0.count: raise RuntimeError('quadrature changed pressure patch count')
        base=rows.reference; old=rows0.reference
        if rule!='centroid':
            parent=base.prepared.cap.integration_parent
            if not torch.equal(base.patches,old.patches[parent]):
                raise RuntimeError('quadrature changed pressure-patch assignment')
        flux=rows.flow(u); motion=rows.material_flow(w)
        probe=torch.sin(torch.arange(rows.count,device=x.device,dtype=x.dtype))+.7
        fluid=sum((a*b*face_weights(shadow.grid,c,x)).sum() for c,(a,b) in enumerate(zip(u,rows.lift(probe))))
        material=(rows.reaction(probe)*w).sum()
        expected=(probe*(flux-motion)).sum()
        relative=abs((fluid-material-expected).item())/max(abs(fluid.item())+abs(material.item())+abs(expected.item()),1.)
        results[rule]=dict(fluid_cap_outward_ml_s=flux.sum().item(),
            error_vs_surface_reference_ml_s=flux.sum().item()-exact,
            material_cap_change_ml_s=(motion-rows0.material_flow(w)).abs().max().item(),
            patch_area_max_change_cm2=(rows.area-rows0.area).abs().max().item(),
            paired_work_relative_error=relative,pressure_unknowns=rows.count,
            surface_points=len(base.prepared.points),setup_seconds=seconds)
        if relative>1e-10: raise RuntimeError('Gaussian cap work pairing failed')
    return dict(source=str(source),reference=str(reference),state_advanced=False,
        reused_measurements=None if prior is None else str(prior),
        reference_cap_flux_ml_s=exact,results=results,
        scope='same saved velocity, same material subtriangles and pressure patches; no FSI solve')
