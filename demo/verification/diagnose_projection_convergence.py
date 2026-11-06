"""Frozen-field IB/P1 projection study; never advances EP, ODEs or FSI.

All refinement levels represent the SAME saved piecewise-affine physical
wall. Metrics use a common current-volume quadrature, not disparate mesh
norms. The analytical Peskin derivative is diagnostic only: no production
operator or numerical method is replaced.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
import time
import numpy as np
import torch
from afsi_torch.p1 import prepare_p1
from afsi_torch.ib import peskin4
from afsi_torch.ep.refined_mesh import nested_mesh, lattice_template
from afsi_torch.fluid.solvers import SolverOptions
from afsi_torch.mac.grid import MACGrid, divergence
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer, InteractionQuadratureOptions
from afsi_torch.mac.compact_transfer import CompactStencil, _prepare


def derivative_phi(s):
    r = s.abs()
    a, b = r.clamp(max=1), r.clamp(min=1, max=2)
    da = (-2 + (2-4*a)/torch.sqrt(1+4*a-4*a*a))/8
    db = (-2 - (6-4*b)/torch.sqrt(-7+12*b-4*b*b))/8
    return s.sign()*torch.where(r < 1, da, torch.where(r < 2, db, torch.zeros_like(r)))


def sample_component(points, value, origin, spacing, component):
    """Value and own-coordinate derivative of the production 4-point kernel."""
    scaled = (points-origin)/spacing
    base = torch.floor(scaled-1).long()
    ids = base[..., None]+torch.arange(4, device=points.device)
    r = scaled[..., None]-ids.to(points.dtype)
    phi, dphi = peskin4(r), derivative_phi(r)
    flat = ((ids[:, 0, :, None, None]*value.shape[1]+ids[:, 1, None, :, None])
            *value.shape[2]+ids[:, 2, None, None, :])
    f = value.flatten()[flat]
    w = phi[:, 0, :, None, None]*phi[:, 1, None, :, None]*phi[:, 2, None, None, :]
    if component == 0:
        dw = dphi[:, 0, :, None, None]*phi[:, 1, None, :, None]*phi[:, 2, None, None, :]
    elif component == 1:
        dw = phi[:, 0, :, None, None]*dphi[:, 1, None, :, None]*phi[:, 2, None, None, :]
    else:
        dw = phi[:, 0, :, None, None]*phi[:, 1, None, :, None]*dphi[:, 2, None, None, :]
    return (f*w).sum((1, 2, 3)), (f*dw).sum((1, 2, 3))/spacing[component]


def physical_divergence(geometry, x, velocity):
    grad = geometry.gradients[:, 0]
    F = torch.einsum('eai,eaj->eij', x[geometry.cells], grad)
    G = torch.einsum('eai,eaj->eij', velocity[geometry.cells], grad)
    return torch.einsum('eij,eji->e', G, torch.linalg.inv(F)), torch.linalg.det(F)


def child_locator(factor, like):
    bary, cells = lattice_template(factor)
    vertices = like.new_tensor(bary[cells, 1:]/factor)
    inv = torch.linalg.inv((vertices[:, 1:]-vertices[:, :1]).transpose(1, 2))
    return vertices[:, 0], inv


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'status.json').write_text(json.dumps({'status': 'running'}))
    source = Path(cfg['checkpoint'])
    source_config = Path(cfg['source_config'])
    c = json.loads(source_config.read_text())
    a = np.load(source)
    device = cfg.get('device', 'cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('This study requires actual GPU validation')
    torch.set_num_threads(2)
    cast = lambda v: torch.as_tensor(np.asarray(v).copy(), device=device, dtype=torch.float64)
    grid = MACGrid(tuple(c['fluid']['shape']), tuple(c['fluid']['lengths_cm']), tuple(c['fluid']['origin_cm']))
    velocity = tuple(cast(a[k]) for k in ('u', 'v', 'w'))
    X0, C0, x0 = a['reference'].copy(), a['cells'].copy(), a['x'].copy()
    factors = cfg['factors']
    if factors != [1, 2, 3]:
        raise ValueError('Common quadrature/locator validation currently covers factors 1,2,3')
    t0 = time.perf_counter()
    dv = divergence(velocity, grid.spacing)
    result = dict(scope='frozen 120ms field; no time advancement; exact prolongation of identical polyhedral wall',
        config=cfg, checkpoint_time_s=float(a['time_s']), checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_config=c, native_mac_divergence=dict(max_s_inv=float(dv.abs().max()), rms_s_inv=float(dv.square().mean().sqrt())),
        device=torch.cuda.get_device_name(), levels=[], checks={})
    s = cast([-1.8, -1.1, -.5, 0., .2, .9, 1.4, 1.9])
    finite_difference = (peskin4(s+1e-6)-peskin4(s-1e-6))/(2e-6)
    derivative_error = float((derivative_phi(s)-finite_difference).abs().max())
    if derivative_error > 1e-8:
        raise AssertionError('Kernel analytical derivative failed finite-difference check')
    result['checks']['kernel_derivative_max_error'] = derivative_error
    fe_div, cell_velocity, nodal_velocity = {}, {}, {}
    reference_volume = None
    for factor in factors:
        begin = time.perf_counter()
        if factor == 1:
            X, cells, x = X0, C0, x0
        else:
            data = nested_mesh(X0, C0, factor)
            X, cells = data['X'], data['cells']
            x = (x0[data['coarse_ids']]*data['weights'][..., None]).sum(1)
            del data
        X, cells, x = cast(X), torch.as_tensor(cells.copy(), device=device), cast(x)
        geometry = prepare_p1(X, cells, degree=2)  # Exact constant consistent P1 mass.
        transfer = AdaptiveP1Transfer(grid, geometry, options=SolverOptions(**c['fsi']['mass_solver']),
            warm_start=False, mass_backend=c['fsi']['execution']['mass_backend'],
            quadrature_options=InteractionQuadratureOptions(**c['fsi']['interaction_quadrature']), fused=True)
        stencil = transfer.prepare(x)
        v, info = transfer.interpolate(velocity, stencil)
        div, J = physical_divergence(geometry, x, v)
        fe_div[factor] = div.clone()
        nodal_velocity[factor] = v.clone()
        # Template child order uses globally sorted ORIGINAL parent vertices.
        vcells = torch.sort(cells, dim=1).values if factor == 1 else cells
        cell_velocity[factor] = v[vcells].clone()
        weights = geometry.weights.sum(1)*J
        current_volume = float(weights.sum())
        if reference_volume is None:
            reference_volume = current_volume
        if float(J.min()) <= 0 or abs(current_volume/reference_volume-1) > 1e-11:
            raise AssertionError('The three levels must have identical positive physical geometry')
        record = dict(factor=factor, nodes=len(X), cells=len(cells),
            interaction=transfer.prepared_quadrature_statistics(), mass_solve=asdict(info),
            current_wall_volume_ml=current_volume, current_volume_relative_difference=current_volume/reference_volume-1,
            divergence_rms_s_inv=float((weights*div.square()).sum().div(weights.sum()).sqrt()),
            divergence_mean_s_inv=float((weights*div).sum()/weights.sum()),
            divergence_max_s_inv=float(div.abs().max()), elapsed_setup_and_projection_s=time.perf_counter()-begin)
        result['levels'].append(record)
        print(json.dumps({'phase': 'projection', **record}), flush=True)
        (out/'partial.json').write_text(json.dumps(result, indent=2))
        if factor != 3:
            del transfer, stencil, geometry, X, cells, x, v, J, weights
            gc.collect()
            torch.cuda.empty_cache()
    # Check analytical regularized divergence and value against production.
    sample_points = x[::max(1, len(x)//23)][:23]
    h = sample_points.new_tensor(grid.spacing)
    origins = sample_points.new_tensor([grid.face_origin(i) for i in range(3)])
    sample = torch.stack([sample_component(sample_points, velocity[i], origins[i], h, i)[0] for i in range(3)], 1)
    b, p = _prepare(sample_points, origins, h, torch.arange(4, device=device))
    native_sample = transfer.gather_grid(velocity, CompactStencil(b.contiguous(), p.contiguous()))
    sample_error = float((sample-native_sample).abs().max())
    if sample_error > 1e-10*max(1., float(sample.abs().max())):
        raise AssertionError('Diagnostic sampling differs from production GPU IB gather')
    result['checks']['sampler_vs_production_max_cm_s'] = sample_error
    own_derivative = sum(sample_component(sample_points, velocity[i], origins[i], h, i)[1] for i in range(3))
    fd_div = torch.zeros_like(own_derivative)
    for i in range(3):
        shift = torch.zeros_like(sample_points); shift[:, i] = 1e-6*h[i]
        fd_div += (sample_component(sample_points+shift, velocity[i], origins[i], h, i)[0]
                   -sample_component(sample_points-shift, velocity[i], origins[i], h, i)[0])/(2e-6*h[i])
    field_fd_error = float((own_derivative-fd_div).abs().max())
    if field_fd_error > 2e-6:
        raise AssertionError('Analytical divergence failed field finite-difference check')
    result['checks']['field_divergence_finite_difference_error_s_inv'] = field_fd_error
    affine_gradient = cast([[.3, .2, -.1], [.1, -.4, .2], [-.2, .1, .1]])
    affine = tuple((grid.coordinates(i, device=device)@affine_gradient[i])+float(i) for i in range(3))
    affine_div = sum(sample_component(sample_points, affine[i], origins[i], h, i)[1] for i in range(3))
    affine_error = float(affine_div.abs().max())
    if affine_error > 1e-11:
        raise AssertionError('Trace-zero affine velocity must remain divergence-free')
    result['checks']['affine_trace_zero_divergence_error_s_inv'] = affine_error
    # A factor-6 TEMPLATE is a common refinement of 1,2,3. No factor-6 FE
    # solver/mesh is built. Using factor-3 quadrature alone would underresolve
    # jumps of factor-2 gradients. Verify coverage and exact child measures.
    coarse_sorted = torch.as_tensor(np.sort(C0, axis=1).copy(), device=device)
    coarse_x = cast(x0)[coarse_sorted]
    cb, cc = lattice_template(6)
    cb = cast(cb[cc]/6)
    qb = geometry.values
    common_bary = torch.einsum('qa,cai->cqi', qb, cb).reshape(-1, 4)
    ratio = torch.linalg.det((cb[:, 1:, 1:]-cb[:, :1, 1:]).transpose(1, 2)).abs()
    if float((ratio-1/216).abs().max()) > 1e-14:
        raise AssertionError('Invalid common-refinement reference cell measure')
    maps, basis = {}, {}
    for factor in factors:
        loc, inv = child_locator(factor, x)
        local = torch.einsum('cij,ecj->eci', inv, common_bary[:, None, 1:]-loc[None])
        bary = torch.cat((1-local.sum(2, keepdim=True), local), 2)
        valid = ((bary >= -2e-12) & (bary <= 1+2e-12)).all(2)
        if not bool(valid.any(1).all()):
            raise AssertionError('Invalid common-refinement child coverage')
        maps[factor] = valid.to(torch.long).argmax(1)
        basis[factor] = bary[torch.arange(len(bary), device=device), maps[factor]]
        counts = torch.bincount(maps[factor], minlength=factor**3)
        if not bool((counts == len(common_bary)//factor**3).all()):
            raise AssertionError('Common quadrature must integrate every FE child equally')
    coarse_current_volume = (geometry.weights.sum(1)*J).reshape(len(C0), 27).sum(1)
    accum = x.new_zeros(2+len(factors)*4)  # W, div_delta integral; four metrics per FE level.
    delta_sq = x.new_zeros(())
    velocity_error = x.new_zeros(len(factors))
    delta_velocity_sq = x.new_zeros(())
    sampler = torch.compile(sample_component, fullgraph=True, dynamic=True,
                            options={'triton.cudagraphs': False})
    chunk = int(cfg.get('chunk_points', 16384))
    qcount = len(common_bary)
    cell_chunk = max(1, chunk//qcount)
    for start in range(0, len(C0), cell_chunk):
        stop = min(start+cell_chunk, len(C0))
        qx = torch.einsum('qa,eai->eqi', common_bary, coarse_x[start:stop]).reshape(-1, 3)
        W = (coarse_current_volume[start:stop, None]/qcount).expand(-1, qcount).reshape(-1)
        parent = torch.arange(start, stop, device=device)[:, None]
        sampled = [sampler(qx, velocity[i], origins[i], h, i) for i in range(3)]
        delta = sum(z[1] for z in sampled)
        delta_velocity = torch.stack([z[0] for z in sampled], 1)
        accum[0] += W.sum(); accum[1] += (W*delta).sum()
        delta_sq += (W*delta.square()).sum()
        delta_velocity_sq += (W*delta_velocity.square().sum(1)).sum()
        for n, factor in enumerate(factors):
            index = (parent*factor**3+maps[factor][None]).reshape(-1)
            d = fe_div[factor][index]
            offset = 2+4*n
            accum[offset] += (W*d).sum()
            accum[offset+1] += (W*d.square()).sum()
            accum[offset+2] += (W*(d-delta).square()).sum()
            accum[offset+3] += (W*d*delta).sum()
            vv = cell_velocity[factor][index].reshape(stop-start, qcount, 4, 3)
            fe_velocity = torch.einsum('eqai,qa->eqi', vv, basis[factor]).reshape(-1, 3)
            velocity_error[n] += (W*(fe_velocity-delta_velocity).square().sum(1)).sum()
    print(json.dumps({'phase': 'common-quadrature', 'processed_parent_cells': len(C0)}), flush=True)
    values = accum.cpu().numpy()
    Wtotal = values[0]
    if abs(Wtotal/reference_volume-1) > 1e-10:
        raise AssertionError('Common quadrature must recover exact physical wall volume')
    result['common_quadrature'] = dict(point_count=len(C0)*qcount, template_factor=6,
        points_per_template_cell=len(qb), volume_ml=Wtotal,
        delta_velocity_rms_cm_s=float((delta_velocity_sq/accum[0]).sqrt()),
        delta_divergence_rms_s_inv=float((delta_sq/accum[0]).sqrt()),
        delta_divergence_mean_s_inv=values[1]/Wtotal,
        meaning='Analytical divergence of the unchanged componentwise regularized MAC velocity, same wall/domain/instant')
    for n, record in enumerate(result['levels']):
        k = 2+4*n
        record['common_metrics'] = dict(mean_divergence_s_inv=values[k]/Wtotal,
            rms_divergence_s_inv=float(np.sqrt(values[k+1]/Wtotal)),
            rms_difference_from_regularized_divergence_s_inv=float(np.sqrt(values[k+2]/Wtotal)),
            divergence_cross_product_s_inv2=values[k+3]/Wtotal)
        record['common_metrics']['velocity_relative_l2_difference_from_regularized_field'] = float((velocity_error[n]/delta_velocity_sq).sqrt())
        if abs(record['divergence_rms_s_inv']-record['common_metrics']['rms_divergence_s_inv']) > 1e-9:
            raise AssertionError('FE divergence common quadrature/child locator is inconsistent')
    result['checks']['all_passed'] = True
    result['elapsed_seconds'] = time.perf_counter()-t0
    result['limitations'] = ['One frozen field at 120ms; not a new time trajectory or a global volume budget.',
        'The accepted x is used, whereas the preceding BE step froze the IB operator at previous_x.',
        'Reference-volume mass projection; current-volume diagnostic norms.',
        'No change to material, EP, MAC, IB kernel, quadrature criterion, nonlinear solver or time integration.',
        'These setup/projection timings are not a steady-state performance comparison.']
    np.savez_compressed(out/'projected_velocity.npz',
        **{f'div_factor{k}': v.cpu().numpy() for k, v in fe_div.items()},
        **{f'v_factor{k}': v.cpu().numpy() for k, v in nodal_velocity.items()})
    (out/'report.json').write_text(json.dumps(result, indent=2))
    (out/'config.json').write_text(json.dumps(cfg, indent=2))
    (out/'status.json').write_text(json.dumps({'status': 'completed', 'all_checks_passed': True}))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
