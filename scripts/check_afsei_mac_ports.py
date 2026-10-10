"""Read-only moving-cap/port diagnostics on existing original-MAC checkpoints.

No fluid/solid/EP advance, valve condition, seal, or RCR pressure is imposed.
Sampled port flows are diagnostic; no arterial ODE is updated from them.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_ports import MACBasalPorts, prepare_flux_rows
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2
from afsi_torch.p1 import cavity_volume


def timed(function, repeats=1):
    torch.cuda.synchronize(); start = perf_counter()
    for _ in range(repeats):
        result = function()
    torch.cuda.synchronize()
    return result, (perf_counter()-start)/repeats


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def diagnose(path, level, repeats, output, write_vtk):
    (driver, state, _), load_s = timed(lambda: checkpoint.load(path))
    grid = driver._fsi_stepper(driver.maximum_ticks).grid
    cap, geometry_s = timed(lambda: MACBasalPorts(driver.mesh.X, driver.model.endo_faces, grid,
                                                 subdivision_level=level))
    x = state.fsi.x
    prepared, first_prepare_s = timed(lambda: cap.prepare(x))
    _, prepare_s = timed(lambda: cap.prepare(x), repeats)
    dt = state.force_load.dt_s
    w = (x-state.force_load.previous_x)/dt
    relative, flow_s = timed(lambda: prepared.relative_flow(state.fsi.velocity, w), repeats)
    absolute = prepared.rows.flow(state.fsi.velocity)
    material = prepared.material_flow(w)
    # Test coefficients only. These pressures do not change any physical state.
    pressure = x.new_tensor([0., 8., 85.])*MMHG_TO_DYN_CM2
    density, density_s = timed(lambda: prepared.rows.pressure_density(pressure), repeats)
    reaction, reaction_s = timed(lambda: prepared.material_reaction(pressure), repeats)
    fluid_power = sum((u*f).sum()*grid.volume for u, f in zip(state.fsi.velocity, density))
    fluid_target = -(pressure*absolute).sum()
    total_power = fluid_power+(reaction*w).sum()
    target_power = -(pressure*relative).sum()
    relative_power_error = ((total_power-target_power).abs()/
        (fluid_power.abs()+(reaction*w).abs().sum()+target_power.abs()).clamp_min(1e-30)).item()
    if relative_power_error > 1e-12:
        raise AssertionError('moving-cap pressure/relative-flow virtual work mismatch')
    # Independently integrate endocardial material flux and compare DV(x).w.
    nodes = x[cap.endo]; wall_velocity = w[cap.endo]
    wall_area = -.5*torch.linalg.cross(nodes[:, 1]-nodes[:, 0], nodes[:, 2]-nodes[:, 0])
    wall_material = (wall_area*wall_velocity.mean(1)).sum()
    geometric_rate = wall_material+material.sum()
    dv = torch.func.jvp(lambda value: cavity_volume(value, cap.endo, cap.rim), (x,), (w,))[1]
    kinematic_error = (geometric_rate-dv).abs().item()
    if kinematic_error > 1e-8*(1+dv.abs().item()):
        raise AssertionError('closed material surface and cavity volume derivative disagree')
    # Three-point triangle rule for sampled fluid wall flux; this is not an IB
    # no-slip assertion and not an exact MAC discrete Gauss identity.
    N = x.new_tensor([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    wall_points = torch.einsum('qa,fai->fqi', N, nodes).reshape(-1, 3)
    wall_areas = (wall_area[:, None].expand(-1, 3, -1)/3).reshape(-1, 3)
    wall_rows, wall_prepare_s = timed(lambda: prepare_flux_rows(grid, wall_points, wall_areas,
        torch.zeros(len(wall_points), device=x.device, dtype=torch.int64)))
    wall_absolute = wall_rows.flow(state.fsi.velocity)[0]
    wall_slip = wall_absolute-wall_material
    closed_fluid_flux = absolute.sum()+wall_absolute
    volume_current = cavity_volume(x, cap.endo, cap.rim)
    volume_previous = cavity_volume(state.force_load.previous_x, cap.endo, cap.rim)
    region_area = cap.region_sum(torch.linalg.vector_norm(prepared.area_vectors, dim=-1))
    region_center = cap.region_sum(prepared.points*torch.linalg.vector_norm(prepared.area_vectors, dim=-1)[:, None])/region_area[:, None]
    report = dict(checkpoint=str(path), time_s=state.fsi.time, step=state.fsi.step,
        fluid_shape=grid.shape, fluid_discretization='original Cartesian staggered MAC',
        geometry=cap.metadata, current_region_area_cm2=region_area.cpu().tolist(),
        current_region_center_cm=region_center.cpu().tolist(),
        current_rim_warp_cm=(x[cap.rim[:, 0]]-x[cap.rim[:, 0]].mean(0))@cap.frame[-1],
        coalesced_flux_coefficients=[len(c[0]) for c in prepared.rows.components],
        sampled_absolute_outward_flux_ml_s=absolute.cpu().tolist(),
        material_cap_outward_flux_ml_s=material.cpu().tolist(),
        sampled_relative_outward_flux_ml_s=relative.cpu().tolist(),
        sampled_q_in_mv_ml_s=-relative[1].item(), sampled_q_out_av_ml_s=relative[2].item(),
        non_port_cap_relative_leak_ml_s=relative[0].item(),
        cavity_volume_ml=volume_current.item(),
        material_volume_rate_ml_s=dv.item(), secant_volume_rate_ml_s=((volume_current-volume_previous)/dt).item(),
        port_only_mass_defect_ml_s=(dv+relative[1:].sum()).item(),
        all_cap_mass_defect_ml_s=(dv+relative.sum()).item(),
        sampled_endocardial_relative_flux_ml_s=wall_slip.item(),
        sampled_closed_surface_absolute_flux_ml_s=closed_fluid_flux.item(),
        material_volume_flux_identity_error_ml_s=kinematic_error,
        fluid_pressure_work_error_erg_s=(fluid_power-fluid_target).abs().item(),
        relative_pressure_work_error_erg_s=(total_power-target_power).abs().item(),
        relative_pressure_work_relative_error=relative_power_error,
        timings_seconds=dict(checkpoint_load=load_s, reference_cap_setup=geometry_s,
            first_prepare=first_prepare_s, mean_cap_prepare=prepare_s,
            mean_relative_flux=flow_s, mean_pressure_density=density_s, mean_material_reaction=reaction_s,
            wall_diagnostic_prepare=wall_prepare_s),
        ep_advanced=False, fluid_advanced=False, myocardium_motion_advanced=False,
        cap_sealed=False, physical_valves_imposed=False, windkessel_ports_connected=False,
        three_field_integration_implemented=False,
        flow_scope='sampled from the existing unsealed MAC field; not a physiological valve solution')
    report['current_rim_warp_cm'] = report['current_rim_warp_cm'].abs().max().item()
    if write_vtk:
        import meshio
        endpoints = x[cap.rim]; center = x[cap.rim[:, 0]].mean(0)
        fan = torch.cat((endpoints, center[None, None].expand(len(cap.rim), 1, 3)), 1)
        sub = torch.einsum('qka,eai->eqki', cap.subtriangles, fan).reshape(-1, 3, 3)
        vertices = sub.reshape(-1, 3).cpu().numpy()
        cells = torch.arange(len(vertices)).reshape(-1, 3).numpy()
        meshio.write(output/'basal_ports.vtu', meshio.Mesh(vertices, [('triangle', cells)],
            cell_data={'port_region': [cap.labels.cpu().numpy()]}))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--subdivision-level', type=int, default=4)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output', type=Path, default=Path('results/mac_ports_diagnostics'))
    p.add_argument('--write-vtk', action='store_true')
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('actual CUDA required')
    if args.repeats < 1 or args.repeats > 100 or not 0 <= args.subdivision_level <= 6:
        p.error('repeats must be 1..100 and subdivision level 0..6')
    if (args.output/'report.json').exists():
        p.error('output report already exists; choose a fresh directory')
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, path in enumerate(args.checkpoint):
        source_hash = checkpoint_digest(path)
        subdir = args.output/f'snapshot_{index}'
        subdir.mkdir(exist_ok=True)
        reports.append(diagnose(path, args.subdivision_level, args.repeats, subdir, args.write_vtk))
        if checkpoint_digest(path) != source_hash:
            raise AssertionError('source checkpoint was modified during diagnostics')
        reports[-1].update(source_checkpoint_sha256=source_hash, source_checkpoint_unchanged=True)
        print(f'Checked snapshot {index}: t={reports[-1]["time_s"]:.6f} s', flush=True)
    result = dict(passed=True, gpu=torch.cuda.get_device_name(), scope=__doc__, snapshots=reports)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
