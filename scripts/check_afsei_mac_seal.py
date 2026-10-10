"""Validate local basal cap projection on immutable original-MAC checkpoints.

Corrected velocities are diagnostic only. No FSI checkpoint is written and
no EP, solid, valve or RCR evolution is performed. Material rim motion is
prescribed from each accepted step, not re-solved under the cap reaction.
"""
import argparse
import json
from pathlib import Path
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.ep.mac_seal import prepare_local_cap, MACCapProjection
from check_afsei_mac_ports import timed, checkpoint_digest


def diagnose(path, level, width, repeats):
    (driver, state, _), load_s = timed(lambda: checkpoint.load(path))
    stepper = driver._fsi_stepper(driver.maximum_ticks)
    cap, cap_s = timed(lambda: MACBasalPorts(driver.mesh.X, driver.model.endo_faces,
                                           stepper.grid, subdivision_level=level))
    prepared, prepared_s = timed(lambda: cap.prepare(state.fsi.x))
    rows, rows_s = timed(lambda: prepare_local_cap(prepared, patch_width_cells=width))
    projection = MACCapProjection(rows, stepper.flow.pressure_solver)
    dt = state.force_load.dt_s
    w = (state.fsi.x-state.force_load.previous_x)/dt
    u = state.fsi.velocity
    before = rows.diagnostics(u, w)
    initial, first_s = timed(lambda: projection.solve(u, w, dt=dt, density=stepper.flow.rho))
    result, solve_s = timed(lambda: projection.solve(u, w, dt=dt, density=stepper.flow.rho), repeats)
    projected, dp, pressure, reaction, solve_report = result
    after = rows.diagnostics(projected, w)
    # Relative scale matters with moving-wall external work.
    work_scale = max(1., abs(solve_report['cap_fluid_power_erg_s'])
                        +abs(solve_report['cap_material_power_erg_s']))
    work_error = abs(solve_report['cap_total_power_erg_s'])/work_scale
    if work_error > 1e-7 or solve_report['energy_balance_relative_error'] > 1e-8:
        raise AssertionError('cap pressure work or kinetic-energy identity failed')
    velocity_scale = max(1., *(a.abs().max().item() for a in projected))
    if solve_report['max_divergence_after_s_inv'] > 1e-8*velocity_scale/min(stepper.grid.spacing):
        raise AssertionError('local seal projection broke MAC incompressibility')
    return dict(checkpoint=str(path), time_s=state.fsi.time, step=state.fsi.step,
        grid_shape=stepper.grid.shape, fluid_discretization='original Cartesian staggered MAC',
        patch_width_cells=width, patch_count=rows.count,
        patch_area_min_cm2=rows.area.min().item(), patch_area_max_cm2=rows.area.max().item(),
        non_port_area_cm2=rows.area.sum().item(), before=before, after=after,
        cap_relative_flux_before_ml_s=prepared.relative_flow(u, w).cpu().tolist(),
        cap_relative_flux_after_ml_s=prepared.relative_flow(projected, w).cpu().tolist(),
        patch_pressure_min_dyn_cm2=pressure.min().item(), patch_pressure_max_dyn_cm2=pressure.max().item(),
        max_cap_reaction_dyn=torch.linalg.vector_norm(reaction, dim=-1).max().item(),
        max_pressure_increment_dyn_cm2=dp.abs().max().item(), solver=solve_report,
        relative_cap_work_error=work_error,
        timings_seconds=dict(checkpoint_load=load_s, reference_cap=cap_s,
            cap_flux_rows=prepared_s, local_constraint_assembly=rows_s,
            first_projection=first_s, mean_projection=solve_s, repeats=repeats),
        local_patch_constraints_imposed_in_diagnostic=True,
        pointwise_no_penetration_imposed=False, corrected_velocity_is_diagnostic_only=True,
        ep_advanced=False, myocardium_motion_advanced=False, coupled_fsi_advanced=False,
        mainline_cap_sealed=False, physical_valves_imposed=False,
        windkessel_ports_connected=False, three_field_integration_implemented=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', nargs='+', required=True)
    parser.add_argument('--subdivision-level', type=int, default=4)
    parser.add_argument('--patch-width-cells', type=float, default=3.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('results/mac_seal_validation'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('actual CUDA required')
    if not 1 <= args.repeats <= 20 or not 0 <= args.subdivision_level <= 6:
        parser.error('repeats must be 1..20 and subdivision level 0..6')
    if (args.output/'report.json').exists():
        parser.error('choose a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in args.checkpoint:
        source_hash = checkpoint_digest(path)
        report = diagnose(path, args.subdivision_level, args.patch_width_cells, args.repeats)
        if source_hash != checkpoint_digest(path):
            raise AssertionError('source checkpoint changed')
        report.update(source_checkpoint_sha256=source_hash, source_checkpoint_unchanged=True)
        reports.append(report)
        print(f'Validated t={report["time_s"]:.6f}s: {report["patch_count"]} patches, '
              f'net leak {report["before"]["net_leak_ml_s"]:.6g} -> '
              f'{report["after"]["net_leak_ml_s"]:.6g} ml/s, '
              f'{report["timings_seconds"]["mean_projection"]:.6g}s/projection', flush=True)
    result = dict(passed=True, gpu=torch.cuda.get_device_name(), scope=__doc__, snapshots=reports)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
