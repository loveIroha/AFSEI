"""Read-only original-MAC pressure/load and closed-surface slip audit."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_pressure_audit import audit_pressure_interface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', nargs='+', required=True)
    p.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--guard-cells', type=int, default=2)
    p.add_argument('--skip-force-decomposition', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        p.error('actual CUDA required')
    if (len(a.levels) < 2 or sorted(set(a.levels)) != a.levels
            or any(k < 0 or k > 4 for k in a.levels) or not 0 <= a.guard_cells <= 4):
        p.error('>=2 increasing levels in 0..4 and guard-cells in 0..4 required')
    a.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for path in a.checkpoint:
        path = Path(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        driver, state, progress = checkpoint.load(path)
        torch.cuda.synchronize(); started = perf_counter()
        report = audit_pressure_interface(driver, state, levels=tuple(a.levels),
            guard_cells=a.guard_cells, decompose=not a.skip_force_decomposition)
        torch.cuda.synchronize()
        report.update(checkpoint=str(path), checkpoint_sha256=sha,
            audit_seconds=perf_counter()-started, gpu=torch.cuda.get_device_name(),
            circuit_condition_probe='circuit_condition_probe' in progress)
        if sha != hashlib.sha256(path.read_bytes()).hexdigest():
            raise AssertionError('source checkpoint changed')
        report['source_checkpoint_unchanged'] = True
        reports.append(report)
        (a.output/'report.json').write_text(json.dumps(dict(
            status='completed' if len(reports) == len(a.checkpoint) else 'incomplete',
            snapshots=reports), indent=2, allow_nan=False)+'\n')
        last = report['wall_levels'][-1]['current']
        print(json.dumps(dict(checkpoint=str(path), control_mmhg=report['source_wall_control_mmhg'],
            mac_core_mean_mmhg=report['mac_cavity_core_pressure']['mean_mmhg'],
            av_multiplier_mmhg=report['av_constraint_multiplier_mmhg'],
            wall_signed_slip_ml_s=last['relative_flux_ml_s'],
            wall_absolute_slip_ml_s=last['absolute_relative_flux_ml_s'],
            closed_surface_absolute_slip_ml_s=report['closed_surface_absolute_slip_ml_s'],
            original_ib_motion_residual_l2_cm=report['original_ib_motion_residual_l2_cm'],
            audit_seconds=report['audit_seconds']), indent=2), flush=True)
        del driver, state
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
