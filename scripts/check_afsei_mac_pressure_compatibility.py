"""Read-only GPU static pressure-load comparison; never writes a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_pressure_compatibility import audit_preload_loads


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--levels', nargs='+', type=int, default=[2, 3])
    p.add_argument('--guard-cells', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('actual CUDA required')
    sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    driver, state, progress = checkpoint.load(a.checkpoint)
    preload = progress.get('preload', {})
    if not (preload.get('preload_performed') is True and preload.get('loading_history_steps') == 100
            and preload.get('strain_history_advanced_during_preload') is True):
        p.error('source 100-step preload with Land strain history required')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize(); started = perf_counter()
    report = audit_preload_loads(driver, state, levels=tuple(a.levels), guard_cells=a.guard_cells)
    torch.cuda.synchronize()
    if sha != hashlib.sha256(a.checkpoint.read_bytes()).hexdigest():
        raise AssertionError('source checkpoint changed')
    report.update(checkpoint=str(a.checkpoint), checkpoint_sha256=sha,
        source_checkpoint_unchanged=True, candidate_checkpoint_written=False,
        elapsed_seconds=perf_counter()-started, gpu=torch.cuda.get_device_name())
    (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], manufactured_control_passed=report['manufactured_control_passed'],
        production_handoff_enabled=False, elapsed_seconds=report['elapsed_seconds'],
        loads={name:dict(core_mmhg=r['mac_core_pressure']['mean_mmhg'],
            max_velocity_cm_s=r['max_velocity_cm_s'], non_gradient_fraction=r['non_gradient_density_fraction'])
            for name, r in report['results'].items()}), indent=2), flush=True)


if __name__ == '__main__': main()
