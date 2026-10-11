"""Recompute source strain history and validate pressure-supported MAC preload.

Writes a distinct static preload result, never an ordinary run checkpoint.
The input checkpoint supplies reference mesh/configuration only; its previous
geometry and loading history are not imported into the new loading path.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_isovolumic import IsovolumicAFSEIStepper
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_balanced_preload import solve, validate, save_result, save_loading_result, reuse_validated_loading
from afsi_torch.two_way.preload import PreloadConfig, PreloadFailure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pressure-mmhg', type=float, default=8.)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--level', type=int, default=2)
    p.add_argument('--reuse-balanced-result', type=Path,
                   help='reuse this NEW checksummed loading path, e.g. at another surface level')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available(): p.error('actual CUDA required')
    if not 1 <= a.trials <= 20 or not 0 <= a.level <= 4:
        p.error('trials in 1..20 and level in 0..4 required')
    settings = PreloadConfig(target_mmhg=a.pressure_mmhg, nominal_steps=100,
        minimum_step_mmhg=.005, force_atol_dyn=1e-5, force_rtol=1e-10)
    sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    reuse_sha = (hashlib.sha256(a.reuse_balanced_result.read_bytes()).hexdigest()
                 if a.reuse_balanced_result else None)
    old, source, _ = checkpoint.load(a.checkpoint)
    config = replace(old.config, prescribed_endocardial_pressure_mmhg=0.)
    driver = IsovolumicAFSEIStepper(old.mesh, config)
    old_volume = MACEndocardialPressure(driver.model).volume(source.fsi.x).item()
    del old, source
    torch.cuda.empty_cache()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize(); started = perf_counter()
    def progress(row):
        print(f"Preload P={row['pressure_mmhg']:.4f}mmHg V={row['volume_ml']:.6f}ml "
              f"residual={row['residual_norm_dyn']:.3g}dyn NR={row['newton_iterations']}", flush=True)
    try:
        if a.reuse_balanced_result is None:
            result, history = solve(driver, settings, progress=progress)
        else:
            result, history = reuse_validated_loading(driver, a.reuse_balanced_result)
            if result.pressure_mmhg != a.pressure_mmhg or history.steps != 100 or history.duration_s != .5:
                p.error('reused balanced result must match target and 100 x 5ms source history')
        torch.cuda.synchronize(); preload_seconds = perf_counter()-started
        save_loading_result(a.output/'loading_result.npz', driver, result, history)
        (a.output/'loading_history.json').write_text(json.dumps(dict(
            status='completed', preload_seconds=preload_seconds,
            input_sha256=sha, balanced_loading_result_reused=a.reuse_balanced_result is not None,
            history=result.history, rejected=result.rejected), indent=2, allow_nan=False)+'\n')
        prepared, state, bp, report = validate(driver, result, history, steps=a.trials, level=a.level)
        torch.cuda.synchronize()
        if sha != hashlib.sha256(a.checkpoint.read_bytes()).hexdigest():
            raise AssertionError('input checkpoint changed')
        if reuse_sha and reuse_sha != hashlib.sha256(a.reuse_balanced_result.read_bytes()).hexdigest():
            raise AssertionError('reused loading result changed')
        report.update(input_checkpoint=str(a.checkpoint), input_sha256=sha, input_checkpoint_unchanged=True,
            old_preload_volume_ml=old_volume, preload_seconds=preload_seconds,
            elapsed_seconds=perf_counter()-started, gpu=torch.cuda.get_device_name(),
            grid_shape=prepared.config.fsi.fluid.shape, loading_steps=len(result.history),
            rejected_loading_attempts=len(result.rejected),
            balanced_loading_result_reused=a.reuse_balanced_result is not None,
            reused_loading_sha256=reuse_sha,
            candidate_result_written=False, ordinary_checkpoint_written=False)
        if report['static_targets_passed']:
            save_result(a.output/'preload_result.npz', prepared, state, bp, report, result.history)
            report['candidate_result_written'] = True
        (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        if not report['static_targets_passed']:
            raise SystemExit('static targets failed; validated preload result was not published')
    except PreloadFailure as exc:
        report = dict(status='preload_failed', error=str(exc), input_sha256=sha,
            history=exc.history, rejected=exc.rejected, candidate_result_written=False)
        (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        raise
    except Exception as exc:
        report = dict(status='validation_failed', error=repr(exc), input_sha256=sha,
            candidate_result_written=False, ordinary_checkpoint_written=False,
            completed_loading_preserved=(a.output/'loading_result.npz').exists())
        (a.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        raise


if __name__ == '__main__': main()
