"""Same-state GPU benchmark: exact analytic vs previous AD active tangent.

Advances scratch states only; writes a benchmark report, never a checkpoint.
Each route uses the saved original MAC solver configuration and tolerances.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.driver import AFSEIStepper
from afsi_torch.ep.solid import EPDrivenTangent


@contextmanager
def legacy_tangent(enabled):
    original = EPDrivenTangent.__init__
    def init(self, model, chunk_size=2048):
        original(self, model, chunk_size)
        kernel = torch.vmap(torch.func.jacrev(model.active_load._element_force, argnums=0))
        self._active_kernel = lambda *args: kernel(*args[:7])
    if enabled:
        EPDrivenTangent.__init__ = init
    try:
        yield
    finally:
        EPDrivenTangent.__init__ = original


def physical_pairs(a, b):
    return [('x_cm', a.fsi.x, b.fsi.x), ('pressure_cgs', a.fsi.pressure, b.fsi.pressure),
        ('force_dyn', a.fsi.force, b.fsi.force), ('voltage_v', a.tissue.voltage_v, b.tissue.voltage_v),
        ('gpb', a.tissue.gpb, b.tissue.gpb), ('land', a.tissue.land, b.tissue.land),
        ('stretch', a.strain.stretch, b.strain.stretch), ('zs', a.strain.zs, b.strain.zs),
        ('zw', a.strain.zw, b.strain.zw),
        *((f'velocity_{i}_cm_s', u, v) for i, (u, v) in enumerate(zip(a.fsi.velocity, b.fsi.velocity)))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', required=True)
    p.add_argument('--duration-ms', type=float, default=1.)
    p.add_argument('--output', default='results/active_tangent_benchmark.json')
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error('actual CUDA device required')
    base, initial, _ = checkpoint.load(args.resume)
    from afsi_torch.ep.protocol import ticks
    target = initial.tick+ticks(args.duration_ms/1000, base.config.protocol.dt_ep_s)
    if target <= initial.tick:
        p.error('positive duration required')
    measurements, finals = [], []
    for legacy in (True, False):
        label = 'legacy-autodiff' if legacy else 'analytic-compiled'
        with legacy_tangent(legacy):
            driver = AFSEIStepper(base.mesh, base.config)
            driver.profile = True
            driver.buffer.stage(initial.force_load)
            state, rows = initial, []
            while state.tick < target:
                torch.cuda.synchronize(); start = perf_counter()
                state, info = driver.step(state, target_tick=target)
                torch.cuda.synchronize()
                assert info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']
                if info['rejected_attempts']:
                    raise RuntimeError('benchmark requires identical accepted time grids without retries')
                rows.append(dict(step_seconds=perf_counter()-start, **info['timings'],
                    **{name: info['nonlinear'][name] for name in ('fluid_solves', 'pressure_cycles',
                        'tangent_assemblies', 'gmres_iterations', 'iterations')}))
                print(f'{label}: {len(rows)} accepted steps, {state.fsi.time*1000:.3f} ms', flush=True)
            warm = rows[1:]
            if not warm:
                raise ValueError('duration must include at least two accepted steps')
            measurements.append(dict(tangent=label, steps=len(rows), first_step_seconds=rows[0]['step_seconds'],
                mean_after_first={key: sum(r[key] for r in warm)/len(warm) for key in warm[0]}, rows=rows))
            finals.append(state)
    comparison = {}
    a, b = finals
    assert a.tick == b.tick and a.fsi.step == b.fsi.step
    for name, left, right in physical_pairs(a, b):
        difference = (left-right).norm()/left.norm().clamp_min(1e-30)
        comparison[name] = dict(max_absolute_difference=(left-right).abs().max().item(),
            relative_l2_difference=difference.item())
        if difference > 2e-7:
            raise AssertionError(f'physical comparison failed: {name}: {difference.item()}')
    result = dict(passed=True, source_checkpoint=str(Path(args.resume)),
        duration_ms=args.duration_ms, fluid_shape=base.config.fsi.fluid.shape,
        mechanical_dt_s=base.config.fsi.time.dt, gpu=torch.cuda.get_device_name(),
        physical_model_unchanged=True, windkessel_ports_connected=False,
        measurements=measurements, final_comparison=comparison,
        timing_scope='synchronized driver steps, first step excluded from means; no output IO; scratch states only')
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
