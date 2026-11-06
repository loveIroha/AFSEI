"""Read-only operator checks for a saved, unaccepted physical-cut iterate.

Stages one EP interval using the existing driver, intercepts Newton, and
never publishes a new physical state or overwrites an experiment checkpoint.
All tolerance/control substitutions are confined to this diagnostic process.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep import mac_balanced_dynamic as dynamic
from afsi_torch.nonlinear import NonlinearFailure


class DiagnosticFinished(Exception):
    pass


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('iterate')
    parser.add_argument('output', type=Path)
    parser.add_argument('--ticks', type=int, default=1)
    args = parser.parse_args()
    if args.ticks not in (1, 2):
        raise ValueError('only one saved failing interval may be staged')
    torch.set_num_threads(4)
    driver, state, _ = restart.load(args.checkpoint, cardiac=True)
    with np.load(args.iterate, allow_pickle=False) as archive:
        if int(archive['last_accepted_tick']) != state.tick:
            raise ValueError('iterate belongs to a different accepted checkpoint')
        saved = torch.as_tensor(archive['unknown'], device=state.fsi.x.device).clone()
    report = dict(checkpoint=args.checkpoint, iterate=args.iterate,
                  accepted_time_ms=state.fsi.time*1000, ticks=args.ticks,
                  physical_state_published=False)
    original_newton = dynamic.newton

    def publish():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def diagnose(residual, initial, **kwargs):
        problem = residual.__self__
        flow = problem.driver.flow
        solver = flow.pressure_solver
        options = solver.cg_options
        storage = problem.cut_storage
        n = problem.mechanical_size
        if saved.shape != initial.shape:
            raise ValueError('saved iterate has a different constraint layout')
        pressure_initial = state.fsi.pressure.clone()
        report.update(dt_s=flow.dt, control_mode=problem.control.mode,
                      cut_plane_count=len(flow.planes), cap_rows=problem.rows.count,
                      nonlinear_tolerance=kwargs['options'].atol,
                      pressure_rtol=options.rtol, pressure_atol=options.atol,
                      preconditioner_setup_seconds=problem.setup_seconds)

        def evaluate(z, *, cold=True):
            problem.last_z = None
            if cold:
                problem.pressure_initial = pressure_initial.clone()
            return residual(z).clone()

        def blocks(r):
            lower = r[n:]
            return dict(total=float(r.norm()), mechanical=float(r[:n].norm()),
                        closed_cap=float(lower[:-2].norm()),
                        basal_sheet=float(lower[-2:].norm()),
                        mechanical_max=float(r[:n].abs().max()),
                        closed_cap_max=float(lower[:-2].abs().max()),
                        largest_closed_row=int(lower[:-2].abs().argmax()))

        try:
            report['repeatability'] = []
            for factor, cold in ((1., True), (1., False), (.1, True), (.01, True)):
                solver.cg_options = replace(options, rtol=options.rtol*factor,
                                            atol=options.atol*factor)
                problem.pressure_initial = pressure_initial.clone()
                values = [evaluate(saved, cold=cold) for _ in range(3)]
                report['repeatability'].append(dict(
                    factor=factor, cold=cold, blocks=[blocks(v) for v in values],
                    maximum_repeat_difference=max(float((v-values[0]).norm()) for v in values),
                    pressure_solve=problem.last_data[6]['pressure']))
            publish()

            # With both cuts sealed, circuit parameters must have no direct
            # influence on the fluid-mechanical residual at a fixed state.
            solver.cg_options = options
            baseline = evaluate(saved)
            control = problem.control
            try:
                wk = control.windkessel
                problem.control = replace(control, arterial_pc_mmhg=control.arterial_pc_mmhg+100.,
                    windkessel=replace(wk, rc_mmhg_s_per_ml=3*wk.rc_mmhg_s_per_ml,
                        rp_mmhg_s_per_ml=.5*wk.rp_mmhg_s_per_ml,
                        compliance_ml_per_mmhg=2*wk.compliance_ml_per_mmhg))
                report['closed_circuit_parameter_probe'] = dict(
                    residual_vector_difference=float((evaluate(saved)-baseline).norm()))
            finally:
                problem.control = control

            # Independent CPU clipping reference checks the new GPU volume
            # gradient and its derivative at this actual stalled configuration.
            x = problem.unpack(saved)[1]
            dy = torch.zeros_like(x)
            ids = torch.arange(storage.nodes.numel()*3, device=x.device, dtype=x.dtype)
            dy[storage.nodes] = torch.sin(ids).reshape(-1, 3)*.01
            g_gpu = storage.discrete(problem.state.x, x).clone()
            j_gpu = storage.derivative(problem.state.x, x)(dy).clone()
            cache = storage.cached
            info = dict(gradient_norm=float(g_gpu.norm()), derivative_action_norm=float(j_gpu.norm()),
                        topology_builds=storage.topology_builds, fallbacks=storage.fallback_count,
                        secant_corrected=bool(cache.get('corrected', False)))
            reference = storage.reference
            g_cpu = reference.discrete(problem.state.x, x)
            j_cpu = reference.derivative(problem.state.x, x)(dy)
            info.update(volume_difference_ml=storage.volume(x)-reference.volume(x),
                        gradient_difference=float((g_gpu-g_cpu).norm()),
                        derivative_action_difference=float((j_gpu-j_cpu).norm()))
            try:
                problem.cut_storage = reference
                report['cpu_geometry_residual'] = dict(blocks=blocks(evaluate(saved)),
                    vector_difference=float((evaluate(saved)-baseline).norm()))
            finally:
                problem.cut_storage = storage
            report['tract_geometry_reference'] = info
            publish()

            # Test the complete action using a fresh material tangent, with
            # pressure solves tighter than the production diagnostic baseline.
            solver.cg_options = replace(options, rtol=options.rtol*.01, atol=options.atol*.01)
            evaluate(saved)
            action = problem.linearization(saved, reuse_tangent=False)
            directions = []
            v = saved.new_zeros(saved.shape)
            v[:n] = torch.sin(torch.arange(n, device=v.device, dtype=v.dtype))/n**.5
            directions.append(('mechanical', v))
            v = saved.new_zeros(saved.shape)
            v[n:] = torch.cos(torch.arange(len(v)-n, device=v.device, dtype=v.dtype))*1333.22368421/problem.cap_scale
            directions.append(('cap_pressure', v))
            report['jacobian_fd'] = []
            for name, direction in directions:
                analytic = action(direction).clone()
                for epsilon in (1e-3, 1e-4, 1e-5):
                    fd = (evaluate(saved+epsilon*direction)-evaluate(saved-epsilon*direction))/(2*epsilon)
                    error = fd-analytic
                    report['jacobian_fd'].append(dict(direction=name, epsilon=epsilon,
                        action_norm=float(analytic.norm()), relative_error=float(error.norm()/analytic.norm()),
                        mechanical_error=float(error[:n].norm()), cap_error=float(error[n:].norm())))
            publish()

            # Attempt to remove the residual floor without relaxing the outer
            # acceptance criterion. No endpoint/state is ever published here.
            report['reroot'] = []
            for factor in (1., .1, .01):
                solver.cg_options = replace(options, rtol=options.rtol*factor, atol=options.atol*factor)
                problem.last_z = None
                problem.pressure_initial = pressure_initial.clone()
                started = perf_counter()
                trial_kwargs = dict(kwargs, linearization_factory=problem.linearization)
                try:
                    result = original_newton(residual, saved.clone(), **trial_kwargs)
                    ok, reason = True, None
                except NonlinearFailure as exc:
                    result, ok, reason = exc.result, False, str(exc)
                checked = evaluate(result.x)
                y, end_x, _ = problem.unpack(result.x)
                u = problem.last_data[4]
                physical_cap = problem.rows.flow(u)-problem.rows.material_flow(y/flow.dt)
                closed_mass = (problem.load.volume(end_x)-problem.load.volume(problem.state.x)
                    +storage.volume(end_x)-storage.volume(problem.state.x))/flow.dt
                report['reroot'].append(dict(factor=factor, converged=ok, reason=reason,
                    seconds=perf_counter()-started, reported_residual=result.residual_norm,
                    cold_check=blocks(checked), tolerance=result.tolerance, history=result.history,
                    closed_cap_flux_norm_ml_s=float(physical_cap[:-2].norm()),
                    total_volume_rate_ml_s=float(closed_mass)))
                publish()
        finally:
            solver.cg_options = options
            problem.cut_storage = storage
        raise DiagnosticFinished()

    dynamic.newton = diagnose
    try:
        driver.step(state, target_tick=state.tick+args.ticks, diagnostics=True)
    except DiagnosticFinished:
        report['completed'] = True
    finally:
        dynamic.newton = original_newton
        publish()
    print(json.dumps({k:v for k,v in report.items() if k not in ('reroot', 'repeatability')}, indent=2))


if __name__ == '__main__':
    main()
