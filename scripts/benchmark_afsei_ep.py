"""Short real-LV EP validation; separates quadrature changes from AMG gains."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from afsi_torch.ep import Monodomain, MonodomainNumerics, GPBLand, initial_state
from afsi_torch.ep.tissue import TissueState
from afsi_torch.ep.mapping import source_p1_fields
from afsi_torch.paper_lv import PaperLVConfig, imported_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', default='/mnt/large2/gjh/realistic_left_ventricle')
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available() or not 10 <= a.steps <= 400 or a.repeats < 1:
        p.error('actual CUDA, 10..400 EP steps and positive repeats required')
    mesh = imported_model(PaperLVConfig(source_dir=a.source_dir), 'cuda').mesh
    fiber, sheet, mask, _ = source_p1_fields(mesh)
    dt = 5e-5
    make = lambda discretization, backend: Monodomain(mesh.X, mesh.cells, fiber, sheet,
        numerics=MonodomainNumerics(discretization, backend))
    reference = make('consistent_p1', 'torch_pcg_jacobi')
    reference.prepare(dt)
    ode = GPBLand(dt)
    state = initial_state(mesh.X[:, 0])
    stretch = torch.ones_like(state.voltage_v); zero = torch.zeros_like(stretch)
    stimulus = zero.clone(); stimulus[mask] = 25.
    inputs = []
    # These are real GPB-generated ionic currents during the first 2 ms, not
    # random algebraic RHSs. Solver benchmarks replay identical frozen inputs.
    for step in range(a.steps):
        tissue, current = ode.advance_ode(state, stretch, zero, zero, .5+step*dt)
        inputs.append((state.voltage_v, current, stimulus))
        voltage, _ = reference.step(state.voltage_v, current, stimulus, dt)
        state = TissueState(voltage, tissue.gpb, tissue.land)
    baseline_state = state
    variants = []
    for discretization in ('consistent_p1', 'nodal_p1'):
        for backend in ('torch_pcg_jacobi', 'amgx_pcg_amg'):
            ep = make(discretization, backend)
            torch.cuda.synchronize(); started = perf_counter()
            ep.prepare(dt)
            torch.cuda.synchronize(); setup = perf_counter()-started
            for V, I, S in inputs[:5]:
                ep.step(V, I, S, dt)
            seconds, iterations, ratios = [], [], []
            max_error = 0.
            comparator = make(discretization, 'torch_pcg_jacobi')
            comparator.prepare(dt)
            expected = [comparator.step(V, I, S, dt)[0] for V, I, S in inputs]
            for repeat in range(a.repeats):
                for idx, (V, I, S) in enumerate(inputs):
                    torch.cuda.synchronize(); started = perf_counter()
                    result, info = ep.step(V, I, S, dt)
                    torch.cuda.synchronize(); seconds.append(perf_counter()-started)
                    iterations.append(info.iterations)
                    ratios.append(info.residual_norm/info.tolerance)
                    if repeat == 0:
                        max_error = max(max_error, (result-expected[idx]).abs().max().item())
            variants.append(dict(discretization=discretization, backend=backend,
                setup_seconds=setup, mean_solve_seconds=sum(seconds)/len(seconds),
                median_solve_seconds=median(seconds), mean_iterations=sum(iterations)/len(iterations),
                max_true_residual_to_tolerance=max(ratios),
                max_same_equation_voltage_difference_mv=max_error*1000,
                amgx_setup_count=ep._systems[dt][2].setup_count if backend=='amgx_pcg_amg' else 0))
            ep.close(); comparator.close()
    nodal = make('nodal_p1', 'amgx_pcg_amg'); nodal.prepare(dt)
    state = initial_state(mesh.X[:, 0]); max_ratio = 0.
    torch.cuda.synchronize(); started = perf_counter()
    for step in range(a.steps):
        state, _, info = ode.step(nodal, state, stretch, zero, zero, .5+step*dt, stimulus)
        max_ratio = max(max_ratio, info.residual_norm/info.tolerance)
    torch.cuda.synchronize(); combined = perf_counter()-started
    for value in (state.voltage_v, state.gpb, state.land):
        if not torch.isfinite(value).all():
            raise FloatingPointError('nonfinite nodal/AmgX GPB-Land trace')
    old_mean, new_mean = variants[0]['mean_solve_seconds'], variants[3]['mean_solve_seconds']
    report = dict(passed=True, gpu=torch.cuda.get_device_name(), node_count=len(mesh.X),
        cell_count=len(mesh.cells), ep_steps=a.steps, ep_dt_s=dt, ode_dt_ms=ode.dt_ode_ms,
        benchmark_scope='PDE solves replay identical frozen CG1/GPB inputs; 5 warmup solves per variant',
        variants=variants, pde_speedup_vs_old=old_mean/new_mean,
        nodal_gpb_land_seconds=combined,
        nodal_gpb_land_seconds_per_ep_step=combined/a.steps,
        nodal_trace_max_residual_to_tolerance=max_ratio,
        final_voltage_min_mv=state.voltage_v.min().item()*1000,
        final_voltage_max_mv=state.voltage_v.max().item()*1000,
        final_calcium_min_um=state.gpb[:,37].min().item()*1000,
        final_calcium_max_um=state.gpb[:,37].max().item()*1000,
        vertex_rule_vs_consistent_final_voltage_max_difference_mv=(
            state.voltage_v-baseline_state.voltage_v).abs().max().item()*1000,
        note='Different quadrature changes discretization; voltage difference is not an AMG solver error. '
             'A 2 ms trace does not establish full-cycle propagation accuracy.',
        full_cycle_run=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    nodal.close(); reference.close()


if __name__ == '__main__':
    main()
