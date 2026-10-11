"""Hydrostatic, work-adjoint and nonpublication checks on actual GPU MAC."""
from dataclasses import replace
import pytest
import torch
from test_afsei_mac_mass_budget import case
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import physical_arrays
from afsi_torch.ep.mac_pressure_compatibility import (surface_stencil, surface_trace,
    surface_pressure_density, compatible_be_density, audit_preload_loads)
from afsi_torch.ep.mac_pressure_audit import cavity_masks
from afsi_torch.ep.mac_preload import solve, initialize
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.two_way.preload import PreloadConfig
from afsi_torch.mac.backward_euler import BackwardEulerFlow
from afsi_torch.mac.multigrid import MGOptions
from afsi_torch.mac.open_boundary import face_weights


@pytest.mark.parametrize('kernel', ['peskin4', 'bs4-bs3'])
def test_boundary_work_and_total_force_are_not_double_weighted(case, kernel):
    x, grid, audit = case
    wall, cap = audit.triangles(x)
    triangles = torch.cat((wall, cap))
    density = surface_pressure_density(audit, triangles, 8., level=3, kernel=kernel)
    # Deterministic non-solenoidal field makes the pressure work nonzero.
    u = tuple(torch.sin(grid.coordinates(c, device=x.device)[..., c]+.3) for c in range(3))
    bary = audit.rule(3)
    points = torch.einsum('qa,fai->fqi', bary, triangles).reshape(-1, 3)
    area = (audit.area(triangles)/len(bary))[:, None].expand(-1, len(bary), -1).reshape(-1, 3)
    fluid = sum((v*f*face_weights(grid, c, f)).sum() for c, (v, f) in enumerate(zip(u, density)))
    boundary = -8*C*(surface_trace(grid, u, points, kernel=kernel)*area).sum()
    torch.testing.assert_close(fluid, boundary, rtol=1e-12, atol=1e-10)
    assert boundary.abs() > 100.
    net = torch.stack([(f*face_weights(grid, c, f)).sum() for c, f in enumerate(density)])
    assert net.norm() < 1e-9
    for ids, weights in surface_stencil(grid, points[:7], kernel=kernel):
        torch.testing.assert_close(weights.sum(-1), torch.ones(7, device=x.device, dtype=x.dtype),
                                   rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError): surface_stencil(grid, points*10, kernel=kernel)


def test_manufactured_hydrostatic_control_uses_load_not_a_gauge_shift(case):
    x, grid, audit = case
    # Verify the operator identity more accurately than the production pressure
    # solver's default tolerance; production settings are not changed.
    flow = BackwardEulerFlow(grid, dt=.0001, device=x.device,
                             pressure_options=MGOptions(rtol=1e-12, atol=1e-13))
    p = torch.zeros(grid.shape, device=x.device, dtype=x.dtype)
    p[5:11, 5:11, 5:11] = 8*C
    u, actual, _ = flow.advance(grid.zeros(device=x.device), compatible_be_density(flow, p))
    assert max(v.abs().max().item() for v in u) < 1e-8
    torch.testing.assert_close(actual, p, rtol=1e-9, atol=1e-6)
    assert actual[7, 7, 7].item()/C == pytest.approx(8., abs=1e-8)


def test_closed_pressure_kernel_comparison_requires_quadrature_convergence(case):
    x, grid, audit = case
    _, core = cavity_masks(x, audit.faces, grid, guard_cells=0)
    flow = BackwardEulerFlow(grid, dt=.0001, device=x.device)
    surface = torch.cat(audit.triangles(x))
    velocities = {}
    pressures = {}
    for kernel, levels in [('peskin4', (4,)), ('bs4-bs3', (3, 4, 5))]:
        for level in levels:
            f = surface_pressure_density(audit, surface, 8., level=level, kernel=kernel)
            u, p, _ = flow.advance(grid.zeros(device=x.device), f)
            velocities[kernel, level] = max(v.abs().max().item() for v in u)
            pressures[kernel, level] = p[core].mean().item()/C
    # The same closed geometry with the ordinary kernel retains circulating
    # velocity; the composite surface kernel converges as quadrature is refined.
    assert velocities['bs4-bs3', 5] < .01*velocities['peskin4', 4]
    assert velocities['bs4-bs3', 5] < .3*velocities['bs4-bs3', 3]
    assert abs(pressures['bs4-bs3', 5]-pressures['bs4-bs3', 4]) < 1e-5


def test_real_gpu_preload_audit_preserves_physical_arrays_and_production_kernel(mac_case):
    d = precise_driver(mac_case)
    result, history = solve(d, PreloadConfig(nominal_steps=4, minimum_step_mmhg=.005))
    original, state, _ = initialize(d, result, history)
    copied = [v.clone() for v in physical_arrays(state)]
    transfer_type = type(original._fsi_stepper(original.maximum_ticks).transfer)
    r = audit_preload_loads(original, state, levels=(0, 1), guard_cells=1)
    assert r['manufactured_control_passed']
    assert not r['production_handoff_enabled'] and not r['static_pressure_preload_implemented']
    assert not r['state_advanced'] and not r['land_history_advanced']
    assert not r['cap_fluid_load_has_paired_material_reaction']
    assert not r['isolated_loads_are_complete_fsi_equilibria']
    assert not r['production_transfer_kernel_changed']
    assert type(original._fsi_stepper(original.maximum_ticks).transfer) is transfer_type
    for a, b in zip(physical_arrays(state), copied): assert torch.equal(a, b)
    active = replace(state, force_load=replace(state.force_load, use_land=True))
    with pytest.raises(ValueError, match='unadvanced'): audit_preload_loads(original, active)
