"""Diagnostic cell masks, Dirichlet pressure reference, open-flow accounting."""
from dataclasses import replace
import pytest
import torch
from test_afsei_mac_mass_budget import case
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_rcr_coupled import driver_for
from test_afsei_driver import physical_arrays
from afsi_torch.ep.mac_pressure_audit import cavity_masks, pressure_statistics, audit_pressure_interface
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.ep.tissue import initial_state, TissueState
from afsi_torch.mac.open_boundary import gradient


def test_cavity_mask_matches_analytic_pyramid_and_excludes_band(case):
    x, grid, audit = case
    grid = replace(grid, shape=(32,)*3)  # Resolve at least one eroded interior cell.
    inside, core = cavity_masks(x, audit.faces, grid, guard_cells=1)
    p = grid.coordinates(device=x.device)
    width = (p[..., 2]+2)/2
    exact = ((p[..., 2] > -2) & (p[..., 2] < 0)
             & (p[..., 0].abs() < width) & (p[..., 1].abs() < width))
    assert torch.equal(inside, exact)
    assert core.any() and (core & ~inside).sum() == 0 and core.sum() < inside.sum()
    with pytest.raises(ValueError): cavity_masks(x, audit.faces, grid, guard_cells=-1)


def test_pressure_units_and_fixed_dirichlet_reference(case):
    x, grid, audit = case
    _, core = cavity_masks(x, audit.faces, grid, guard_cells=0)
    p = torch.full(grid.shape, 8.*C, device=x.device, dtype=x.dtype)
    r = pressure_statistics(p, core)
    assert r['mean_mmhg'] == pytest.approx(8.)
    assert r['standard_deviation_mmhg'] < 1e-14
    g = gradient(p, grid.spacing)
    # Constant pressure is not a gauge mode under this solver's zero box BC.
    for c, v in enumerate(g):
        assert v.abs().max().item() == pytest.approx(2*8.*C/grid.spacing[c])
        middle = [slice(None)]*3; middle[c] = slice(1, -1)
        assert torch.count_nonzero(v[tuple(middle)]) == 0
    with pytest.raises(ValueError): pressure_statistics(p, torch.zeros_like(core))


@pytest.mark.parametrize('opened', [False, True])
def test_offline_gpu_decomposition_preserves_history_and_excludes_open_port(mac_case, opened):
    d = driver_for(mac_case)
    tissue = initial_state(d.mesh.X[:, 0]); land = tissue.land.clone()
    land[:, 0] = .01; land[:, 1] = .01
    old = d.connect(d.initialize(tissue=TissueState(tissue.voltage_v, tissue.gpb, land)))
    if opened: old = replace(old, arterial_pc_mmhg=-100.)  # mathematical regression probe
    state, info = d.step(old)
    assert info['rcr']['outlet_open'] is opened
    arrays = physical_arrays(state)+[state.cap_pressure, state.cap_geometry]
    copies = [v.clone() for v in arrays]
    clocks = (state.tick, state.fsi.time, state.accepted_port_steps, state.arterial_pc_mmhg)
    r = audit_pressure_interface(d, state, levels=(0, 1), guard_cells=1)
    assert not r['simulation_state_advanced'] and not r['production_step_overhead_added']
    assert not r['pressure_identification_established']
    assert r['forcing_pressure_decomposition']['pressure_superposition_relative_error'] < 1e-7
    assert r['forcing_pressure_decomposition']['velocity_superposition_relative_error'] < 1e-7
    assert r['frozen_actual_relative_av_flux_ml_s'] == pytest.approx(
        info['rcr']['measured_relative_q_ml_s'], abs=1e-7)
    assert abs(r['finite_volume_plus_frozen_av_flux_ml_s']) < 1e-5
    cap = r['cap_current_geometry']
    wall = r['wall_levels'][-1]['current']
    allowed = cap[2]['relative_flux_ml_s'] if opened else 0.
    assert r['closed_surface_signed_slip_ml_s'] == pytest.approx(
        wall['relative_flux_ml_s']+sum(v['relative_flux_ml_s'] for v in cap)-allowed)
    closed = cap[:2] if opened else cap
    assert r['closed_surface_absolute_slip_ml_s'] == pytest.approx(
        wall['absolute_relative_flux_ml_s']+sum(v['absolute_relative_flux_ml_s'] for v in closed))
    assert r['open_av_throughflow_excluded_from_closed_surface_slip'] is opened
    assert clocks == (state.tick, state.fsi.time, state.accepted_port_steps, state.arterial_pc_mmhg)
    for a, b in zip(arrays, copies): assert torch.equal(a, b)
