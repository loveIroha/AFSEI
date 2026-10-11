"""Actual GPU work pairing, rigid consistency, tangent and static preload."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from afsi_torch.ep.mac_pressure_mode import PressureModeTransfer
from afsi_torch.ep.mac_balanced_preload import (
    EndocardialAndCapPressure, BalancedCapProblem, solve, validate, save_result, read_result,
    reuse_validated_loading, save_loading_result)
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_pressure_handoff import TractionFreeCapProblem
from afsi_torch.ep.mac_transport import closed_volume_covectors
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.two_way.preload import PreloadConfig
from afsi_torch.mac.grid import divergence
from afsi_torch.mac.open_boundary import face_weights, gradient
from afsi_torch.ep.mac_pressure_compatibility import compatible_be_density


def test_gpu_cap_follower_action_and_source_force_remain_separate(mac_case):
    d = precise_driver(mac_case)
    x = d.mesh.X*1.002; load = EndocardialAndCapPressure(d.model)
    endo = MACEndocardialPressure(d.model)
    v = torch.sin(x+1.)
    eps = 1e-6
    fd = (load.force(x+eps*v, 8.)-load.force(x-eps*v, 8.))/(2*eps)
    torch.testing.assert_close(load.action(x, 8., v), fd, rtol=1e-7, atol=1e-5)
    directions = torch.stack((v, torch.cos(x)), -1)
    torch.testing.assert_close(load.action(x, 8., directions),
        torch.stack(tuple(load.action(x, 8., z) for z in directions.unbind(-1)), -1))
    torch.testing.assert_close(load.force(x, 8.)-endo.force(x, 8.),
                              8*C*closed_volume_covectors(x, load.faces, load.rim)[1])


def test_gpu_pressure_mode_work_rigid_fields_and_closed_transport(mac_case):
    d = precise_driver(mac_case); d.closed_cap = True
    s = d.initialize(x=d.mesh.X*1.002); e = d._fsi_stepper(d.maximum_ticks)
    e.transfer = PressureModeTransfer(e, level=2)
    stencil = e.transfer.prepare(s.fsi.x); rows = stencil.rows
    f = torch.sin(s.fsi.x*3.)*1000.
    u = tuple(torch.sin(e.grid.coordinates(c, device=f.device)[..., c]+.2) for c in range(3))
    density, _ = e.transfer.spread(f, stencil)
    nodal, _ = e.transfer.interpolate(u, stencil)
    fluid_power = sum((a*b*face_weights(e.grid, c, a)).sum()
                      for c, (a, b) in enumerate(zip(density, u)))
    torch.testing.assert_close(fluid_power, (f*nodal).sum(), rtol=1e-9, atol=1e-6)
    actual, _ = e.transfer.spread(stencil.mode, stencil)
    lift = rows.lift(torch.full_like(rows.area, C))
    for a, b, g in zip(actual, lift, gradient(stencil.force_potential, e.grid.spacing)):
        torch.testing.assert_close(a-b, g, rtol=1e-9, atol=1e-7)
    identity = ((stencil.mode*nodal).sum()-C*rows.flow(u).sum()
                +e.grid.volume*(stencil.force_potential*divergence(u, e.grid.spacing)).sum())
    assert abs(identity.item()) < 1e-6
    shift = f.new_tensor([.3, -.2, .4])
    uniform = tuple(torch.full_like(z, shift[c]) for c, z in enumerate(e.grid.zeros(device=f.device)))
    translated, _ = e.transfer.interpolate(uniform, stencil)
    torch.testing.assert_close(translated, shift.expand_as(f), rtol=1e-9, atol=1e-9)
    omega = f.new_tensor([.02, -.03, .04]); center = s.fsi.x.mean(0)
    rigid = tuple(torch.linalg.cross(omega.expand_as(e.grid.coordinates(c, device=f.device)),
        e.grid.coordinates(c, device=f.device)-center)[..., c] for c in range(3))
    rotated, _ = e.transfer.interpolate(rigid, stencil)
    torch.testing.assert_close(rotated, torch.linalg.cross(omega.expand_as(f), s.fsi.x-center),
                               rtol=1e-8, atol=1e-9)
    assert stencil.report['correction_net_force_norm_dyn'] < 1e-6
    assert stencil.report['correction_torque_norm_dyn_cm'] < 1e-6
    with pytest.raises(ValueError): PressureModeTransfer(e)


def test_gpu_corrected_joint_jacobian(mac_case):
    d = precise_driver(mac_case); d.closed_cap = True
    s = d.initialize(x=d.mesh.X*1.002); e = d._fsi_stepper(d.maximum_ticks)
    e.transfer = PressureModeTransfer(e, level=1)
    stencil = e.transfer.prepare(s.fsi.x)
    problem = BalancedCapProblem(e, s.fsi, stencil, e.flow.advect(s.fsi.velocity))
    z = s.fsi.x.new_zeros(s.fsi.x.numel()+problem.extra_unknowns); z[-1] = .003
    v = torch.cos(torch.arange(len(z), device=z.device, dtype=z.dtype)); v /= v.norm()
    action = problem.linearization(z); eps = 2e-6
    fd = (problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v), fd, rtol=5e-4, atol=4e-6)


def test_gpu_exact_pressure_superposition_agrees_with_full_original_be_equations(mac_case):
    d = precise_driver(mac_case); d.closed_cap = True
    s = d.initialize(x=d.mesh.X*1.002); e = d._fsi_stepper(d.maximum_ticks)
    e.transfer = PressureModeTransfer(e, level=2)
    stencil = e.transfer.prepare(s.fsi.x)
    for a, b in zip(stencil.gradient, compatible_be_density(e.flow, stencil.potential)):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-9)
    f = .4*stencil.mode+torch.sin(s.fsi.x)*3.
    bp = C*(.4+.01*torch.sin(torch.arange(stencil.rows.count,
                      device=f.device, dtype=f.dtype)))
    complete, _ = e.transfer.relative_density(f, bp, stencil)
    remainder, amplitude, _ = e.transfer.relative_parts(f, bp, stencil)
    departure = tuple(.01*torch.sin(e.grid.coordinates(c, device=f.device)[..., c]) for c in range(3))
    direct_u, direct_p, _ = e.flow.advance(departure, complete)
    split_u, split_p, _ = e.flow.advance(departure, remainder)
    for a, b in zip(direct_u, split_u):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=2e-9)
    torch.testing.assert_close(direct_p, split_p+amplitude*stencil.potential, rtol=1e-8, atol=2e-6)


def test_new_preload_recomputes_land_history_and_solves_paired_reactions(mac_case, tmp_path):
    d = precise_driver(mac_case)
    cfg = PreloadConfig(nominal_steps=4, minimum_step_mmhg=.005,
                       force_atol_dyn=1e-5, force_rtol=1e-10)
    result, history = solve(d, cfg)
    assert history.steps == 4 and history.duration_s == .5
    assert len([r for r in result.history if r['physical_loading_endpoint']]) == 4
    assert history.zs.abs().max() > 0 and history.zw.abs().max() > 0
    prepared, state, bp, r = validate(d, result, history, steps=3, level=2, guard_cells=1)
    assert r['static_targets_passed'], dict(failed=r['failed_targets'],
        rows=[{k: row[k] for k in ('cap_total_power_erg_s', 'cap_closed_power_relative_error',
            'cap_work_adjoint_relative_error', 'cap_projected_flux_norm_ml_s',
            'nonlinear_residual_norm', 'nonlinear_tolerance', 'max_velocity_cm_s',
            'displacement_from_preload_l2_cm')} for row in r['trials']])
    assert r['paired_cap_material_reactions_solved'] and r['land_strain_history_recomputed']
    assert not r['ep_advanced'] and not r['circulation_advanced']
    assert not r['production_driver_enabled'] and not r['full_cardiac_cycle_implemented']
    assert prepared.config.prescribed_endocardial_pressure_mmhg == 0.
    assert state.tick == round(d.config.start_time_s/d.protocol.dt_ep_s)
    assert state.fsi.step == 0 and state.fsi.time == .5
    assert all(v.abs().max() == 0 for v in state.fsi.velocity)
    assert abs(r['initial_pressure_core']['mean_mmhg']-8.) < .05
    torch.testing.assert_close(state.strain.zs, history.zs)
    torch.testing.assert_close(state.strain.zw, history.zw)
    assert bp.min() == bp.max()
    path = tmp_path/'preload_result.npz'
    save_result(path, prepared, state, bp, r, result.history)
    metadata, data = read_result(path)
    assert metadata['report']['static_targets_passed']
    assert metadata['production_checkpoint'] is False
    assert (data['zs'] == history.zs.cpu().numpy()).all()
    reused, h = reuse_validated_loading(d, path)
    assert reused.force_residual_norm_dyn <= reused.force_tolerance_dyn
    torch.testing.assert_close(h.zs, history.zs)
    torch.testing.assert_close(h.zw, history.zw)
    import numpy as np
    loading = tmp_path/'loading.npz'
    save_loading_result(loading, d, result, history)
    reused_loading, loading_h = reuse_validated_loading(d, loading)
    torch.testing.assert_close(reused_loading.x, result.x)
    torch.testing.assert_close(loading_h.zw, history.zw)
    with pytest.raises(ValueError): read_result(loading)
    corrupt = dict(data); corrupt['zs'] = corrupt['zs'].copy(); corrupt['zs'][0] += .01
    with np.load(path, allow_pickle=False) as archive:
        raw_metadata = str(archive['metadata'])
    bad = tmp_path/'corrupt.npz'
    np.savez_compressed(bad, metadata=raw_metadata, **corrupt)
    with pytest.raises(ValueError, match='checksum'): read_result(bad)
    from afsi_torch.ep import checkpoint
    with pytest.raises(ValueError, match='invalid AFSEI checkpoint'):
        checkpoint.load(path)
    with pytest.raises(ValueError, match='validated'):
        save_result(tmp_path/'failed.npz', prepared, state, bp,
                    dict(r, static_targets_passed=False), result.history)
    assert not (tmp_path/'failed.npz').exists()
