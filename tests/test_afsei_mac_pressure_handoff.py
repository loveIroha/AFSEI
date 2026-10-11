"""Actual GPU traction-free Jacobian, paired work and failed-handoff isolation."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import physical_arrays
from afsi_torch.ep.mac_pressure_handoff import (
    TractionFreeCapProblem, traction_free_step, validate_static_handoff, HandoffTargets)
from afsi_torch.ep.mac_preload import solve, initialize
from afsi_torch.two_way.preload import PreloadConfig
from afsi_torch.ep import checkpoint


def test_source_free_gpu_jacobian_and_cap_pressure_columns(mac_case):
    d = precise_driver(mac_case); d.closed_cap = True
    s = d.initialize(x=d.mesh.X*1.002)
    e = d._fsi_stepper(d.maximum_ticks)
    p = TractionFreeCapProblem(e, s.fsi, e.transfer.prepare(s.fsi.x), e.flow.advect(s.fsi.velocity))
    z = s.fsi.x.new_zeros(s.fsi.x.numel()+p.extra_unknowns)
    z[-1] = .003
    v = torch.cos(torch.arange(len(z), device=z.device, dtype=z.dtype)); v /= v.norm()
    action = p.linearization(z); eps = 2e-6
    fd = (p.residual(z+eps*v)-p.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v), fd, rtol=4e-4, atol=3e-6)
    assert len(z) == s.fsi.x.numel()+p.rows.count
    assert d.config.prescribed_endocardial_pressure_mmhg == 0.
    loaded = precise_driver(mac_case)
    loaded.config = replace(loaded.config, prescribed_endocardial_pressure_mmhg=8.)
    el = loaded._fsi_stepper(loaded.maximum_ticks)
    with pytest.raises(ValueError, match='zero prescribed'):
        TractionFreeCapProblem(el, s.fsi, el.transfer.prepare(s.fsi.x), el.flow.advect(s.fsi.velocity))


def test_source_free_true_residual_and_cap_work(mac_case):
    d = precise_driver(mac_case); d.closed_cap = True
    s = d.initialize(x=d.mesh.X*1.002)
    copied = [v.clone() for v in physical_arrays(s)]
    trial, r = traction_free_step(d._fsi_stepper(d.maximum_ticks), s.fsi)
    assert r['nonlinear_residual_norm'] <= r['nonlinear_tolerance']
    assert r['closed_cap_flux_norm_ml_s'] < 1e-5
    assert r['cap_work_adjoint_relative_error'] < 1e-7
    assert r['cap_closed_power_relative_error'] < 1e-7
    assert r['paired_cap_reaction_applied']
    assert r['wall_pressure_unknowns'] == 0
    assert not r['global_volume_constraint_added'] and not r['full_fsi_outer_iterations_added']
    assert trial.time > s.fsi.time and trial.x is not s.fsi.x
    for a, b in zip(physical_arrays(s), copied): assert torch.equal(a, b)


def test_failed_preload_handoff_preserves_checkpoint_land_and_geometry(mac_case, tmp_path):
    d = precise_driver(mac_case)
    result, history = solve(d, PreloadConfig(nominal_steps=4, minimum_step_mmhg=.005))
    original, s, _ = initialize(d, result, history)
    arrays = physical_arrays(s)
    copied = [v.clone() for v in arrays]
    path = tmp_path/'before.npz'; checkpoint.save(path, original, s)
    contents = path.read_bytes()
    config = original.config
    r = validate_static_handoff(original, s, diagnostic_guard_cells=1)
    assert not r['static_handoff_validated']
    assert 'static_displacement' in r['failed_targets'] and 'static_velocity' in r['failed_targets']
    assert r['baseline_static_displacement_l2_cm'] < 1e-9
    assert r['baseline_static_velocity_max_cm_s'] < 1e-6
    assert not r['candidate_published'] and not r['production_handoff_enabled']
    assert not r['land_history_advanced'] and not r['source_state_advanced']
    assert r['rows'][0]['retained_prescribed_wall_traction_mmhg'] == 0.
    for a, b in zip(arrays, copied): assert torch.equal(a, b)
    assert original.config is config and path.read_bytes() == contents
    checkpoint.save(tmp_path/'after.npz', original, s)
    _, restored, _ = checkpoint.load(tmp_path/'after.npz')
    for a, b in zip(arrays, physical_arrays(restored)): assert torch.equal(a, b)
    active = replace(s, force_load=replace(s.force_load, use_land=True))
    with pytest.raises(ValueError, match='unadvanced'):
        validate_static_handoff(original, active)
    with pytest.raises(ValueError): HandoffTargets(velocity_max_cm_s=0.)
