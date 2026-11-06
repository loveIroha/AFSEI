"""CPU assembly checks only: no CUDA compilation or FSI time advancement."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import math
import numpy as np
import pytest
import torch
from afsi_torch.solids.circular_tube import (CircularTubeGeometry, generate_circular_tube,
                                             section_areas, lumen_volume)
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.tissue import initial_state
from afsi_torch.ep.planar_tube import land_resting_equilibrium
from afsi_torch.ep.land import land_tension
from afsi_torch.simulation.circular_tube_config import resolve_config, config_summary


def example(*, scale=1., spring=0.):
    c = CircularTubeGeometry(axial_elements=3, circumferential_elements=12, thickness_elements=1)
    mesh = generate_circular_tube(c)
    model = CircularTubeSolid(mesh, c, young_modulus_dyn_cm2=1e6, poisson_ratio=0.,
        end_spring_dyn_cm3=spring, tension_scale=scale, end_taper_cm=.2, dt_s=5e-5)
    state = initial_state(mesh.X[:, 0])
    one = torch.ones(len(mesh.X), dtype=torch.float64)
    zero = torch.zeros_like(one)
    state = replace(state, land=land_resting_equilibrium(state.gpb[:, 37]*1000, one))
    ref = land_tension(one, one, state.land[:, 0], state.land[:, 1], zero, zero, 5e-5)
    model.set_reference(ref); model.stage(mesh.X, state, zero, zero)
    return c, mesh, model, state, zero


def test_mesh_is_conforming_and_matches_analytic_polygon_volumes():
    c, mesh, model, _, _ = example()
    assert len(torch.unique(mesh.X, dim=0)) == len(mesh.X)
    xyz = mesh.X[mesh.cells]
    D = (xyz[:, 1:]-xyz[:, :1]).transpose(-1, -2)
    assert (torch.linalg.det(D) > 0).all()
    n = c.circumferential_elements
    polygon = .5*n*math.sin(2*math.pi/n)
    assert float(model.volumes.sum()) == pytest.approx(
        polygon*((c.inner_radius_cm+c.wall_thickness_cm)**2-c.inner_radius_cm**2)*c.length_cm)
    assert float(lumen_volume(mesh.X, mesh)) == pytest.approx(polygon*c.inner_radius_cm**2*c.length_cm)
    torch.testing.assert_close(section_areas(mesh.X, mesh),
        torch.full((c.axial_elements+1,), polygon*c.inner_radius_cm**2, dtype=mesh.X.dtype))
    assert len(mesh.inner_faces) == 2*c.axial_elements*n
    # A real seam would leave extra boundary triangles; check all face counts.
    faces = mesh.cells[:, [[1,2,3], [0,3,2], [0,1,3], [0,2,1]]].reshape(-1, 3).sort(-1).values
    _, counts = torch.unique(faces, dim=0, return_counts=True)
    assert ((counts == 1) | (counts == 2)).all()
    assert int((counts == 1).sum()) == 4*n*(c.axial_elements+c.thickness_elements)


def test_rest_is_balanced_and_active_stress_contracts_without_net_force_or_torque():
    c, mesh, model, state, zero = example()
    assert float(model.force(mesh.X, 0.).abs().max()) < 1e-8
    land = state.land.clone(); land[:, 0] = .05
    model.stage(mesh.X, replace(state, land=land), zero, zero)
    force = model.force(mesh.X, 0.)
    # Virtual work against a uniform radial expansion must be negative.
    radial = mesh.X.clone(); radial[:, :2] -= mesh.X.new_tensor(c.center_xy_cm); radial[:, 2] = 0
    assert float((force*radial).sum()) < 0
    assert float(force.sum(0).abs().max()) < 1e-7
    assert float(torch.linalg.cross(mesh.X, force).sum(0).abs().max()) < 1e-7


def test_force_is_objective_under_rigid_motion():
    c, mesh, model, state, zero = example()
    land = state.land.clone(); land[:, 0] = .03
    model.stage(mesh.X, replace(state, land=land), zero, zero)
    angle = .31
    R = mesh.X.new_tensor([[math.cos(angle), -math.sin(angle), 0.],
                          [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    x = mesh.X.clone(); x[:, :2] *= .98
    shift = mesh.X.new_tensor([.3, -.2, .1])
    torch.testing.assert_close(model.force(x@R.T+shift, .1), model.force(x, .1)@R.T, rtol=1e-10, atol=1e-7)


def test_tangent_includes_land_and_end_spring_derivatives():
    _, mesh, model, state, zero = example(spring=1e7)
    land = state.land.clone(); land[:, 0] = .02
    model.stage(mesh.X, replace(state, land=land), zero, zero)
    x = mesh.X.clone(); x[:, :2] *= .99
    # Avoid the clamp kink at stretch=1 in a centered finite difference.
    generator = torch.Generator().manual_seed(7)
    direction = torch.randn(x.shape, generator=generator, dtype=x.dtype)
    direction /= torch.linalg.vector_norm(direction)
    matrix = model.tangent_factory().assemble(x, .1)
    actual = torch.mv(matrix, direction.reshape(-1)).reshape_as(x)
    epsilon = 2e-7
    finite = (model.force(x+epsilon*direction, .1)-model.force(x-epsilon*direction, .1))/(2*epsilon)
    torch.testing.assert_close(actual, finite, rtol=2e-5, atol=.01)


def test_config_is_explicit_and_rejects_unsupported_physics():
    path = Path(__file__).resolve().parents[1]/'demo/active_tube_3d/configs/tube_gpu.json'
    c = json.loads(path.read_text())
    summary = config_summary(c)
    assert summary['simulation_started'] is False
    assert summary['porous_media'] is False and summary['planned_steps'] == 10000
    assert summary['spatial_coupling'] == 'griffith-luo-2017-unified'
    assert summary['time_scheme'] == 'be-be'
    assert summary['cnab_vs_bebe_performance_measured'] is False
    assert summary['solid_nodes'] == 6336 and summary['solid_tetrahedra'] == 24576
    geometry, fsi, _, _ = resolve_config(c)
    assert fsi.fluid.mu == 10. and c['solid']['poisson_ratio'] == 0.
    bad = json.loads(json.dumps(c)); bad['time']['ep_dt_s'] *= 2
    with pytest.raises(ValueError, match='one EP update'):
        resolve_config(bad)
    bad = json.loads(json.dumps(c)); bad['fluid']['porosity'] = .9
    with pytest.raises(ValueError, match='fluid requires exactly'):
        resolve_config(bad)
    bad = json.loads(json.dumps(c)); bad['geometry']['axial_start_cm'] = .01
    with pytest.raises(ValueError, match='two MAC spacings'):
        resolve_config(bad)
    bad = json.loads(json.dumps(c)); bad['fsi']['time_scheme'] = 'cn-ab2'
    with pytest.raises(ValueError, match='CN-AB2 cannot be selected'):
        resolve_config(bad)


def test_original_mac_quadrature_ib_uses_consistent_mass_and_adjoint_power():
    """Construct the actual driver, but do not advance any physical time step."""
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.mac.backward_euler import BackwardEulerFlow
    from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer
    from afsi_torch.mac.open_boundary import face_weights
    c, mesh, model, _, _ = example()
    path = Path(__file__).resolve().parents[1]/'demo/active_tube_3d/configs/tube_gpu.json'
    config = json.loads(path.read_text())
    config['geometry'] = asdict(c)
    config['fluid'].update(shape=[16,16,16], lengths_cm=[8.,8.,8.], origin_cm=[-2.,-2.,-2.])
    config['fsi']['execution'] = dict(execution_backend='torch', pressure_backend='torch', mass_backend='pcg')
    config['fsi']['flow']['helmholtz_backend'] = 'reference'
    config['fsi']['interaction_quadrature'] = dict(mode='adaptive', point_density=2.,
        max_order=12, max_points=12000000, rule_family='conical', transfer_backend='reference')
    _, options, _, _ = resolve_config(config)
    driver = BEIBStepper(model, options, 'cpu')
    assert isinstance(driver.flow, BackwardEulerFlow)
    transfer = driver.transfer
    assert isinstance(transfer, AdaptiveP1Transfer)
    mass = transfer.mass.to_dense()
    assert (mass-torch.diag(mass.diagonal())).abs().max() > 0  # no EP-style lumping in FSI
    stencil = transfer.prepare(mesh.X)
    generator = torch.Generator().manual_seed(9)
    b = torch.randn(mesh.X.shape, dtype=mesh.X.dtype, generator=generator)
    velocity = tuple(torch.randn(u.shape, dtype=u.dtype, generator=generator) for u in driver.grid.zeros())
    density, _ = transfer.spread(b, stencil)
    U, _ = transfer.interpolate(velocity, stencil)
    solid_power = (b*U).sum()
    fluid_power = sum((u*f*face_weights(driver.grid, axis, mesh.X)).sum()
                      for axis, (u, f) in enumerate(zip(velocity, density)))
    torch.testing.assert_close(solid_power, fluid_power, rtol=2e-9, atol=2e-9)
