"""Physical loading, no double-counted PK1, consistent tangent, and GPU staging."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from afsi_torch.solids.circular_tube import CircularTubeGeometry, generate_circular_tube
from afsi_torch.ep.circular_tube import CircularTubeSolid
from afsi_torch.ep.planar_tube import land_resting_equilibrium
from afsi_torch.ep.tissue import initial_state
from afsi_torch.ep.land import land_tension
from afsi_torch.simulation.circular_tube_config import resolve_config


def setup(device):
    g = CircularTubeGeometry(length_cm=2., inner_radius_cm=.3, wall_thickness_cm=.1,
        axial_elements=8, circumferential_elements=16, thickness_elements=1)
    mesh = generate_circular_tube(g, device=device)
    kw = dict(young_modulus_dyn_cm2=3e5, poisson_ratio=.3, end_spring_dyn_cm3=0.,
        end_taper_cm=.15, dt_s=5e-5, stretch_feedback='isometric-driver')
    active = CircularTubeSolid(mesh, g, **kw, tension_scale=.02, mechanism='paired-transverse-land-traction')
    passive = CircularTubeSolid(mesh, g, **kw, tension_scale=0.)
    state = initial_state(mesh.X[:, 0]); one = torch.ones_like(state.voltage_v); zero = torch.zeros_like(one)
    state = replace(state, land=land_resting_equilibrium(state.gpb[:, 37]*1000, one))
    ref = land_tension(one, one, state.land[:, 0], state.land[:, 1], zero, zero, 5e-5)
    for m in (active, passive):
        m.set_reference(ref); m.stage(mesh.X, state, zero, zero)
    return g, mesh, active, passive, state, zero


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_load_direction_work_units_and_no_double_counting(device):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA required')
    g, mesh, m, passive, state, zero = setup(device)
    assert m.force(mesh.X, 0.).abs().max() < 1e-8
    land = state.land.clone(); land[:, 0] = .05
    m.stage(mesh.X, replace(state, land=land), zero, zero)
    load = m.paired_load.boundary.force(mesh.X, mesh.X.new_tensor(0.))
    # Exact reference-area units for uniform increment: force_y per face = -p*n_y*A.
    expected = -m.paired_load.area*m.paired_load.normal_y*m.paired_load.pressure.mean(1)
    torch.testing.assert_close(m.paired_load.local_force[..., 1].sum(1), expected)
    assert load[:, [0, 2]].abs().max() == 0
    assert ((mesh.X[:, 1]-g.center_xy_cm[1])*load[:, 1]).sum() < 0
    assert load.sum(0).abs().max() < 1e-9
    assert torch.linalg.cross(mesh.X, load).sum(0).abs().max() < 1e-9
    x = mesh.X.clone(); x[:, 1] = g.center_xy_cm[1]+.92*(x[:, 1]-g.center_xy_cm[1])
    torch.testing.assert_close(m.force(x, 0.)-passive.force(x, 0.), load, atol=2e-9, rtol=1e-9)
    torch.testing.assert_close(m.activation_stretch(m.cell_stretch(x)), torch.ones(len(mesh.cells), device=device, dtype=x.dtype))
    # Consistent total tangent includes passive mechanics only for this dead load.
    direction = torch.sin(torch.arange(x.numel(), device=device, dtype=x.dtype)).reshape_as(x)
    eps = 1e-7
    tangent = m.tangent_factory().assemble(x, 0.)
    actual = torch.mv(tangent, direction.flatten()).reshape_as(x)
    finite = (m.force(x+eps*direction, 0.)-m.force(x-eps*direction, 0.))/(2*eps)
    torch.testing.assert_close(actual, finite, rtol=3e-5, atol=.02)
    if device.startswith('cuda'):
        execution = m.execution_factory()
        torch.testing.assert_close(execution.force(x, 0.), m.force(x, 0.), atol=1e-8, rtol=1e-9)
        # New states must update the persistent load seen by the compiled kernel.
        land[:, 0] = .03
        m.stage(x, replace(state, land=land), zero, zero)
        torch.testing.assert_close(execution.force(x, 0.), m.force(x, 0.), atol=1e-8, rtol=1e-9)
    m.validate(x)
    closed = mesh.X.clone(); closed[:, 1] = g.center_xy_cm[1]+.1*(closed[:, 1]-g.center_xy_cm[1])
    with pytest.raises(ValueError, match='invalid P1'):
        m.validate(closed)


def test_config_requires_explicit_isometric_single_layer():
    path = Path(__file__).resolve().parents[1]/'demo/active_tube_3d/configs/tube_paired_y_cg_euler_gpu.json'
    c = json.loads(path.read_text()); resolve_config(c)
    c['active']['stretch_feedback'] = 'circumferential'
    with pytest.raises(ValueError, match='paired actuator'):
        resolve_config(c)
