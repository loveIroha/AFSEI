"""Local cap constraints, pressure projection, moving-cap work and linearity."""
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.dirichlet_mg import DirichletMultigrid
from afsi_torch.mac.spectral_dirichlet import SpectralDirichletPressure
from afsi_torch.mac.multigrid import MGOptions
from afsi_torch.mac.open_boundary import gradient
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.ep.mac_seal import prepare_local_cap, MACCapProjection
from afsi_torch.circulation.geometry import PortGeometry
from afsi_torch.fluid.solvers import SolverOptions


@pytest.fixture(params=[('cpu','multigrid'),('cuda','multigrid'),('cpu','spectral'),('cuda','spectral')])
def case(request):
    device,backend = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    X = torch.tensor([[0., 0., -2.], [-1., -1., 0.], [1., -1., 0.],
                      [1., 1., 0.], [-1., 1., 0.]], device=device, dtype=torch.float64)
    endo = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], device=X.device)
    grid = MACGrid((16,)*3, (8.,)*3, (-4.,)*3)
    ports = PortGeometry(mv_center_cm=(.5, 0.), av_center_cm=(-.5, 0.),
                         mv_radius_cm=.45, av_radius_cm=.4)
    cap = MACBasalPorts(X, endo, grid, ports=ports, subdivision_level=5)
    x = X.clone(); x[:, 2] += .08*x[:, 0]*x[:, 1]
    p = cap.prepare(x)
    rows = prepare_local_cap(p, patch_width_cells=2.)
    cls = SpectralDirichletPressure if backend=='spectral' else DirichletMultigrid
    mg = cls(grid, device=X.device, dtype=X.dtype,
                           options=MGOptions(rtol=1e-12, atol=1e-14, max_cycles=100))
    projector = MACCapProjection(rows, mg, options=SolverOptions(
        rtol=1e-10, atol=1e-12, max_iterations=100, check_every=1, recompute_every=10))
    return X, p, rows, projector


def test_partition_and_moving_reaction_adjoint(case):
    X, p, rows, projector = case
    assert rows.count > 1
    assert len(rows.selected) == int((p.cap.labels == 0).sum())
    torch.testing.assert_close(rows.area.sum(), p.cap.region_sum(
        torch.linalg.vector_norm(p.area_vectors, dim=-1))[0], rtol=1e-13, atol=1e-13)
    u = tuple(torch.sin(p.cap.grid.coordinates(c, device=X.device, dtype=X.dtype)[..., c]+.2)
              for c in range(3))
    w = torch.cos(X)*.03
    pressure = torch.arange(rows.count, device=X.device, dtype=X.dtype)+.5
    fluid = -sum((a*b*v).sum() for a, b, v in zip(u, rows.lift(pressure), projector.weights))
    solid = (rows.reaction(pressure)*w).sum()
    target = -(pressure*(rows.flow(u)-rows.material_flow(w))).sum()
    torch.testing.assert_close(fluid+solid, target, rtol=1e-12, atol=1e-13)


def test_local_projection_preserves_divergence_work_energy_and_input(case):
    X, p, rows, projector = case
    # Uniform flow is divergence-free, but moves through every cap patch.
    u = tuple(t+.4*(c+1) for c, t in enumerate(p.cap.grid.zeros(device=X.device, dtype=X.dtype)))
    old = tuple(t.clone() for t in u)
    w = .05*torch.cos(X)
    before = rows.diagnostics(u, w)
    result, _, pressure, reaction, report = projector.solve(u, w, dt=.001)
    after = rows.diagnostics(result, w)
    assert before['max_patch_average_normal_slip_cm_s'] > 1.
    assert after['max_patch_average_normal_slip_cm_s'] < 1e-8
    assert abs(after['net_leak_ml_s']) < 1e-9
    assert report['max_divergence_after_s_inv'] < 1e-9
    assert report['energy_balance_relative_error'] < 1e-10
    assert abs(report['cap_total_power_erg_s']) < 1e-7
    assert pressure.isfinite().all() and reaction.isfinite().all()
    for a, b in zip(u, old):
        assert torch.equal(a, b)
    # Patch constraints are weak, not falsely claimed pointwise no-slip.
    assert after['pointwise_normal_slip_max_cm_s'] > after['max_patch_average_normal_slip_cm_s']


def test_schur_symmetry_and_projection_linearity(case):
    X, p, rows, projector = case
    a = torch.sin(torch.arange(rows.count, device=X.device, dtype=X.dtype)+.3)
    b = torch.cos(torch.arange(rows.count, device=X.device, dtype=X.dtype)+.1)
    torch.testing.assert_close(a@projector.action(b), b@projector.action(a), rtol=1e-10, atol=1e-11)
    zero = p.cap.grid.zeros(device=X.device, dtype=X.dtype)
    u = tuple(t+.3*(c+1) for c, t in enumerate(zero))
    v = tuple(t-.2*(c+1) for c, t in enumerate(zero))
    w = .03*torch.sin(X); z = .02*torch.cos(X)
    pu = projector.solve(u, w, dt=.001)[0]
    pv = projector.solve(v, z, dt=.001)[0]
    both = projector.solve(tuple(a+b for a, b in zip(u, v)), w+z, dt=.001)[0]
    for a, b, c in zip(pu, pv, both):
        torch.testing.assert_close(a+b, c, rtol=1e-8, atol=1e-9)


def test_arbitrary_divergent_input_and_pressure_increment(case):
    X, p, rows, projector = case
    torch.manual_seed(714)
    u = tuple(torch.randn_like(t)*.2 for t in p.cap.grid.zeros(device=X.device, dtype=X.dtype))
    w = torch.zeros_like(X)
    result, dp, pressure, _, report = projector.solve(u, w, dt=.001)
    assert report['max_divergence_before_s_inv'] > 1.
    assert report['max_divergence_after_s_inv'] < 1e-9
    assert report['divergence_projection_loss_erg'] > 0.
    assert report['energy_balance_relative_error'] < 1e-10
    density = rows.lift(pressure)
    for old, new, f, g in zip(u, result, density, gradient(dp, p.cap.grid.spacing)):
        torch.testing.assert_close(new, old-.001*(f+g), rtol=1e-10, atol=1e-11)
    assert rows.diagnostics(result, w)['max_patch_average_normal_slip_cm_s'] < 1e-8


def test_invalid_controls_and_solver_failure(case):
    X, p, rows, projector = case
    with pytest.raises(ValueError):
        prepare_local_cap(p, patch_width_cells=1.)
    with pytest.raises(ValueError):
        projector.solve(p.cap.grid.zeros(device=X.device), torch.zeros_like(X), dt=0.)
    projector.options = SolverOptions(rtol=1e-13, atol=1e-14, max_iterations=1)
    u = tuple(t+.4 for t in p.cap.grid.zeros(device=X.device, dtype=X.dtype))
    with pytest.raises(RuntimeError):
        projector.solve(u, .03*torch.sin(X), dt=.001)
