"""Planar simplex invariants and two-open-end MAC boundary regression."""
import pytest
import torch
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.mac2d.grid import ChannelGrid, divergence, negative_laplacian
from afsi_torch.mac2d.multigrid import ChannelMultigrid
from afsi_torch.mac2d.pressure_channel import PressureChannelFlow, pressure_gradient, open_laplacian


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return request.param


def test_nodal_triangles_and_insulating_diffusion(device):
    X = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=torch.float64, device=device)
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]], device=device)
    f, s = X.new_tensor([1., 0.]).expand_as(X), X.new_tensor([0., 1.]).expand_as(X)
    ep = Monodomain(X, cells, f, s, numerics=MonodomainNumerics(backend='torch_pcg_jacobi_light'))
    torch.testing.assert_close(ep.mass_diagonal, X.new_tensor([1/3, 1/6, 1/3, 1/6]))
    torch.testing.assert_close(ep.stiffness.to_dense().sum(1), X.new_zeros(4), atol=1e-15, rtol=0)
    torch.testing.assert_close(ep.stiffness.to_dense(), ep.stiffness.to_dense().T)
    # The nodal rule must still integrate the constant-tensor diffusion exactly.
    consistent = Monodomain(X, cells, f, s, numerics=MonodomainNumerics(
        discretization='consistent_p1', backend='torch_pcg_jacobi_light'))
    torch.testing.assert_close(ep.stiffness.to_dense(), consistent.stiffness.to_dense())
    v = X.new_tensor([1., 0., 0., 0.])
    out, _ = ep.step(v, torch.zeros_like(v), torch.zeros_like(v), 1e-3)
    torch.testing.assert_close(ep.mass_diagonal@v, ep.mass_diagonal@out, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(ep.project_cell_field(X.new_ones(2)), X.new_ones(4))
    ep.close(); consistent.close()


def test_two_end_pressure_gradient_and_multigrid(device):
    grid = ChannelGrid((32, 16), (2., 1.))
    flow = PressureChannelFlow(grid, dt=1e-4, rho=1., mu=.04, inlet_pressure=2., outlet_pressure=.5,
        device=device, fused=False)
    zero = grid.zeros(device=device)
    result = flow.project(zero)
    coordinates = grid.coordinates(device=device)
    expected = 2.-.75*coordinates[..., 0]
    torch.testing.assert_close(result.pressure, expected, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(result.velocity[0], torch.full_like(zero[0], 7.5e-5), rtol=1e-7, atol=1e-12)
    assert divergence(result.velocity, grid.spacing).abs().max() < 1e-10
    p = torch.sin(coordinates[..., 0])*torch.cos(coordinates[..., 1])
    torch.testing.assert_close(negative_laplacian(p, grid.spacing, True),
                               -divergence(pressure_gradient(p, grid.spacing), grid.spacing))
    # Discrete integration by parts includes half-sized end-face volumes.
    u = tuple(torch.randn_like(a) for a in zero)
    u[1][:, 0] = 0; u[1][:, -1] = 0
    dg = grid.volume*(p*divergence(u, grid.spacing)).sum()
    gu = sum((a*g*flow.power_weights(c, a)).sum() for c, (a, g) in enumerate(zip(u, pressure_gradient(p, grid.spacing))))
    torch.testing.assert_close(dg, -gu, atol=1e-12, rtol=1e-12)


def test_pressure_graph_matches_operator(device):
    if device != 'cuda':
        pytest.skip('graph path requires CUDA')
    grid = ChannelGrid((32, 16), (2., 1.))
    solver = ChannelMultigrid(grid, device=device, backend='graph', fused=True, pressure_inlet=True)
    p = torch.randn(grid.shape, device=device, dtype=torch.float64)
    rhs = negative_laplacian(p, grid.spacing, True)
    result, info = solver.solve(rhs)
    assert torch.linalg.vector_norm(rhs-negative_laplacian(result, grid.spacing, True)) <= info['tolerance']
    torch.testing.assert_close(result, p, rtol=1e-7, atol=1e-7)
