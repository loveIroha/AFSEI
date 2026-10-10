"""MAC cap adjoints, orientation, material motion and closed-surface kinematics."""
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.open_boundary import face_weights
from afsi_torch.p1 import cavity_volume
from afsi_torch.ep.mac_ports import MACBasalPorts, prepare_flux_rows
from afsi_torch.circulation.geometry import PortGeometry


@pytest.fixture(params=['cpu', 'cuda'])
def case(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    X = torch.tensor([[0., 0., -2.], [-1., -1., 0.], [1., -1., 0.],
                      [1., 1., 0.], [-1., 1., 0.]], device=request.param, dtype=torch.float64)
    # Source endocardium faces point INTO blood; cap is outward from blood.
    endo = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], device=X.device)
    grid = MACGrid((16,)*3, (8.,)*3, (-4.,)*3)
    ports = PortGeometry(mv_center_cm=(.5, 0.), av_center_cm=(-.5, 0.),
                         mv_radius_cm=.45, av_radius_cm=.4)
    cap = MACBasalPorts(X, endo, grid, ports=ports, subdivision_level=5)
    return X, cap


def test_material_warp_area_and_closed_volume_derivative(case):
    X, cap = case
    x = X.clone(); x[:, 0] *= 1.2
    x[:, 2] += .06*x[:, 0]*x[:, 1]+.15
    p = cap.prepare(x)
    assert p.points[:, 2].max() > p.points[:, 2].min()+.1
    v = torch.sin(x)*.02
    nodes = x[cap.endo]; w = v[cap.endo]
    # Wall normal is opposite to the source's solid-outward normal.
    wall_area = -.5*torch.linalg.cross(nodes[:, 1]-nodes[:, 0], nodes[:, 2]-nodes[:, 0])
    geometric_rate = (wall_area*w.mean(1)).sum()+p.material_flow(v).sum()
    derivative = torch.func.jvp(lambda y: cavity_volume(y, cap.endo, cap.rim), (x,), (v,))[1]
    torch.testing.assert_close(geometric_rate, derivative, rtol=1e-12, atol=1e-13)
    assert cap.metadata['anatomical_valves_present'] is False
    assert cap.metadata['cap_sealed'] is False
    torch.testing.assert_close(cap.region_sum(p.area_vectors).sum(0), x.new_tensor([0., 0., 4.8]), rtol=1e-13, atol=1e-12)


def test_mac_affine_reproduction_and_region_partition(case):
    X, cap = case
    p = cap.prepare(X)
    M = X.new_tensor([[.13, .02, -.03], [-.07, .1, .02], [.08, -.03, .11]])
    offset = X.new_tensor([.2, -.1, .7])
    u = tuple((cap.grid.coordinates(c, device=X.device, dtype=X.dtype)@M.T+offset)[..., c] for c in range(3))
    expected = cap.region_sum(((p.points@M.T+offset)*p.area_vectors).sum(-1))
    torch.testing.assert_close(p.rows.flow(u), expected, rtol=2e-13, atol=1e-12)
    area = cap.region_sum(torch.linalg.vector_norm(p.area_vectors, dim=-1))
    torch.testing.assert_close(area.sum(), X.new_tensor(4.), rtol=1e-13, atol=1e-12)
    w = X@M.T+offset
    torch.testing.assert_close(p.relative_flow(u, w), X.new_zeros(3), rtol=0, atol=1e-12)


def test_fluid_and_relative_work_adjoint_on_warped_cap(case):
    X, cap = case
    x = X.clone(); x[:, 2] += .08*x[:, 0]*x[:, 1]
    p = cap.prepare(x)
    torch.manual_seed(72)
    u = tuple(torch.randn_like(t) for t in cap.grid.zeros(device=X.device, dtype=X.dtype))
    w = torch.randn_like(X)*.02
    pressure = X.new_tensor([0., 10665., 113324.])
    density = p.rows.pressure_density(pressure)
    fluid_work = sum((a*b*face_weights(cap.grid, c, X)).sum() for c, (a, b) in enumerate(zip(u, density)))
    torch.testing.assert_close(fluid_work, -(pressure*p.rows.flow(u)).sum(), rtol=3e-13, atol=1e-9)
    reaction = p.material_reaction(pressure)
    total = fluid_work+(reaction*w).sum()
    torch.testing.assert_close(total, -(pressure*p.relative_flow(u, w)).sum(), rtol=3e-13, atol=1e-9)
    # Uniform translation of the material cap must transfer its total force.
    expected = (p.area_vectors*pressure[cap.labels, None]).sum(0)
    torch.testing.assert_close(reaction.sum(0), expected, rtol=3e-13, atol=1e-9)


def test_reject_underresolved_invalid_and_incomplete_support(case):
    X, cap = case
    with pytest.raises(ValueError, match='underresolved'):
        MACBasalPorts(X, cap.endo, cap.grid, ports=cap.ports, subdivision_level=0)
    with pytest.raises(ValueError, match='fit strictly'):
        MACBasalPorts(X, cap.endo, cap.grid, ports=PortGeometry(mv_center_cm=(2., 0.)))
    with pytest.raises(ValueError, match='boundary'):
        cap.prepare(X+X.new_tensor([4., 0., 0.]))
    with pytest.raises(ValueError, match='pressure'):
        cap.prepare(X).rows.pressure_density(X.new_tensor([float('nan'), 1., 2.]))
    collapsed = X.clone(); collapsed[:, 0] *= 0
    with pytest.raises(ValueError, match='folded'):
        cap.prepare(collapsed)
    with pytest.raises(ValueError, match='labels'):
        prepare_flux_rows(cap.grid, X, X, torch.full((len(X),), 3, device=X.device))
