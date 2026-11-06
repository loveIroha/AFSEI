"""Manufactured cut-face boundary data and implicit RCR linearization."""
import os
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.backward_euler import BackwardEulerFlow, BEFlowOptions
from afsi_torch.mac.sealed_flow import SealedBEFlow
from afsi_torch.mac.cut_plane import CutPlane, CutPlaneFlow
from afsi_torch.ep.windkessel import WindkesselParameters, MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.open_boundary import face_weights


def channel(dt=1e-4, device=None,compact=False,pressure_tolerances=None):
    device = device or os.environ.get('AFSEI_TEST_DEVICE', 'cpu')
    grid = MACGrid((8, 8, 16), (1., 1., 2.))
    masks = list(torch.ones(grid.face_shape(c), device=device, dtype=torch.float64) for c in range(3))
    for axis in (0, 1):
        for side in (0, -1):
            masks[axis].select(axis, side).zero_()
    domain = SimpleNamespace(masks=tuple(masks), planes=(), config={'pressure_max_iterations': 400,
        'pressure_compact_geometry':compact,'pressure_solver_tolerances':pressure_tolerances})
    domain.mask = lambda u: tuple(a * m for a, m in zip(u, domain.masks))
    base = BackwardEulerFlow(grid, dt=dt, device=device,
        options=BEFlowOptions(convection=False, helmholtz_backend='workspace'))
    sealed = SealedBEFlow(base, domain, fused=False, graphs=False)
    aperture = torch.ones(grid.shape[:2], device=device, dtype=torch.bool)
    return CutPlaneFlow(sealed, (CutPlane(2, 0, aperture, 'inlet'), CutPlane(2, -1, aperture, 'outlet')))


@pytest.mark.parametrize('dt',[1e-4,5e-5])
def test_stricter_inner_pressure_retains_original_rcr_boundary(dt):
    f=channel(dt,pressure_tolerances=dict(rtol=1e-12,atol=1e-14))
    assert f.pressure_solver.cg_options.rtol==1e-12
    assert f.base.base.pressure_solver.options.rtol==1e-10
    zero=f.grid.zeros(device=f.domain.masks[0].device)
    f.rcr(7.,WindkesselParameters(),outlet=1,other_pressures_mmhg=[8.,0.])
    u,_,_=f.advance(zero,zero)
    conductance=dt*C/2
    intercept=(1.3*7+dt*8/1.05)/(1.3+dt/1.05)
    impedance=.05+dt/(1.3+dt/1.05)
    q=f.flux(u)[1].item()
    assert abs(q-conductance*(8-intercept)/(1+conductance*impedance))<1e-10
    assert abs(f.last_boundary['rcr_gap_mmhg'].item())<1e-10


@pytest.mark.parametrize('dt', [1e-4, 5e-5])
@pytest.mark.parametrize('compact',[False,True])
def test_uniform_pressure_and_linear_pressure_acceleration(dt,compact):
    f = channel(dt,compact=compact)
    zero = f.grid.zeros(device=f.domain.masks[0].device)
    f.prescribe([8., 8.])
    u, p, _ = f.advance(zero, zero)
    assert max(v.abs().max().item() for v in u) < 1e-9
    assert (p / C - 8.).abs().max().item() < 1e-8
    f.prescribe([8., 7.])
    u, p, _ = f.advance(zero, zero)
    z = f.grid.coordinates(device=p.device)[..., 2]
    assert (p / C - (8. - z / 2.)).abs().max().item() < 1e-8
    assert (u[2] - dt * C / 2.).abs().max().item() < 1e-9
    assert f.last_boundary['outward_flux_ml_s'].sum().abs().item() < 1e-9
    energy=.5*f.rho*sum((v*v*face_weights(f.grid,c,p)).sum() for c,v in enumerate(u))
    # Uniform acceleration: no viscous work. BE storage and BE time defect
    # are equal, so physical boundary power must be 2 E[n+1] / dt.
    power=f.last_boundary['power_into_fluid_erg_s']
    assert (abs(power-2*energy/dt)/power.abs()).item() < 1e-8


@pytest.mark.parametrize('compact',[False,True])
def test_rcr_true_balance_and_homogeneous_derivative(compact):
    f = channel(compact=compact)
    zero = f.grid.zeros(device=f.domain.masks[0].device)
    f.rcr(7., WindkesselParameters(), outlet=1, other_pressures_mmhg=[8., 0.])
    u0, p0, _ = f.advance(zero, zero)
    q = f.last_boundary['outward_flux_ml_s'][1]
    trial = f.last_circuit
    # Independent uniform-channel formula, not another call to the RCR code.
    conductance=f.dt*C/(f.rho*2.)
    intercept=(1.3*7.+f.dt*8./1.05)/(1.3+f.dt/1.05)
    impedance=.05+f.dt/(1.3+f.dt/1.05)
    exact_q=conductance*(8.-intercept)/(1.+conductance*impedance)
    assert abs(q.item()-exact_q) < 1e-9
    assert f.last_boundary['rcr_gap_mmhg'].abs().item() < 1e-10
    assert abs(1.3 * (trial.pc_mmhg - 7.) / f.dt - q + (trial.pc_mmhg - 8.) / 1.05) < 1e-9
    assert f.old_pc == 7.
    forcing = tuple(torch.sin(torch.arange(v.numel(), device=v.device, dtype=v.dtype)).reshape_as(v) for v in zero)
    du = f.response(forcing)
    u1, _, _ = f.advance(zero, forcing)
    error = torch.sqrt(sum((a - b - d).square().sum() for a, b, d in zip(u1, u0, du)))
    scale = torch.sqrt(sum(v.square().sum() for v in du))
    assert (error / scale).item() < 2e-7
    assert torch.isfinite(p0).all()


@pytest.mark.parametrize('compact',[False,True])
def test_cap_preconditioner_uses_the_same_masked_mac_projection(compact):
    from afsi_torch.ep.mac_seal import MACCapProjection
    f=channel(compact=compact)
    rows=SimpleNamespace(prepared=SimpleNamespace(cap=SimpleNamespace(grid=f.grid)),
                         area=f.pressures.new_ones(1))
    projector=MACCapProjection(rows,f.pressure_solver)
    zero=f.grid.zeros(device=f.domain.masks[0].device)
    u=tuple(torch.cos(torch.arange(v.numel(),device=v.device,dtype=v.dtype)).reshape_as(v) for v in zero)
    projected,_=projector.divergence_project(u)
    from afsi_torch.mac.grid import divergence
    assert divergence(projected,f.grid.spacing).abs().max().item()<1e-7
    assert all(v[m==0].abs().max().item()==0. for v,m in zip(projected[:2],f.domain.masks[:2]))
    again,_=projector.divergence_project(projected)
    assert max((a-b).abs().max().item() for a,b in zip(again,projected))<1e-8
