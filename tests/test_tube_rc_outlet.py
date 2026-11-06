"""Boundary physics checks, independent inhomogeneous projection, and FSI JVP."""
import json
from pathlib import Path
import pytest
import torch
from afsi_torch.circulation.rc_outlet import (RCParameters, RCChannelFlow, RCBoxFlow,
                                              MMHG_TO_DYN_CM2, parse_rc)


def test_paper_units_and_explicit_planar_depth():
    p = RCParameters()
    assert p.resistance == pytest.approx(60*1333.22387415*1000)
    assert p.compliance == pytest.approx(.26/1000/1333.22387415)
    assert p.resistance*p.compliance == pytest.approx(15.6)
    with pytest.raises(ValueError):
        RCParameters(resistance_mmhg_s_per_mm3=0)
    root = Path(__file__).resolve().parents[1]
    for dim in (2,3):
        c = json.loads((root/f'demo/active_tube_{dim}d/configs/tube_sharifi_rc_gpu.json').read_text())
        parse_rc(c['circulation'], dimension=dim)
        if dim == 2:
            from afsi_torch.simulation.active_tube import validate_config
            validate_config(c)
            del c['circulation']['extrusion_depth_cm']
            with pytest.raises(ValueError, match='extrusion_depth'):
                validate_config(c)
        else:
            from afsi_torch.simulation.circular_tube_config import config_summary
            s = config_summary(c)
            assert s['circulation'] and not s['porous_media']
            assert not s['physical_tube_mouth_boundary_aligned']


def make_flow(dim, device):
    like = torch.empty((),dtype=torch.float64,device=device)
    params = RCParameters(initial_pressure_mmhg=.001)
    if dim == 2:
        from afsi_torch.mac2d.grid import ChannelGrid
        from afsi_torch.mac2d.pressure_channel import PressureChannelFlow
        from afsi_torch.mac2d.multigrid import MGOptions
        grid = ChannelGrid((16,8),(2.,1.))
        base = PressureChannelFlow(grid,dt=5e-5,rho=1.,mu=.04,
            inlet_pressure=0.,outlet_pressure=0.,device=device,fused=False,
            options=MGOptions(rtol=1e-12,atol=1e-13,max_cycles=200))
        flow = RCChannelFlow(base,params,like,extrusion_depth_cm=.7)
    else:
        from afsi_torch.mac.grid import MACGrid
        from afsi_torch.mac.backward_euler import BackwardEulerFlow, BEFlowOptions
        from afsi_torch.mac.multigrid import MGOptions
        grid = MACGrid((8,8,12),(2.,2.,3.))
        base = BackwardEulerFlow(grid,dt=5e-5,rho=1.,mu=.04,device=device,
            options=BEFlowOptions(convection=False),
            pressure_options=MGOptions(rtol=1e-12,atol=1e-13,max_cycles=200),
            backend='torch',pressure_backend='torch')
        flow = RCBoxFlow(base,params,like)
    return flow,grid,like


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dim',[2,3])
def test_implicit_rc_matches_direct_pressure_boundary_without_history_mutation(dim,device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    flow,grid,like = make_flow(dim,device)
    old = flow.pressure.clone()
    zero = grid.zeros(device=device,dtype=like.dtype)
    # Exercise inflow/outflow signs, not just an identically zero resting case.
    for sign in (1.,-1.):
        density = tuple(torch.full_like(v,sign*100 if j == flow.axis else 0)
                        for j,v in enumerate(zero))
        if dim == 2:
            b = flow.base.step(zero,density,time=0.)
            r = flow.step(zero,density,time=0.)
            u0,p0 = b.velocity,b.pressure
            u,p = r.velocity,r.pressure
            from afsi_torch.mac2d.pressure_channel import pressure_gradient
            from afsi_torch.mac2d.grid import divergence
            grad0 = pressure_gradient(p0,grid.spacing,0.,0.)
        else:
            u0,p0,_ = flow.base.advance(zero,density)
            u,p,_ = flow.advance(zero,density)
            from afsi_torch.mac.open_boundary import gradient
            from afsi_torch.mac.grid import divergence
            grad0 = gradient(p0,grid.spacing)
        torch.testing.assert_close(flow.pressure,old,rtol=0,atol=0)
        assert flow.committed_steps == 0
        # Recover the actual tentative velocity; solve a new inhomogeneous
        # pressure system with the RC-consistent pressure, independently.
        tentative = tuple(v+flow.dt/flow.rho*g for v,g in zip(u0,grad0))
        q = flow.flux(u)
        pout = (flow.C*old+flow.dt*q)/flow.D
        rhs = -flow.rho/flow.dt*divergence(tentative,grid.spacing)
        rhs.select(flow.axis,-1).add_(2*pout/grid.spacing[flow.axis]**2)
        direct_p,_ = flow.base.pressure_solver.solve(rhs)
        if dim == 2:
            direct_g = pressure_gradient(direct_p,grid.spacing,0.,pout)
        else:
            direct_g = list(gradient(direct_p,grid.spacing))
            direct_g[2][:,:,-1] += 2*pout/grid.spacing[2]
        direct_u = tuple(v-flow.dt/flow.rho*g for v,g in zip(tentative,direct_g))
        torch.testing.assert_close(p,direct_p,rtol=2e-9,atol=2e-10)
        for a,b in zip(u,direct_u):
            torch.testing.assert_close(a,b,rtol=2e-8,atol=2e-12)
        assert float(divergence(u,grid.spacing).abs().max()) < 1e-10
    # Commit only the second accepted state. No positive-flow clipping.
    flow.commit(u,step=1)
    d = flow.diagnostics()
    assert d['rc_flow_ml_s'] < 0
    assert abs(d['rc_balance_ml_s']) < 1e-12
    assert abs(d['rc_energy_error_erg_s']) < 1e-12
    with pytest.raises(ValueError, match='exactly once'):
        flow.commit(u,step=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_three_dimensional_fsi_response_includes_rc_derivative(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    flow,grid,like = make_flow(3,device)
    zero = grid.zeros(device=device,dtype=like.dtype)
    forcing = tuple(torch.full_like(v,.1*(j+1)) for j,v in enumerate(zero))
    direction = tuple(torch.sin(grid.coordinates(j,device=device,dtype=like.dtype)[...,j]+.3)
                      for j in range(3))
    eps = .01
    plus = tuple(f+eps*v for f,v in zip(forcing,direction))
    minus = tuple(f-eps*v for f,v in zip(forcing,direction))
    a,_,_ = flow.advance(zero,plus)
    b,_,_ = flow.advance(zero,minus)
    derivative = flow.response(direction)
    for x,y,d in zip(a,b,derivative):
        torch.testing.assert_close((x-y)/(2*eps),d,rtol=2e-7,atol=2e-11)
    assert flow.committed_steps == 0
