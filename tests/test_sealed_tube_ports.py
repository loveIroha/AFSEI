"""Discrete geometry, projection, RC power and FSI derivative checks."""
import pytest
import torch
import torch.nn.functional as F
from afsi_torch.mac.sealed_ports import SealedDomain
from afsi_torch.mac.sealed_pressure import action,gradient,divergence,coarsen
from afsi_torch.mac.sealed_flow import SealedBEFlow,SealedChannelFlow
from afsi_torch.circulation.rc_outlet import RCParameters,RCBoxFlow,RCChannelFlow


def build(dim,device):
    like=torch.empty((),dtype=torch.float64,device=device)
    c=dict(kind='sealed-end-reservoirs',aperture_margin_cells=.25,collar_cells=2,pressure_max_iterations=200)
    if dim==2:
        from afsi_torch.mac2d.grid import ChannelGrid
        from afsi_torch.solids.planar_tube import TubeGeometry
        from afsi_torch.mac2d.pressure_channel import PressureChannelFlow
        from afsi_torch.mac.multigrid import MGOptions
        grid=ChannelGrid((32,16),(3.2,1.6));geometry=TubeGeometry()
        domain=SealedDomain(grid,geometry,c,like,dimension=2,depth=.7)
        base=PressureChannelFlow(grid,dt=5e-5,rho=1.,mu=.04,inlet_pressure=0.,outlet_pressure=0.,
            device=device,fused=False,options=MGOptions(rtol=1e-11,atol=1e-12))
        flow=SealedChannelFlow(base,domain,fused=False,graphs=False)
        rc=RCChannelFlow(flow,RCParameters(initial_pressure_mmhg=.001),like,extrusion_depth_cm=.7)
    else:
        from afsi_torch.mac.grid import MACGrid
        from afsi_torch.solids.circular_tube import CircularTubeGeometry
        from afsi_torch.mac.backward_euler import BackwardEulerFlow,BEFlowOptions
        from afsi_torch.mac.multigrid import MGOptions
        grid=MACGrid((8,8,12),(4.,4.,6.));geometry=CircularTubeGeometry(axial_start_cm=1.)
        domain=SealedDomain(grid,geometry,c,like,dimension=3)
        base=BackwardEulerFlow(grid,dt=5e-5,rho=1.,mu=.04,device=device,
            options=BEFlowOptions(convection=False),backend='torch',pressure_backend='torch',
            pressure_options=MGOptions(rtol=1e-11,atol=1e-12))
        flow=SealedBEFlow(base,domain,fused=False,graphs=False)
        rc=RCBoxFlow(flow,RCParameters(initial_pressure_mmhg=.001),like)
    return grid,domain,rc,like


@pytest.mark.parametrize('dim',[2,3])
def test_masked_gradient_adjoint_spd_and_exact_galerkin(dim):
    grid,domain,flow,like=build(dim,'cpu')
    p=torch.randn(grid.shape,dtype=like.dtype);q=torch.randn_like(p)
    Ap=action(p,grid.spacing,domain.masks);Aq=action(q,grid.spacing,domain.masks)
    torch.testing.assert_close((p*Aq).sum(),(q*Ap).sum(),rtol=1e-12,atol=1e-10)
    assert float((p*Ap).sum())>0
    velocity=domain.mask(tuple(torch.randn_like(v) for v in grid.zeros(dtype=like.dtype)))
    g=gradient(p,grid.spacing,domain.masks)
    power=p.new_zeros(())
    for c,(u,v) in enumerate(zip(velocity,g)):
        weights=torch.ones_like(u);weights.select(c,0).fill_(.5);weights.select(c,-1).fill_(.5)
        power+=(u*v*weights).sum()
    torch.testing.assert_close(power,-(p*divergence(velocity,grid.spacing)).sum(),rtol=1e-12,atol=1e-10)
    coarse=tuple(n//2 for n in grid.shape);z=torch.randn(coarse,dtype=like.dtype);fine=z
    for a in range(dim):fine=fine.repeat_interleave(2,dim=a)
    pool=F.avg_pool2d if dim==2 else F.avg_pool3d
    RAp=pool(action(fine,grid.spacing,domain.masks)[None,None],2,2)[0,0]
    Ac=action(z,tuple(2*h for h in grid.spacing),coarsen(domain.masks))
    torch.testing.assert_close(RAp,Ac,rtol=1e-12,atol=1e-11)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dim',[2,3])
def test_reservoir_flux_matches_aperture_and_rc_power(dim,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    grid,domain,flow,like=build(dim,device)
    zero=grid.zeros(dtype=like.dtype,device=device)
    force=tuple(torch.full_like(v,100. if j==domain.axis else 0.) for j,v in enumerate(zero))
    if dim==2:
        r=flow.step(zero,force,time=0.);u=r.velocity
    else:u,_,_=flow.advance(zero,force)
    assert float(divergence(u,grid.spacing).abs().max())<1e-10
    for v,m in zip(u,domain.masks):
        assert float(v[m==0].abs().max())==0.
    assert flow.committed_steps==0
    flow.commit(u,step=1)
    d=domain.diagnostics()
    assert abs(d['outlet_flux_mismatch_ml_s'])<1e-10
    assert abs(d['inlet_flux_mismatch_ml_s'])<1e-10
    assert abs(flow.diagnostics()['rc_port_power_error_erg_s'])<1e-8
    assert abs(flow.diagnostics()['rc_balance_ml_s'])<1e-12


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_newton_response_includes_sealed_geometry_and_rc(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    grid,domain,flow,like=build(3,device)
    zero=grid.zeros(dtype=like.dtype,device=device)
    force=tuple(torch.full_like(v,50.*(c+1)) for c,v in enumerate(zero))
    direction=tuple(torch.cos(grid.coordinates(c,device=device,dtype=like.dtype)[...,c]) for c in range(3))
    eps=.1
    p,_,_=flow.advance(zero,tuple(f+eps*d for f,d in zip(force,direction)))
    m,_,_=flow.advance(zero,tuple(f-eps*d for f,d in zip(force,direction)))
    j=flow.response(direction)
    for a,b,v in zip(p,m,j):torch.testing.assert_close((a-b)/(2*eps),v,rtol=1e-5,atol=1e-10)
    assert flow.committed_steps==0


@pytest.mark.parametrize('dim',[2,3])
def test_original_wall_overlaps_collar_and_detachment_is_rejected(dim):
    like=torch.empty((),dtype=torch.float64)
    c=dict(kind='sealed-end-reservoirs',aperture_margin_cells=.25,collar_cells=2,pressure_max_iterations=200)
    if dim==2:
        from afsi_torch.mac2d.grid import ChannelGrid
        from afsi_torch.solids.planar_tube import TubeGeometry,generate_tube
        grid=ChannelGrid((64,32),(3.2,1.6));g=TubeGeometry();mesh=generate_tube(g)
    else:
        from afsi_torch.mac.grid import MACGrid
        from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube
        grid=MACGrid((64,64,80),(4.,4.,5.));g=CircularTubeGeometry();mesh=generate_circular_tube(g)
    domain=SealedDomain(grid,g,c,like,dimension=dim)
    domain.bind_wall(mesh);domain.validate_wall(mesh.X)
    x=mesh.X.clone();x[domain.end_ids[0],domain.axis]+=.3
    with pytest.raises(ValueError,match='detached'):
        domain.validate_wall(x)
