"""Analytic closed-volume identities and independent MAC flux diagnostics."""
import pytest
import torch
from afsi_torch.simulation.tube_budget import TubeBudget, rt0_velocity


@pytest.fixture(params=['cpu','cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def setup_case(dim,device):
    from afsi_torch.mac.sealed_ports import SealedDomain
    opts=dict(kind='sealed-end-reservoirs',aperture_margin_cells=.25,collar_cells=2,pressure_max_iterations=200)
    if dim == 2:
        from afsi_torch.solids.planar_tube import TubeGeometry,generate_tube
        from afsi_torch.mac2d.grid import ChannelGrid
        g=TubeGeometry(axial_elements=8,thickness_elements=1)
        m=generate_tube(g,device=device)
        grid=ChannelGrid((64,32),(3.2,1.6))
        lo=torch.where((m.X[:m.vertex_count,1]-.5).abs()<1e-12)[0]
        hi=torch.where((m.X[:m.vertex_count,1]-1.1).abs()<1e-12)[0]
        domain=SealedDomain(grid,g,opts,m.X,dimension=2,depth=.7)
        b=TubeBudget(m,domain,lower=lo,upper=hi)
    else:
        from afsi_torch.solids.circular_tube import CircularTubeGeometry,generate_circular_tube
        from afsi_torch.mac.grid import MACGrid
        g=CircularTubeGeometry(axial_elements=4,circumferential_elements=16,thickness_elements=1)
        m=generate_circular_tube(g,device=device)
        grid=MACGrid((32,32,40),(4.,4.,5.))
        domain=SealedDomain(grid,g,opts,m.X,dimension=3)
        b=TubeBudget(m,domain)
    return m,grid,domain,b


@pytest.mark.parametrize('dim',[2,3])
def test_curved_volume_and_sweeps(device,dim):
    m,grid,domain,b=setup_case(dim,device)
    old=m.X.clone()
    # Non-affine deformation: exercises curved P2 edges / nonplanar P1 caps,
    # plus substantial end motion. Not used as a physical simulation.
    new=old.clone()
    new[:,0]+=.008*torch.sin(old[:,1]*3)
    new[:,1]+=.012*torch.sin(old[:,0]*2)
    if dim == 3:
        new[:,2]+=.01*torch.sin(old[:,0]*4)
    velocity=tuple(old.new_zeros(grid.face_shape(c)) for c in range(dim))
    b.accepted(old,new,velocity,.01)
    r=b.diagnostics(velocity)
    assert abs(r['budget_geometric_identity_error_ml'])<2e-12
    assert abs(r['budget_fixed_geometric_identity_error_ml'])<2e-12
    assert torch.equal(old,m.X)
    assert not r['budget_integrated_net_inflow_ml']
    assert r['budget_accepted_steps']==1
    if dim == 2:
        assert float(b.initial_moving_volume)==pytest.approx(2.4*.6*.7)
    else:
        from afsi_torch.solids.circular_tube import lumen_volume
        assert r['budget_moving_lumen_volume_ml']==pytest.approx(float(lumen_volume(new,m)),abs=2e-12)


@pytest.mark.parametrize('dim',[2,3])
def test_constant_transport_is_not_a_leak(device,dim):
    m,grid,domain,b=setup_case(dim,device)
    # Uniform axial velocity does not obey the physical septum masks, so this
    # checks surface integration on a closed surface, not the sealed BC.
    v=tuple(m.X.new_full(grid.face_shape(c),.3*(c+1)) for c in range(dim))
    x=b.extend(m.X)
    total=sum(s.flux(x,v,grid,1) for s in (b.wall,b.connectors,b.fixed_caps))
    assert float(total)==pytest.approx(0.,abs=2e-12)
    points=m.X[:10]+.001
    exact=m.X.new_tensor([.3*(c+1) for c in range(dim)]).expand_as(points)
    torch.testing.assert_close(rt0_velocity(grid,v,points),exact,atol=1e-14,rtol=1e-14)


@pytest.mark.parametrize('dim',[2,3])
def test_per_step_accumulation_and_checkpoint(device,dim):
    m,grid,domain,b=setup_case(dim,device)
    v=tuple(m.X.new_zeros(grid.face_shape(c)) for c in range(dim))
    axis=domain.axis
    for step in range(3):
        v[axis].select(axis,domain.planes[0]).fill_(step+1)
        v[axis].select(axis,domain.planes[1]).fill_(.2*(step+1))
        b.accepted(m.X,m.X,v,.001)
    expected=6*.001*float(domain.aperture.sum())*domain.area
    r=b.diagnostics()
    assert r['budget_integrated_inlet_ml']==pytest.approx(expected)
    assert r['budget_integrated_outlet_ml']==pytest.approx(.2*expected)
    assert r['budget_fixed_control_imbalance_ml']==pytest.approx(-.8*expected)
    assert b.checkpoint()['tube_budget_steps']==3


def test_deformed_window_baseline(device):
    m,grid,domain,_=setup_case(3,device)
    old=m.X.clone();old[:,0]+=.01*torch.sin(old[:,2])
    b=TubeBudget(m,domain,initial_x=old)
    v=tuple(old.new_zeros(grid.face_shape(c)) for c in range(3))
    new=old.clone();new[:,1]+=.01*torch.cos(old[:,2])
    b.accepted(old,new,v,.001)
    r=b.diagnostics()
    assert abs(r['budget_geometric_identity_error_ml'])<2e-12
    assert abs(r['budget_fixed_geometric_identity_error_ml'])<2e-12
