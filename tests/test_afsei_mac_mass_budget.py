"""Closed-surface orientation, exact swept-volume identity and GPU trace audit."""
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.ep.mac_mass_budget import ClosedCavityMassAudit,rt0_trace


@pytest.fixture(params=['cpu','cuda'])
def case(request):
    if request.param=='cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    X=torch.tensor([[0.,0.,-2.],[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]],
        device=request.param,dtype=torch.float64)
    faces=torch.tensor([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],device=X.device,dtype=torch.int64)
    grid=MACGrid((16,)*3,(8.,)*3,(-4.,)*3)
    return X,grid,ClosedCavityMassAudit(X,faces,grid)


def test_uniform_translation_and_unchanged_inputs(case):
    X,grid,audit=case; dt=.03
    speed=X.new_tensor([.2,-.3,.4]); x=X+dt*speed
    u=tuple(torch.full_like(v,speed[c]) for c,v in enumerate(grid.zeros(device=X.device)))
    copies=[v.clone() for v in (X,x,*u)]
    r=audit.evaluate(X,x,u,dt=dt)
    assert r['surface_mass_audit_passed']
    assert r['swept_geometry_identity_error_ml_s']<1e-12
    assert r['levels'][-1]['closed_absolute_relative_flux_ml_s']<1e-12
    for a,b in zip((X,x,*u),copies): assert torch.equal(a,b)


def test_exact_swept_geometry_and_nonzero_endpoint_error(case):
    X,grid,audit=case
    x=X*1.07+.03*torch.sin(X+1.)
    r=audit.evaluate(X,x,grid.zeros(device=X.device),dt=.013)
    assert r['swept_geometry_identity_error_ml_s']<1e-10
    assert abs(r['endpoint_vs_secant_material_rate_ml_s'])>.01
    assert not r['surface_mass_audit_passed']


def test_affine_dilation_closed_gauss_flux_and_quadrature(case):
    X,grid,audit=case; rate=.15
    u=tuple(rate*grid.coordinates(c,device=X.device)[...,c] for c in range(3))
    r=audit.evaluate(X,X,u,dt=.001)
    expected=3*rate*r['volume_current_ml']
    for level in r['levels']:
        assert level['closed_fluid_flux_ml_s']==pytest.approx(expected,abs=1e-12)
        assert level['rt0_closed_fluid_flux_ml_s']==pytest.approx(expected,abs=1e-12)
        assert level['flux_decomposition_error_ml_s']<1e-12
    assert not r['surface_mass_audit_passed']
    assert r['peskin_quadrature_converged_to_targets']


def test_rt0_cell_derivative_equals_mac_divergence(case):
    X,grid,_=case
    torch.manual_seed(81)
    u=tuple(torch.randn_like(v) for v in grid.zeros(device=X.device))
    cell=torch.tensor([[5,6,7],[8,9,10]],device=X.device)
    point=X.new_tensor(grid.origin)+(cell+.37)*X.new_tensor(grid.spacing)
    eps=1e-5; derivative=point.new_zeros(len(point))
    for c,h in enumerate(grid.spacing):
        delta=torch.zeros_like(point); delta[:,c]=eps
        derivative+=(rt0_trace(grid,u,point+delta)[:,c]-rt0_trace(grid,u,point-delta)[:,c])/(2*eps)
    from afsi_torch.mac.grid import divergence
    exact=divergence(u,grid.spacing)[cell[:,0],cell[:,1],cell[:,2]]
    torch.testing.assert_close(derivative,exact,rtol=1e-10,atol=1e-10)


def test_invalid_history_and_controls(case):
    X,grid,audit=case; u=grid.zeros(device=X.device)
    with pytest.raises(ValueError): audit.evaluate(X,X,u,dt=0)
    with pytest.raises(ValueError): audit.evaluate(X,X,u,dt=.001,levels=(2,1))
    with pytest.raises(ValueError): audit.evaluate(X,X[:2],u,dt=.001)
