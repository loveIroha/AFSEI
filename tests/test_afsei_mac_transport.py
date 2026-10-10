"""Exact discrete volume transport, source pressure and paired MAC adjoint."""
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.mac.grid import MACGrid
from afsi_torch.p1 import cavity_rim,cavity_volume
from afsi_torch.ep.mac_transport import (closed_volume_covectors,closed_volume_gradient,
    ClosedVolumeTransport,audit_projected_transport)
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver


@pytest.fixture(params=['cpu','cuda'])
def case(request):
    if request.param=='cuda' and not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    X=torch.tensor([[0.,0.,-2.],[-1.,-1.,0.],[1.,-1.,0.],[1.,1.,0.],[-1.,1.,0.]],
        device=request.param,dtype=torch.float64)
    faces=torch.tensor([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],device=X.device,dtype=torch.int64)
    return X,faces,cavity_rim(faces)


def test_analytic_gradient_warped_cap_and_translation_invariance(case):
    X,faces,rim=case
    x=X*1.08+.1*torch.sin(X+1.3)
    x[:,2]+=.12*x[:,0]*x[:,1]
    expected=torch.func.grad(lambda z:cavity_volume(z,faces,rim))(x)
    torch.testing.assert_close(closed_volume_gradient(x,faces,rim),expected,rtol=2e-13,atol=1e-13)
    torch.testing.assert_close(closed_volume_gradient(x+X.new_tensor([20.,-30.,12.]),faces,rim),
        expected,rtol=2e-13,atol=1e-13)
    torch.testing.assert_close(expected.sum(0),X.new_zeros(3),rtol=0,atol=1e-13)


def test_source_endocardial_force_has_no_cap_added(case):
    X,faces,rim=case; x=X+.12*torch.cos(X+1.)
    wall,cap=closed_volume_covectors(x,faces,rim)
    load=MACEndocardialPressure(SimpleNamespace(endo_faces=faces,rim=rim))
    torch.testing.assert_close(load.force(x,8.),8*MMHG_TO_DYN_CM2*wall,rtol=2e-13,atol=1e-10)
    assert cap.norm()>0
    assert not torch.allclose(load.force(x,8.),8*MMHG_TO_DYN_CM2*(wall+cap))


def test_swept_chain_rule_and_geometric_pressure_work(case):
    X,faces,rim=case; x=X*1.06+.03*torch.sin(X+1.2); dt=.017
    t=ClosedVolumeTransport.prepare(X,x,faces,dt,rim=rim)
    w=(x-X)/dt; r=t.audit(w)
    assert r['swept_geometry_identity_error_ml_s']<1e-11
    assert abs(r['projected_volume_balance_error_ml_s'])<1e-11
    pressure=X.new_tensor([9100.,16300.])
    torch.testing.assert_close((t.pressure_covector(pressure)*w).sum(),
        (pressure*t.rates(w)).sum(),rtol=2e-13,atol=1e-9)
    assert abs(r['endpoint_geometry_rate_difference_ml_s'])>.01


def test_fixed_volume_finite_motion_is_not_endpoint_zero_rate(case):
    X,faces,rim=case; x=X*X.new_tensor([1.1,1/1.1,1.]); dt=.01
    t=ClosedVolumeTransport.prepare(X,x,faces,dt,rim=rim)
    r=t.audit((x-X)/dt)
    assert abs(r['secant_volume_rate_ml_s'])<1e-11
    assert abs(r['projected_closed_transport_ml_s'])<1e-11
    assert abs(r['current_closed_transport_ml_s'])>.1


def test_projection_defect_predicts_transport_balance_and_inputs_unchanged(case):
    X,faces,rim=case; x=X*1.02+.01*torch.sin(X+1.); dt=.001
    t=ClosedVolumeTransport.prepare(X,x,faces,dt,rim=rim)
    nodal=(x-X)/dt+.003*torch.sin(X+2.)
    copies=[v.clone() for v in (X,x,nodal)]
    r=t.audit(nodal)
    assert r['transport_defect_identity_error_ml_s']<1e-11
    assert r['projected_volume_balance_error_ml_s']==pytest.approx(r['predicted_projection_defect_ml_s'],abs=1e-11)
    assert abs(r['projected_volume_balance_error_ml_s'])>1e-5
    for a,b in zip((X,x,nodal),copies): assert torch.equal(a,b)


def test_invalid_geometry_history_pressure_and_velocity(case):
    X,faces,rim=case
    with pytest.raises(ValueError): ClosedVolumeTransport.prepare(X,X,faces,0,rim=rim)
    with pytest.raises(ValueError): ClosedVolumeTransport.prepare(X,X[:2],faces,.001,rim=rim)
    with pytest.raises(ValueError): ClosedVolumeTransport.prepare(X,-X,faces,.001,rim=rim)
    t=ClosedVolumeTransport.prepare(X,X,faces,.001,rim=rim)
    with pytest.raises(ValueError): t.rates(X[:2])
    with pytest.raises(ValueError): t.pressure_covector(X.new_tensor([1.]))
    with pytest.raises(ValueError): t.pressure_covector(X.new_tensor([1.,float('nan')]))


def test_original_gpu_ib_adjoint_for_nonaffine_velocity(mac_case):
    d=precise_driver(mac_case); endpoint=d._fsi_stepper(d.maximum_ticks)
    X=d.mesh.X; x=X*1.002+.001*torch.sin(X)
    t=ClosedVolumeTransport.prepare(X,x,d.model.endo_faces,.001,rim=d.model.rim)
    stencil=endpoint.transfer.prepare(X)
    u=tuple(torch.sin(endpoint.grid.coordinates(c,device=X.device)[...,c]+.5)
        for c in range(3))
    copies=[v.clone() for v in (X,x,*u)]
    r=audit_projected_transport(t,endpoint.transfer,stencil,u)
    assert r['adjoint_power_relative_error']<1e-7
    assert r['transport_defect_identity_error_ml_s']<1e-7
    assert not r['volume_transport_audit_passed'] # Deliberately not a solved motion.
    assert not r['probe_density_applied_to_simulation']
    for a,b in zip((X,x,*u),copies): assert torch.equal(a,b)
