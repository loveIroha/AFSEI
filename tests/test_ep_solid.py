from dataclasses import replace
import pytest
import torch
from afsi_torch.ep import LandQuadratureLoad, EPDrivenPaperSolid
from afsi_torch.paper_lv import imported_model
from afsi_torch.mac.paper_coupling import BEIBStepper
from test_real_lv import real_case
from test_paper_lv import config_for


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_ep_active_replacement_sparse_tangent_and_zero_double_pressure(real_case,device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    config = replace(config_for(real_case),load_protocol='active-cycle')
    baseline = imported_model(config,device)
    mesh = baseline.mesh
    one = mesh.X.new_ones(len(mesh.X)); zero = torch.zeros_like(one)
    load = LandQuadratureLoad.prepare(mesh.X,mesh.cells,mesh.fiber,mesh.X,
        .03*one+.001*(mesh.X[:,0]-mesh.X[:,0].min()),.05*one,zero,zero,config.time.dt)
    model = EPDrivenPaperSolid(mesh,config,load)
    # Deform away from material kinks for the independent finite difference.
    x = mesh.X+.01*(mesh.X-mesh.X.new_tensor([7.5,7.5,7.5]))
    F,area,_ = model.geometry_state(x)
    passive = baseline.force_from_geometry(x,F,area,(0.,0.))
    expected = passive+load.force(x)
    torch.testing.assert_close(model.force(x,.4),expected,rtol=1e-12,atol=1e-8)
    # CUDA nodal assembly uses atomic additions; ordering can differ at roundoff.
    torch.testing.assert_close(model.force(x,.7),expected,rtol=1e-12,atol=1e-8)
    assert model.loads.at(.4) == (0.,0.)
    # A stale synthetic load pair must not sneak back through cached geometry.
    torch.testing.assert_close(model.force_from_geometry(x,F,area,(1e5,1e6)),expected,rtol=1e-12,atol=1e-8)
    direction = torch.sin(x); direction /= direction.norm()
    J = model.tangent_factory(16).assemble(x,.4)
    action = torch.sparse.mm(J,direction.reshape(-1,1)).reshape_as(x)
    eps = 2e-6
    fd = (model.force(x+eps*direction,.4)-model.force(x-eps*direction,.4))/(2*eps)
    torch.testing.assert_close(action,fd,rtol=3e-6,atol=3e-3)
    diagnostics = model.diagnostics(x)
    assert diagnostics['additional_endocardial_pressure_cgs'] == 0
    assert diagnostics['minimum_active_stretch_multiplier'] >= 1


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_ep_load_through_actual_be_ib_endpoint(real_case,device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    config = config_for(real_case,solver='newton')
    baseline = imported_model(config,device)
    mesh = baseline.mesh
    one = mesh.X.new_ones(len(mesh.X)); zero = torch.zeros_like(one)
    load = LandQuadratureLoad.prepare(mesh.X,mesh.cells,mesh.fiber,mesh.X,
        1e-5*one,1e-5*one,zero,zero,config.time.dt)
    model = EPDrivenPaperSolid(mesh,config,load)
    driver = BEIBStepper(model,config,device)
    old = driver.initialize(mesh.X)
    original = old.x.clone()
    new,info = driver.step(old)
    torch.testing.assert_close(old.x,original,rtol=0,atol=0)
    torch.testing.assert_close(new.force,model.force(new.x,new.time),rtol=1e-10,atol=1e-8)
    assert new.force_time == new.time
    assert info['divergence_l2'] < 1e-8
    assert info['power_error'] < 1e-6
    assert (new.x-old.x).abs().max() > 0
