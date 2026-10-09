import pytest
import torch
from afsi_torch.two_way.preload import ClosedPressure,coarse_basis,CoarseInverse,PreloadConfig
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from test_moving_cavity import cube
from test_three_field_driver import physical_case,arrays
from test_real_lv import real_case


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_closed_pressure_is_volume_gradient_and_exact_follower_derivative(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    mesh=cube(device); X=mesh.reference_solid; load=ClosedPressure(mesh)
    v=torch.sin(3*X+X.new_tensor([.1,.3,.7])); eps=1e-6
    g=load.gradient(X)
    derivative=(load.volume(X+eps*v)-load.volume(X-eps*v))/(2*eps)
    torch.testing.assert_close(derivative,(g*v).sum(),rtol=1e-8,atol=1e-9)
    actual=load.action(X,8.,v)
    fd=(load.force(X+eps*v,8.)-load.force(X-eps*v,8.))/(2*eps)
    torch.testing.assert_close(actual,fd,rtol=1e-8,atol=2e-5)
    many=torch.stack((v,2*v,-v),2)
    torch.testing.assert_close(load.action(X,8.,many),torch.stack((actual,2*actual,-actual),2),rtol=1e-12,atol=1e-10)
    # Hessian of p*V must be symmetric, including the mapped basal cap.
    w=torch.cos(X+1.)
    torch.testing.assert_close((w*actual).sum(),(v*load.action(X,8.,w)).sum(),rtol=1e-12,atol=1e-9)


def test_coarse_space_contains_rigid_and_affine_fields():
    X=cube().reference_solid; P=coarse_basis(X,(3,3,3))
    expected=(X@X.new_tensor([[1.,2.,3.],[-1.,0.,1.],[.1,.2,.3]])+X.new_tensor([1.,2.,3.])).flatten()
    reconstructed=P@torch.linalg.lstsq(P,expected).solution
    torch.testing.assert_close(reconstructed,expected,rtol=1e-11,atol=1e-11)
    diagonal=torch.ones_like(X)*2
    action=lambda v:2*v
    inverse=CoarseInverse(action,diagonal,P)
    rhs=torch.cos(X)
    torch.testing.assert_close(action(inverse(rhs)),rhs,rtol=1e-11,atol=1e-11)


def test_preload_configuration_rejects_bad_continuation():
    with pytest.raises(ValueError): PreloadConfig(target_mmhg=-8.)
    with pytest.raises(ValueError): PreloadConfig(minimum_step_mmhg=2.)


def test_passive_equilibrium_initialization_checkpoint_and_first_step(physical_case,tmp_path):
    from dataclasses import replace
    from afsi_torch.two_way import ThreeFieldStepper,checkpoint
    from afsi_torch.two_way.preload import PassivePreloader,initialize_preloaded
    inherited,old=physical_case
    d=ThreeFieldStepper(inherited.mesh,inherited.config,inherited.blood_driver.mesh,
        replace(inherited.blood_driver.config,atrium_mmhg=8.))
    reference=d.mesh.X.clone()
    initial=d.initialize_three_fields()
    result=PassivePreloader(d).solve()
    state,report=initialize_preloaded(d,result)
    assert result.pressure_mmhg==8. and result.volume_ml>result.reference_volume_ml
    assert report['preload_force_residual_norm_dyn']<=report['preload_force_tolerance_dyn']
    assert state.blood.opened==(False,False) and state.ep.fsi.step==state.blood.step==0
    assert state.ep.tick==state.blood.tick==initial.ep.tick
    assert not state.ep.force_load.use_land
    for value in (state.ep.fsi.velocity,state.blood.velocity,state.ep.strain.zs,state.ep.strain.zw):
        assert torch.count_nonzero(value)==0
    for name in ('voltage_v','gpb','land'):
        torch.testing.assert_close(getattr(state.ep.tissue,name),getattr(initial.ep.tissue,name),rtol=0,atol=0)
    torch.testing.assert_close(d.mesh.X,reference,rtol=0,atol=0)
    torch.testing.assert_close(state.blood.pressure,torch.full_like(state.blood.pressure,8*C),rtol=0,atol=0)
    assert report['circulation_stationary'] is False and report['initial_pc_runoff_ml_s']>0
    path=tmp_path/'preloaded.npz'; checkpoint.save(path,d,state,report)
    restored,restart,progress=checkpoint.load(path)
    assert progress['preload_performed'] is True
    for x,y in zip(arrays(state),arrays(restart)): torch.testing.assert_close(x,y,rtol=0,atol=0)
    direct,diag,_=d.step_three_fields(state)
    resumed,_,_=restored.step_three_fields(restart)
    for x,y in zip(arrays(direct),arrays(resumed)): torch.testing.assert_close(x,y,rtol=3e-7,atol=2e-7)
    assert diag['solid_momentum_residual_norm_dyn']<=diag['solid_momentum_tolerance_dyn']
    assert abs(diag['mass_error_ml_s'])<=1e-8
    assert direct.ep.tick>state.ep.tick and direct.blood.pc_mmhg<state.blood.pc_mmhg
    with pytest.raises(ValueError,match='atrium'):
        initialize_preloaded(inherited,result)


def test_preload_failure_retains_only_last_converged_geometry(physical_case,monkeypatch):
    from afsi_torch.two_way import preload
    d,_=physical_case; reference=d.mesh.X.clone()
    def fail(*args,**kwargs): raise RuntimeError('injected linear failure')
    monkeypatch.setattr(preload,'newton',fail)
    with pytest.raises(preload.PreloadFailure) as info: preload.PassivePreloader(d).solve()
    error=info.value
    assert not error.history and len(error.rejected)==5
    torch.testing.assert_close(error.accepted_x,reference,rtol=0,atol=0)
    torch.testing.assert_close(d.mesh.X,reference,rtol=0,atol=0)


def test_later_failure_retains_converged_deformed_state(physical_case,monkeypatch):
    from afsi_torch.two_way import preload
    d,_=physical_case; solve=preload.newton; accepted=[]
    def fail_after_one(*args,**kwargs):
        if accepted: raise RuntimeError('injected failure at subsequent pressure')
        result=solve(*args,**kwargs); accepted.append(result.x.clone()); return result
    monkeypatch.setattr(preload,'newton',fail_after_one)
    with pytest.raises(preload.PreloadFailure) as info:
        preload.PassivePreloader(d,preload.PreloadConfig(target_mmhg=1.)).solve()
    error=info.value
    assert len(error.history)==1 and error.history[0]['pressure_mmhg']==.5
    assert error.rejected and not torch.equal(error.accepted_x,d.mesh.X)
    torch.testing.assert_close(error.accepted_x,accepted[0],rtol=0,atol=0)


def test_progress_error_is_not_retried_as_numerical_failure(physical_case):
    from afsi_torch.two_way import preload
    d,_=physical_case; calls=[]
    def broken_output(entry): calls.append(entry); raise RuntimeError('injected output failure')
    with pytest.raises(RuntimeError,match='output failure'):
        preload.PassivePreloader(d,preload.PreloadConfig(target_mmhg=.5)).solve(broken_output)
    assert len(calls)==1
