from dataclasses import replace
import pytest
import torch
from test_three_field_driver import physical_case
from test_real_lv import real_case
from afsi_torch.two_way import ThreeFieldStepper,checkpoint
from afsi_torch.two_way.preload import (ClosedPressure,EndocardialPressure,PassivePreloader,PreloadConfig,
    SourceLoadingHistory,initialize_preloaded)
from afsi_torch.paper_lv import PaperLVSolid
from afsi_torch.ep.land import distortion_step


def test_source_endo_pressure_matches_inherited_pressure_and_follower_derivative(physical_case):
    d,_=physical_case; X=d.mesh.X; x=X+torch.sin(X)*.002; load=EndocardialPressure(d.mesh,d.blood_driver.mesh)
    F,area,_=d.model.geometry_state(x)
    pure=PaperLVSolid.force_from_geometry(d.model,x,F,area,(8*1333.22368421,0.))
    torch.testing.assert_close(d.model.force(x,.5)+load.force(x,8.),pure,rtol=1e-11,atol=1e-8)
    v=torch.sin(3*X); eps=1e-6
    fd=(load.force(x+eps*v,8)-load.force(x-eps*v,8))/(2*eps)
    torch.testing.assert_close(load.action(x,8,v),fd,rtol=2e-8,atol=2e-5)
    torch.testing.assert_close(load.action(x,8,torch.stack((v,2*v),2)),torch.stack((load.action(x,8,v),2*load.action(x,8,v)),2),rtol=1e-11,atol=1e-8)
    # Open-surface endocardial force has a net resultant; closed pressure does not.
    assert load.force(x,8).sum(0).norm()>1e-2
    assert ClosedPressure(d.blood_driver.mesh).force(x,8).sum(0).norm()<1e-8


def test_source_loading_history_samples_only_physical_endpoints(physical_case):
    d,_=physical_case; history=SourceLoadingHistory(d,8,.5,2); X=d.mesh.X
    a={"pressure_mmhg":2.}; history.observe(a,X)
    assert history.accepted==0 and not a['physical_loading_endpoint']
    history.observe({'pressure_mmhg':4.},X*1.01)
    first=history.stretch.clone(); z1=history.zs.clone(); w1=history.zw.clone()
    history.observe({'pressure_mmhg':8.},X*1.02)
    state=history.final()
    torch.testing.assert_close(state.zs,distortion_step(z1,state.stretch-first,.25,40.14),rtol=0,atol=0)
    torch.testing.assert_close(state.zw,distortion_step(w1,state.stretch-first,.25,405.86),rtol=0,atol=0)
    assert state.zs.abs().max()>0 and state.zw.abs().max()>0
    with pytest.raises(ValueError,match='incomplete'): SourceLoadingHistory(d).final()
    with pytest.raises(ValueError,match='skipped'): SourceLoadingHistory(d).observe({'pressure_mmhg':8.},X)


def test_preload_ramp_history_checkpoint_and_endocardial_negative_control(physical_case,tmp_path):
    d,_=physical_case
    d=ThreeFieldStepper(d.mesh,d.config,d.blood_driver.mesh,replace(d.blood_driver.config,atrium_mmhg=8.))
    h=SourceLoadingHistory(d,8,.5,4)
    settings=PreloadConfig(nominal_steps=4)
    result=PassivePreloader(d,settings).solve(observe_state=h.observe)
    assert [e['pressure_mmhg'] for e in result.history if e['physical_loading_endpoint']]==[2.,4.,6.,8.]
    state,report=initialize_preloaded(d,result,strain_history=h.final())
    assert report['loading_history_steps']==4 and report['initial_zs_max_abs']>0
    torch.testing.assert_close(state.ep.force_load.zs_old,state.ep.strain.zs,rtol=0,atol=0)
    checkpoint.save(tmp_path/'ramp.npz',d,state,report)
    restored,new,progress=checkpoint.load(tmp_path/'ramp.npz')
    torch.testing.assert_close(new.ep.strain.zs,state.ep.strain.zs,rtol=0,atol=0)
    torch.testing.assert_close(new.ep.strain.zw,state.ep.strain.zw,rtol=0,atol=0)
    a,_,_=d.step_three_fields(state); b,_,_=restored.step_three_fields(new)
    torch.testing.assert_close(a.ep.fsi.x,b.ep.fsi.x,rtol=3e-7,atol=2e-7)
    torch.testing.assert_close(a.ep.tissue.land,b.ep.tissue.land,rtol=3e-7,atol=2e-7)
    source=PassivePreloader(d,pressure_load=EndocardialPressure(d.mesh,d.blood_driver.mesh)).solve()
    with pytest.raises(RuntimeError,match='actual closed blood reaction'): initialize_preloaded(d,source)


def test_invalid_history_grid():
    with pytest.raises(ValueError): PreloadConfig(nominal_steps=-1)
    with pytest.raises(ValueError): PreloadConfig(nominal_steps=1.5)
    with pytest.raises(ValueError): PreloadConfig(pressure_predictor=1)


def test_pressure_predictor_preserves_equilibrium(physical_case):
    d,_=physical_case
    a=PassivePreloader(d,PreloadConfig(nominal_steps=8,pressure_predictor=True)).solve()
    b=PassivePreloader(d,PreloadConfig(nominal_steps=8,pressure_predictor=False)).solve()
    assert any(e['pressure_predictor_used'] for e in a.history)
    assert not any(e['pressure_predictor_used'] for e in b.history)
    assert a.force_residual_norm_dyn<=a.force_tolerance_dyn and b.force_residual_norm_dyn<=b.force_tolerance_dyn
    torch.testing.assert_close(a.x,b.x,rtol=1e-6,atol=1e-6)
    assert abs(a.volume_ml-b.volume_ml)<1e-6
