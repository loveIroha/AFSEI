"""Prescribed source preload must remain balanced in the unchanged MAC solver."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from test_real_lv import real_case
from test_paper_lv import config_for
from test_afsei_driver import assert_same
from afsi_torch.paper_lv import PaperLVConfig,PaperLVSolid,imported_model
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.driver import AFSEIStepper
from afsi_torch.ep.mac_preload import solve,initialize,MACEndocardialPressure,reuse_source_result
from afsi_torch.ep import checkpoint
from afsi_torch.two_way.preload import PreloadConfig
from afsi_torch.ep.tissue import initial_state
from afsi_torch.circulation.checkpoint import digest


@pytest.fixture
def mac_case(real_case,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    # A MAC preload must not initialize a fitted blood mesh.
    from afsi_torch.moving_cavity import geometry
    from afsi_torch.two_way import preload
    def forbidden(*a,**k): raise AssertionError('fitted blood geometry was constructed')
    monkeypatch.setattr(geometry,'swept_geometry',forbidden)
    monkeypatch.setattr(preload,'swept_geometry',forbidden)
    fsi=replace(config_for(real_case,solver='anderson-newton'),execution=PaperLVConfig().execution)
    cfg=AFSEIConfig.source(fsi=fsi)
    mesh=imported_model(cfg.fsi,'cuda').mesh
    return AFSEIStepper(mesh,cfg)


def test_prescribed_pressure_force_tangent_and_compiled_path(mac_case):
    old=mac_case
    d=AFSEIStepper(old.mesh,replace(old.config,prescribed_endocardial_pressure_mmhg=8.))
    x=d.mesh.X*1.002; F,area,_=d.model.geometry_state(x)
    pressure=MACEndocardialPressure(d.model)
    expected=old.model.force(x,.5)+pressure.force(x,8.)
    torch.testing.assert_close(d.model.force(x,.5),expected,rtol=2e-11,atol=1e-8)
    compiled=d.model.execution_factory()
    torch.testing.assert_close(compiled.force(x,.5),expected,rtol=2e-10,atol=2e-8)
    tangent=d.model.tangent_factory().assemble(x,.5)
    v=torch.sin(x); v/=v.norm(); eps=2e-6
    actual=torch.sparse.mm(tangent,v.reshape(-1,1)).reshape_as(v)
    finite=(d.model.force(x+eps*v,.5)-d.model.force(x-eps*v,.5))/(2*eps)
    torch.testing.assert_close(actual,finite,rtol=4e-6,atol=3e-3)
    assert d.model.loads.at(.5)[1]==0
    with pytest.raises(ValueError): replace(d.config,prescribed_endocardial_pressure_mmhg=-1.)


def test_mac_preload_gates_history_static_drift_and_restart(mac_case,tmp_path):
    d=mac_case
    result,h=solve(d,PreloadConfig(nominal_steps=4,minimum_step_mmhg=.005))
    prepared,state,report=initialize(d,result,h)
    assert report['preload_performed'] and not report['physical_cavity_pressure_initialized']
    assert report['loading_history_steps']==4
    assert report['force_residual_norm_dyn']<=report['force_tolerance_dyn']
    assert report['static_probe_displacement_l2_cm']<=1e-9
    assert report['static_probe_fluid_speed_max_cm_s']<=1e-6
    initial=initial_state(d.mesh.X[:,0])
    for name in ('voltage_v','gpb','land'):
        torch.testing.assert_close(getattr(state.tissue,name),getattr(initial,name),rtol=0,atol=0)
    assert state.fsi.step==0 and state.fsi.pressure.abs().max()==0
    assert all(v.abs().max()==0 for v in state.fsi.velocity)
    assert state.strain.zs.abs().max()>0
    torch.testing.assert_close(state.force_load.zs_old,state.strain.zs,rtol=0,atol=0)
    path=tmp_path/'mac.npz'
    checkpoint.save(path,prepared,state,dict(preload=report))
    restored,restart,progress=checkpoint.load(path)
    assert progress['preload']['preload_performed']
    assert restored.config.prescribed_endocardial_pressure_mmhg==8.
    assert_same(state,restart,exact=True)
    a,info=prepared.step(state); b,_=restored.step(restart)
    assert_same(a,b)
    assert info['ep']['steps']==2
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']


def test_source_import_rejects_wrong_binding_and_fluid_checkpoint(mac_case,tmp_path):
    d=mac_case
    path=tmp_path/'wrong.npz'
    data=dict(reference_x_cm=d.mesh.X.cpu().numpy(),x_cm=d.mesh.X.cpu().numpy(),
        zs=np.zeros(len(d.mesh.X)),zw=np.zeros(len(d.mesh.X)))
    metadata=dict(producer='afsei-source-endo-preload',schema=1,completed=True,
        pressure_mmhg=8.,mesh_binding_sha256='wrong',loading_history=[])
    metadata['sha256']=digest(metadata,data)
    np.savez(path,metadata=json.dumps(metadata),**data)
    with pytest.raises(ValueError,match='mesh'): reuse_source_result(d,path)
    metadata['producer']='afsei-three-fields'
    np.savez(path,metadata=json.dumps(metadata),**data)
    with pytest.raises(ValueError): reuse_source_result(d,path)
