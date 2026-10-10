"""Actual CUDA fixed-volume solve, follower derivative, rollback and restart."""
from dataclasses import replace
import pytest
import torch
from test_afsei_mac_preload import mac_case
from test_real_lv import real_case
from test_afsei_driver import assert_same,physical_arrays
from afsi_torch.ep.mac_isovolumic import IsovolumicAFSEIStepper,IsovolumicProblem,IsovolumicState
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.driver import CoupledStepFailure
from afsi_torch.ep.checkpoint import save,load
from afsi_torch.ep.windkessel import flow_driven_trial
from afsi_torch.ep.tissue import initial_state,TissueState


def precise_driver(base):
    # A force/pressure tensor comparison requires more precision than the
    # production displacement stopping target. Tighten BOTH execution paths
    # here; production presets and acceptance thresholds remain unchanged.
    opt=base.config.fsi.nonlinear
    opt=replace(opt,rtol=1e-10,atol=1e-12,linear=replace(opt.linear,atol=1e-14))
    # The 8-cell/13-cm import fixture has cells larger than this entire LV.
    # Resolve its pressure mobility before testing full pressure/force fields.
    fluid=replace(base.config.fsi.fluid,shape=(32,)*3,lengths=(5.,)*3,origin=(5.,)*3)
    cfg=replace(base.config,fsi=replace(base.config.fsi,nonlinear=opt,fluid=fluid))
    return IsovolumicAFSEIStepper(base.mesh,cfg)


def test_augmented_gpu_jacobian_with_moving_cap(mac_case):
    d=IsovolumicAFSEIStepper.from_driver(mac_case)
    old=d.lock(d.initialize())
    endpoint=d._fsi_stepper(d.maximum_ticks)
    stencil=endpoint.transfer.prepare(old.fsi.x)
    problem=IsovolumicProblem(endpoint,old.fsi,stencil,endpoint.flow.advect(old.fsi.velocity),old)
    z=torch.cat((old.fsi.x.new_zeros(old.fsi.x.numel()),old.fsi.x.new_tensor([2./problem.pressure_scale])))
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype)); v/=v.norm()
    v[-1]=.1
    action=problem.linearization(z)
    eps=2e-6
    fd=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),fd,rtol=2e-4,atol=2e-6)
    load=MACEndocardialPressure(d.model)
    # Include a moving basal rim, independently comparing volume finite differences.
    dx=torch.sin(old.fsi.x)
    derivative=(torch.func.grad(load.volume)(old.fsi.x)*dx).sum()
    finite=(load.volume(old.fsi.x+eps*dx)-load.volume(old.fsi.x-eps*dx))/(2*eps)
    torch.testing.assert_close(derivative,finite,rtol=1e-7,atol=1e-8)


def test_fixed_volume_pressure_rcr_checkpoint_and_rollback(mac_case,tmp_path,monkeypatch):
    d=precise_driver(mac_case)
    tissue=initial_state(d.mesh.X[:,0]); land=tissue.land.clone()
    land[:,0]=1e-4; land[:,1]=1e-4
    old=d.lock(d.initialize(tissue=TissueState(tissue.voltage_v,tissue.gpb,land)))
    copies=[v.clone() for v in physical_arrays(old)]
    new,info=d.step(old)
    assert abs(info['isovolumic']['volume_error_ml'])<1e-3
    assert new.target_volume_ml==old.target_volume_ml
    assert new.accepted_closed_steps==1
    assert new.arterial_pc_mmhg==pytest.approx(flow_driven_trial(85.,0.,info['accepted_dt_s']).pc_mmhg)
    assert new.cavity_pressure_mmhg!=old.cavity_pressure_mmhg
    for a,b in zip(copies,physical_arrays(old)): assert torch.equal(a,b)
    path=tmp_path/'iso.npz'; save(path,d,new)
    restored,restart,_=load(path)
    assert isinstance(restored,IsovolumicAFSEIStepper) and isinstance(restart,IsovolumicState)
    assert restart.target_volume_ml==new.target_volume_ml
    assert restart.cavity_pressure_mmhg==new.cavity_pressure_mmhg
    assert_same(new,restart,exact=True)
    a,_=d.step(new); b,_=restored.step(restart)
    assert_same(a,b)
    assert a.cavity_pressure_mmhg==pytest.approx(b.cavity_pressure_mmhg,rel=2e-7,abs=2e-7)
    def fail(*args,**kwargs): raise RuntimeError('forced fluid failure')
    monkeypatch.setattr(d,'_fsi_stepper',fail)
    with pytest.raises(CoupledStepFailure): d.step(new)
    assert d._iso_context is None
    assert new.target_volume_ml==old.target_volume_ml and new.accepted_closed_steps==1


def test_negative_pressure_restart_is_signed(mac_case,tmp_path):
    d=IsovolumicAFSEIStepper.from_driver(mac_case)
    state=replace(d.lock(d.initialize()),cavity_pressure_mmhg=-2.)
    state=replace(state,ep=replace(state.ep,fsi=replace(state.fsi,force=d.force_for_state(state))))
    path=tmp_path/'negative.npz'; save(path,d,state)
    _,restored,_=load(path)
    assert restored.cavity_pressure_mmhg==-2.


def test_residual_cache_invalidates_changed_unknown(mac_case):
    d=IsovolumicAFSEIStepper.from_driver(mac_case)
    old=d.lock(d.initialize()); endpoint=d._fsi_stepper(d.maximum_ticks)
    stencil=endpoint.transfer.prepare(old.fsi.x)
    problem=IsovolumicProblem(endpoint,old.fsi,stencil,endpoint.flow.advect(old.fsi.velocity),old)
    z=old.fsi.x.new_zeros(old.fsi.x.numel()+1)
    r=problem.residual(z); calls=endpoint.flow.calls
    torch.testing.assert_close(problem.residual(z.clone()),r,rtol=0,atol=0)
    assert endpoint.flow.calls==calls and problem.cache_hits==1
    changed=z.clone(); changed[-1]=1e-5
    problem.residual(changed)
    assert endpoint.flow.calls==calls+1


def test_reference_and_optimized_execution_preserve_step_and_restart(mac_case,tmp_path):
    fast=precise_driver(mac_case)
    ref=IsovolumicAFSEIStepper(fast.mesh,fast.config)
    ref.set_execution(pressure_backend='multigrid',response_optimizations=False)
    tissue=initial_state(fast.mesh.X[:,0]); land=tissue.land.clone()
    land[:,0]=1e-4; land[:,1]=1e-4
    tissue=TissueState(tissue.voltage_v,tissue.gpb,land)
    a=fast.lock(fast.initialize(tissue=tissue)); b=ref.lock(ref.initialize(tissue=tissue))
    for _ in range(2):
        a,ai=fast.step(a); b,bi=ref.step(b)
        assert_same(a,b)
        assert ai['nonlinear']['tolerance']==bi['nonlinear']['tolerance']
        assert ai['nonlinear']['residual_cache_hits']>0
        assert ai['nonlinear']['pressure_direct_solves']>0
        assert ai['nonlinear']['pressure_cycles']==0
        assert a.cavity_pressure_mmhg==pytest.approx(b.cavity_pressure_mmhg,rel=1e-8,abs=1e-8)
    for driver,state in ((fast,a),(ref,b)):
        path=tmp_path/(driver.pressure_backend+'.npz'); save(path,driver,state)
        restored,restart,_=load(path)
        assert restored.execution_settings()==driver.execution_settings()
        assert_same(state,restart,exact=True)
