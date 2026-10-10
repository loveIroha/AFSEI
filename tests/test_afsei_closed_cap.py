"""CUDA coupled reaction adjoint, Jacobian, acceptance, restart and rollback."""
from dataclasses import replace
import pytest
import torch
from test_afsei_mac_preload import mac_case
from test_real_lv import real_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import assert_same,physical_arrays
from afsi_torch.ep.mac_cap_coupled import CapIsovolumicProblem,closed_rows
from afsi_torch.ep.driver import CoupledStepFailure
from afsi_torch.ep.tissue import initial_state,TissueState
from afsi_torch.ep.checkpoint import save,load
from afsi_torch.mac.open_boundary import face_weights


def case(base):
    d=precise_driver(base); d.set_execution(closed_cap=True,cap_patch_width_cells=3.)
    tissue=initial_state(d.mesh.X[:,0]); land=tissue.land.clone()
    land[:,0]=land[:,1]=1e-4
    state=d.lock(d.initialize(tissue=TissueState(tissue.voltage_v,tissue.gpb,land)))
    endpoint=d._fsi_stepper(d.maximum_ticks)
    stencil=endpoint.transfer.prepare(state.fsi.x)
    problem=CapIsovolumicProblem(endpoint,state.fsi,stencil,
        endpoint.flow.advect(state.fsi.velocity),state)
    return d,state,endpoint,problem


def test_relative_adjoint_and_augmented_jacobian(mac_case):
    d,old,e,p=case(mac_case)
    assert len(p.rows.selected)==len(p.rows.prepared.cap.labels)
    u=tuple(torch.cos(e.grid.coordinates(c,device='cuda',dtype=torch.float64)[...,c]) for c in range(3))
    w,_=e.transfer.interpolate(u,p.stencil)
    pressure=torch.sin(torch.arange(p.rows.count,device='cuda',dtype=torch.float64))
    density=p.cap_density(pressure)
    power=sum((a*b*face_weights(e.grid,c,old.fsi.x)).sum() for c,(a,b) in enumerate(zip(u,density)))
    expected=-(pressure*(p.rows.flow(u)-p.rows.material_flow(w))).sum()
    torch.testing.assert_close(power,expected,rtol=1e-10,atol=1e-10)
    z=old.fsi.x.new_zeros(old.fsi.x.numel()+p.extra_unknowns)
    # Avoid the source PK1 clamp kink at reference fiber stretch exactly 1:
    # a centred finite difference there selects a different subgradient.
    z[:old.fsi.x.numel()]=(.002*old.fsi.x).flatten()
    z[-1]=1e-7
    v=torch.sin(torch.arange(len(z),device=z.device,dtype=z.dtype)); v/=v.norm()
    v[-p.extra_unknowns:]*=1e-2
    action=p.linearization(z); eps=2e-6
    fd=(p.residual(z+eps*v)-p.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),fd,rtol=3e-4,atol=3e-6)


def test_reaction_checkpoint_restart_and_failed_trial(mac_case,tmp_path,monkeypatch):
    d,old,_,_=case(mac_case)
    copies=[v.clone() for v in physical_arrays(old)]
    new,info=d.step(old,diagnostics=True)
    assert info['cap']['constraint_residual_norm_ml_s']<1e-5
    assert info['cap']['coupled_solid_advanced'] and not info['cap']['prescribed_material_motion']
    assert info['cap']['reaction_norm_dyn']>0
    assert info['cap']['cap_power_relative_error']<1e-7
    assert abs(info['isovolumic']['volume_error_ml'])<1e-3
    assert new.cap_pressure is not None
    torch.testing.assert_close(new.fsi.force,d.force_for_state(new),rtol=1e-9,atol=1e-7)
    for a,b in zip(copies,physical_arrays(old)): assert torch.equal(a,b)
    path=tmp_path/'cap.npz'; save(path,d,new)
    restored,restart,_=load(path)
    assert restored.closed_cap
    assert_same(new,restart,exact=True)
    torch.testing.assert_close(new.cap_pressure,restart.cap_pressure,rtol=0,atol=0)
    a,_=d.step(new); b,_=restored.step(restart)
    assert_same(a,b)
    def fail(*args,**kwargs): raise RuntimeError('forced cap trial failure')
    monkeypatch.setattr(d,'_fsi_stepper',fail)
    with pytest.raises(CoupledStepFailure): d.step(new)
    assert d._iso_context is None
    torch.testing.assert_close(new.cap_pressure,restart.cap_pressure,rtol=0,atol=0)


def test_preconditioner_overlap_remaps_permuted_material_patches(mac_case):
    d,old,e,p=case(mac_case)
    original_b=p.cap_b.clone(); original_h=p.cap_h.clone()
    n=p.rows.count
    assert n>1
    p.rows=replace(p.rows,patches=n-1-p.rows.patches,area=p.rows.area.flip(0),
        diagonal=p.rows.diagonal.flip(0),
        components=tuple((n-1-r,i,a) for r,i,a in p.rows.components))
    p.cap_scale=p.cap_scale.flip(0)
    before=e.flow.pressure_solver.direct_solves
    p._prepare_preconditioner()
    assert e.flow.pressure_solver.direct_solves==before
    torch.testing.assert_close(p.cap_b,original_b.flip(1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(p.cap_h,original_h.flip((0,1)),rtol=1e-12,atol=1e-12)
