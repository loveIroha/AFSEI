"""GPU moving-solid RCR Jacobian, true flow balance, rollback and restart."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import assert_same, physical_arrays
from afsi_torch.ep.mac_rcr_coupled import RCRAFSEIStepper,RCRCoupledProblem,RCRState,outlet_rows
from afsi_torch.ep.mac_isovolumic import IsovolumicAFSEIStepper
from afsi_torch.ep.windkessel import flow_driven_trial,MMHG_TO_DYN_CM2 as C
from afsi_torch.ep.tissue import initial_state,TissueState
from afsi_torch.ep.driver import CoupledStepFailure
from afsi_torch.ep import checkpoint
from afsi_torch.circulation.geometry import PortGeometry


def driver_for(mac_case):
    d = RCRAFSEIStepper.from_driver(precise_driver(mac_case))
    d.port_geometry = PortGeometry(mv_center_cm=(.25,0.),av_center_cm=(-.25,0.),
                                  mv_radius_cm=.18,av_radius_cm=.15)
    return d


@pytest.mark.parametrize('opened',[False,True])
def test_full_gpu_jacobian_closed_and_open(mac_case,opened):
    # Move away from the passive fibre/sheet tension-only kink at F=I;
    # centred differences there compare two constitutive branches.
    d=driver_for(mac_case); old=d.connect(d.initialize(x=d.mesh.X*1.002))
    if opened: d.one_way=False  # Exercise the differentiable open law away from its switch.
    e=d._fsi_stepper(d.maximum_ticks); stencil=e.transfer.prepare(old.fsi.x)
    problem=RCRCoupledProblem(e,old.fsi,stencil,e.flow.advect(old.fsi.velocity),old)
    z=old.fsi.x.new_zeros(old.fsi.x.numel()+problem.extra_unknowns)
    z[old.fsi.x.numel()]=2./problem.pressure_scale
    if opened: z[-1]=120.*C/problem.cap_scale[-1]
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype)); v/=v.norm()
    v[old.fsi.x.numel()]=.1; v[-1]=.02
    action=problem.linearization(z)
    assert problem.linear_open is opened
    eps=2e-6
    fd=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),fd,rtol=4e-4,atol=3e-6)
    assert old.arterial_pc_mmhg==85. and old.accepted_port_steps==0


def test_coupled_balance_checkpoint_replay_and_rollback(mac_case,tmp_path,monkeypatch):
    d=driver_for(mac_case); old=d.connect(d.initialize())
    copies=[v.clone() for v in physical_arrays(old)]
    new,info=d.step(old)
    r=info['rcr']
    assert not r['outlet_open'] and abs(r['measured_relative_q_ml_s'])<1e-5
    assert abs(r['mass_error_ml_s'])<1e-5
    assert abs(r['rcr_balance_ml_s'])<1e-9
    assert new.arterial_pc_mmhg==pytest.approx(flow_driven_trial(85.,r['measured_relative_q_ml_s'],info['accepted_dt_s']).pc_mmhg)
    assert new.accepted_port_steps==1 and new.boundary_history=='rcr'
    assert info['cap']['cap_power_relative_error']<1e-7
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    for a,b in zip(copies,physical_arrays(old)): assert torch.equal(a,b)
    path=tmp_path/'rcr.npz'; checkpoint.save(path,d,new)
    restored,restart,_=checkpoint.load(path)
    assert isinstance(restored,RCRAFSEIStepper) and isinstance(restart,RCRState)
    assert_same(new,restart,exact=True)
    assert new.arterial_pc_mmhg==restart.arterial_pc_mmhg
    assert torch.equal(new.cap_pressure,restart.cap_pressure)
    a,ia=d.step(new); b,ib=restored.step(restart)
    assert_same(a,b)
    assert ia['rcr']['measured_relative_q_ml_s']==pytest.approx(ib['rcr']['measured_relative_q_ml_s'],abs=1e-6)
    assert a.arterial_pc_mmhg==pytest.approx(b.arterial_pc_mmhg,abs=1e-9)
    def failure(*args,**kwargs): raise RuntimeError('forced post-trial rejection')
    monkeypatch.setattr(d._fsi_stepper(d.maximum_ticks).flow,'advance',failure)
    for ticks in range(1,d.maximum_ticks):
        monkeypatch.setattr(d._fsi_stepper(ticks).flow,'advance',failure)
    copies=[v.clone() for v in physical_arrays(new)]
    with pytest.raises(CoupledStepFailure): d.step(new)
    assert d._iso_context is None and new.accepted_port_steps==1
    assert new.arterial_pc_mmhg==restart.arterial_pc_mmhg
    for a,b in zip(copies,physical_arrays(new)): assert torch.equal(a,b)


def test_open_interval_uses_measured_flow_not_volume_surrogate(mac_case):
    d=driver_for(mac_case)
    tissue=initial_state(d.mesh.X[:,0]); land=tissue.land.clone()
    land[:,0]=.01; land[:,1]=.01
    old=d.connect(d.initialize(tissue=TissueState(tissue.voltage_v,tissue.gpb,land)))
    # Explicit circuit-condition regression probe, not a physiological beat.
    old=replace(old,arterial_pc_mmhg=-100.)
    new,info=d.step(old); r=info['rcr']
    assert r['outlet_open'] and r['measured_relative_q_ml_s']>0, r
    assert r['volume_ml']<r['volume_old_ml']
    assert abs(r['mass_error_ml_s'])<1e-5
    assert r['open_pressure_law_error_mmhg']<1e-7
    assert r['rcr_energy_relative_error']<1e-9
    assert r['boundary_pressure_mmhg']-r['arterial_pc_mmhg']==pytest.approx(
        old.windkessel.rc_mmhg_s_per_ml*r['measured_relative_q_ml_s'],abs=1e-7)
    assert new.arterial_pc_mmhg==pytest.approx(flow_driven_trial(-100.,r['measured_relative_q_ml_s'],info['accepted_dt_s']).pc_mmhg)
    assert new.target_volume_ml!=old.target_volume_ml


def test_closed_history_transition_retained_on_save(mac_case,tmp_path):
    iso=precise_driver(mac_case); iso.closed_cap=True
    old=iso.lock(iso.initialize()); old,_=iso.step(old)
    d=driver_for(iso); connected=d.connect(old)
    assert connected.boundary_history=='closed'
    torch.testing.assert_close(d.force_for_state(connected),old.fsi.force,rtol=1e-9,atol=1e-7)
    path=tmp_path/'transition.npz'; checkpoint.save(path,d,connected)
    _,restored,_=checkpoint.load(path)
    assert restored.boundary_history=='closed' and torch.equal(restored.cap_pressure,old.cap_pressure)


def test_ordinary_resume_retains_rcr_mode_and_reports_actual_flow(mac_case,tmp_path):
    from afsi_torch.simulation.afsei import run
    d=driver_for(mac_case); old=d.connect(d.initialize())
    state,_=d.step(old)
    path=tmp_path/'rcr.npz'
    checkpoint.save(path,d,state,dict(ep_steps=2,ode_substeps=50,elapsed_seconds=0.))
    report=run(resume=path,output=tmp_path/'resume',duration_s=.0002,profile=True)
    assert report['status']=='completed'
    assert report['cavity_mode']=='mac-rcr-outlet-interval-validation'
    assert report['windkessel_ports_connected']
    assert report['rcr_boundaries_in_same_implicit_fsi_solve']
    assert not report['assumed_closed_outlet_for_0d_update']
    assert not report['isovolumic_constraint_implemented']
    assert not report['three_field_integration_implemented']
    assert abs(report['last_state']['rcr_mass_error_ml_s'])<1e-5
    _,restored,_=checkpoint.load(tmp_path/'resume'/'checkpoint.npz')
    assert restored.accepted_port_steps==state.accepted_port_steps+2
    with pytest.raises(ValueError,match='phase controller'):
        run(resume=path,output=tmp_path/'invalid',duration_s=.0001,isovolumic=True)
