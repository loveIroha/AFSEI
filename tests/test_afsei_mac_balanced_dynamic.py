"""Actual GPU active Jacobians, measured RCR, force history and rollback."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import assert_same,physical_arrays
from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper,BalancedState,BalancedDynamicProblem
from afsi_torch.ep.mac_balanced_preload import solve,validate,save_result
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_cap_coupled import closed_rows
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.driver import CoupledStepFailure
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C,flow_driven_trial
from afsi_torch.two_way.preload import PreloadConfig
from afsi_torch.circulation.geometry import PortGeometry


def dynamic_driver(base):
    base = precise_driver(base)
    # Resolve the same physical outputs; avoid an unattainable 1e-12 cap norm
    # after large-source-force cancellation. Independent targets stay fixed.
    opt = replace(base.config.fsi.nonlinear,atol=3e-11,rtol=1e-10)
    d = BalancedAFSEIStepper(base.mesh,replace(base.config,fsi=replace(base.config.fsi,nonlinear=opt)),level=2)
    d.pressure_guard_cells = 1
    d.port_geometry = PortGeometry(mv_center_cm=(.25,0.),av_center_cm=(-.25,0.),mv_radius_cm=.18,av_radius_cm=.15)
    return d


@pytest.mark.parametrize('mode,opened',[('closed',False),('rcr',False),('rcr',True),('filling',False)])
def test_active_gpu_full_joint_jacobian(mac_case,mode,opened):
    d = dynamic_driver(mac_case); d.one_way = not opened
    ep = d.initialize(x=d.mesh.X*1.002)
    # Exercise the nonzero source active tangent, away from constitutive clamps.
    land = ep.tissue.land.clone(); land[:,0]=.001; land[:,1]=.002
    tissue = replace(ep.tissue,land=land)
    load = replace(ep.force_load,xs=land[:,0],xw=land[:,1]); d.buffer.stage(load)
    ep = replace(ep,tissue=tissue,force_load=load)
    e = d._fsi_stepper(d.maximum_ticks); stencil = e.transfer.prepare(ep.fsi.x)
    rows = outlet_rows(e,ep.fsi.x,label=1 if mode=='filling' else 2) if mode!='closed' else stencil.rows
    control = BalancedState(ep,MACEndocardialPressure(d.model).volume(ep.fsi.x).item(),
        torch.full_like(rows.area,8*C),ep.fsi.x.clone(),mode=mode)
    p = BalancedDynamicProblem(e,ep.fsi,stencil,e.flow.advect(ep.fsi.velocity),control)
    z = torch.cat((torch.zeros_like(ep.fsi.x).flatten(),control.cap_pressure/p.cap_scale))
    z[:ep.fsi.x.numel()] = 2e-4*torch.sin(torch.arange(ep.fsi.x.numel(),device=z.device,dtype=z.dtype))
    v = torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype)); v /= v.norm()
    v[-1]=.02
    action = p.linearization(z); eps=2e-6
    assert p.linear_open is opened
    fd = (p.residual(z+eps*v)-p.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),fd,rtol=6e-4,atol=8e-6)
    # Finite-step volume transport uses the integrated P1 gradient rather
    # than the old geometry, also for a nonzero candidate displacement.
    from afsi_torch.ep.mac_transport import ClosedVolumeTransport
    p.residual(z)
    transport = ClosedVolumeTransport.prepare(ep.fsi.x,p.unpack(z)[1],d.model.endo_faces,e.flow.dt,rim=d.model.rim)
    u,nodal = p.last_data[4],p.last_data[7]
    rates = rows.flow(u)-rows.material_flow(nodal)
    identity = transport.rates(nodal).sum()+rates.sum()
    assert abs(identity.item())<1e-7
    assert transport.audit((p.unpack(z)[1]-ep.fsi.x)/e.flow.dt)['swept_geometry_identity_error_ml_s']<1e-7
    # All cap partitions integrate the SAME complete cap for unit pressure.
    full = stencil.rows.lift(torch.full_like(stencil.rows.area,C))
    selected = rows.lift(torch.full_like(rows.area,C))
    for a,b in zip(full,selected): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-8)


@pytest.fixture
def balanced_path(mac_case,tmp_path):
    base = dynamic_driver(mac_case)
    result,h = solve(base,PreloadConfig(nominal_steps=4,minimum_step_mmhg=.005,force_atol_dyn=1e-5,force_rtol=1e-10))
    prepared,state,bp,report = validate(base,result,h,steps=3,level=2,guard_cells=1)
    assert report['static_targets_passed']
    path = tmp_path/'preload.npz'
    save_result(path,prepared,state,bp,report,result.history)
    return path,h


def test_active_preload_checkpoint_replay_and_failure_rollback(mac_case,balanced_path,tmp_path,monkeypatch):
    path,h = balanced_path; d = dynamic_driver(mac_case)
    old = d.initialize_preload(path)
    torch.testing.assert_close(old.strain.zs,h.zs)
    assert old.tick==10000 and old.cycle.mode=='land' and old.force_load.use_land
    arrays = [v.clone() for v in physical_arrays(old)]
    new,info = d.step(old)
    assert info['ep']['steps']==2 and info['ep']['ode_substeps']==50
    assert info['ep']['land_advanced'] and info['balanced']['pressure_core']['mean_mmhg']>7.9
    assert abs(info['balanced']['mass_error_ml_s'])<1e-5
    assert info['balanced']['boundary_work_relative_error']<1e-7
    assert info['balanced']['transfer_power_relative_error']<1e-7
    for a,b in zip(arrays,physical_arrays(old)): assert torch.equal(a,b)
    saved = tmp_path/'dynamic.npz'; restart.save(saved,d,new)
    restored,state,_ = restart.load(saved)
    assert_same(new,state,exact=True)
    torch.testing.assert_close(state.cap_pressure,new.cap_pressure)
    with pytest.raises(ValueError,match='invalid AFSEI'): checkpoint.load(saved)
    a,ia=d.step(new); b,ib=restored.step(state)
    assert_same(a,b)
    assert ia['balanced']['pressure_core']['mean_mmhg']==pytest.approx(ib['balanced']['pressure_core']['mean_mmhg'],abs=1e-7)
    def fail(*args,**kwargs): raise RuntimeError('forced flow failure')
    for n in range(1,d.maximum_ticks+1): monkeypatch.setattr(d._fsi_stepper(n).flow,'advance',fail)
    arrays = [v.clone() for v in physical_arrays(new)]; bp = new.cap_pressure.clone()
    with pytest.raises(CoupledStepFailure): d.step(new)
    assert d._boundary_context is None
    assert torch.equal(new.cap_pressure,bp)
    for a,b in zip(arrays,physical_arrays(new)): assert torch.equal(a,b)


def test_gpu_actual_positive_outlet_flow_rcr_and_restart(mac_case,balanced_path,tmp_path):
    path,_ = balanced_path; d = dynamic_driver(mac_case)
    # Explicit circuit probe; all pressures positive, not a physiological beat.
    old = d.initialize_preload(path,mode='rcr',arterial_pc_mmhg=7.9)
    new,info = d.step(old); r=info['balanced']
    assert r['outlet_open'] and r['measured_relative_q_ml_s']>0
    assert abs(r['mass_error_ml_s'])<1e-5
    assert abs(r['rcr_balance_ml_s'])<1e-8
    assert r['rcr_energy_relative_error']<1e-8
    assert new.arterial_pc_mmhg==pytest.approx(flow_driven_trial(7.9,r['measured_relative_q_ml_s'],info['accepted_dt_s']).pc_mmhg)
    assert new.cap_pressure[-1].item()/C==pytest.approx(new.arterial_pc_mmhg+
        old.windkessel.rc_mmhg_s_per_ml*r['measured_relative_q_ml_s'],abs=1e-7)
    saved = tmp_path/'rcr.npz'; restart.save(saved,d,new)
    restored,state,_ = restart.load(saved)
    a,_=d.step(new); b,_=restored.step(state)
    assert_same(a,b); assert a.arterial_pc_mmhg==pytest.approx(b.arterial_pc_mmhg,abs=1e-8)
