"""Actual original-MAC relative RCR closure, work, one-way and pure trials."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.ep.mac_rcr_boundary import MACRCRBoundaryResponse,MACRCRBoundaryOptions
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C,flow_driven_trial
from afsi_torch.circulation.geometry import PortGeometry
from afsi_torch.mac.open_boundary import face_weights


@pytest.fixture
def boundary(mac_case):
    d=precise_driver(mac_case); e=d._fsi_stepper(d.maximum_ticks)
    X=d.mesh.X; stencil=e.transfer.prepare(X)
    # Generated regression fixture has a smaller basal rim than the real LV.
    ports=PortGeometry(mv_center_cm=(.25,0.),av_center_cm=(-.25,0.),
        mv_radius_cm=.18,av_radius_cm=.15)
    cap=MACBasalPorts(X,d.model.endo_faces,e.grid,ports=ports,subdivision_level=5)
    solver=MACRCRBoundaryResponse(e.flow,e.transfer,stencil,cap.prepare(X),geometry=X)
    return solver,e


def drive(solver,pressure_mmhg):
    pressure=solver.like.new_zeros(solver.count); pressure[-1]=-pressure_mmhg*C
    return solver.boundary_density(pressure)


def test_actual_relative_operator_and_paired_reaction(boundary):
    s,e=boundary
    u=tuple(torch.sin(e.grid.coordinates(c,device=s.like.device)[...,c]+1.) for c in range(3))
    w=s.interpolate(u)
    p=torch.cos(torch.arange(s.count,device=s.like.device,dtype=s.like.dtype))*9100
    density=s.boundary_density(p); reaction=s.reaction(p)
    # boundary_density includes S(reaction), so its TOTAL effective power
    # equals -p*(B u-C Ju). It must not be counted again on top of this.
    effective=sum((a*b*face_weights(s.grid,c,s.like)).sum() for c,(a,b) in enumerate(zip(u,density)))
    torch.testing.assert_close(effective,-(p*s.relative_rates(u,w)).sum(),rtol=1e-7,atol=1e-5)
    # Separate fluid-only B load and its positive material reaction.
    port=s.like.new_zeros(3); port[2]=p[-1]
    lift=s.closed.lift(p[:-1]); out=s.prepared.rows.pressure_density(port)
    fluid_only=tuple(-a+b for a,b in zip(lift,out))
    power=sum((a*b*face_weights(s.grid,c,s.like)).sum() for c,(a,b) in enumerate(zip(u,fluid_only)))
    torch.testing.assert_close(power+(reaction*w).sum(),-(p*s.relative_rates(u,w)).sum(),rtol=1e-7,atol=1e-5)


def test_open_closed_and_signed_flow_without_source_state_change(boundary):
    s,e=boundary; old_pc=85.; zero=s.grid.zeros(device=s.like.device)
    copies=[v.clone() for v in (s.geometry,*zero)]
    high=drive(s,90.)
    result=s.advance(zero,high,old_pc)
    r=result.report
    assert r['outlet_open'] and r['measured_relative_q_ml_s']>0
    assert r['open_pressure_law_error_mmhg']<1e-7
    assert r['closed_region_flux_norm_ml_s']<1e-5
    assert abs(r['rcr_balance_ml_s'])<1e-9
    assert r['rcr_energy_relative_error']<1e-9
    assert r['relative_work_relative_error']<1e-7
    expected=flow_driven_trial(old_pc,r['measured_relative_q_ml_s'],e.flow.dt,s.parameters)
    assert r['pc_mmhg']==pytest.approx(expected.pc_mmhg,abs=1e-12)
    s.options=replace(s.options,valve_resistance_mmhg_s_per_ml=.02)
    resistive=s.advance(zero,high,old_pc).report
    assert 0<resistive['measured_relative_q_ml_s']<r['measured_relative_q_ml_s']
    assert resistive['boundary_pressure_mmhg']-resistive['rcr_outlet_mmhg']==pytest.approx(
        .02*resistive['measured_relative_q_ml_s'],abs=1e-7)
    assert resistive['rcr_outlet_mmhg']-resistive['pc_mmhg']==pytest.approx(
        s.parameters.rc_mmhg_s_per_ml*resistive['measured_relative_q_ml_s'],abs=1e-7)
    s.options=replace(s.options,valve_resistance_mmhg_s_per_ml=0.)
    low=drive(s,80.)
    shut=s.advance(zero,low,old_pc)
    assert not shut.report['outlet_open']
    assert abs(shut.report['measured_relative_q_ml_s'])<1e-5
    assert shut.report['rcr_input_q_ml_s']==0.
    assert shut.report['valve_gap_mmhg']>0
    assert shut.report['pc_mmhg']<old_pc
    s.options=replace(s.options,one_way=False)
    reverse=s.advance(zero,low,old_pc)
    assert reverse.report['outlet_open'] and reverse.report['measured_relative_q_ml_s']<0
    for a,b in zip((s.geometry,*zero),copies): assert torch.equal(a,b)


def test_trial_replay_direct_residual_and_changed_operator_rejected(boundary,monkeypatch):
    s,e=boundary; zero=s.grid.zeros(device=s.like.device); source=drive(s,90.)
    a=s.advance(zero,source,85.); b=s.advance(zero,source,85.)
    for u,v in zip(a.velocity,b.velocity): torch.testing.assert_close(u,v,rtol=2e-12,atol=2e-14)
    torch.testing.assert_close(a.pc_mmhg,b.pc_mmhg,rtol=0,atol=2e-12)
    full=torch.cat((a.cap_pressure_cgs,a.outlet_pressure_cgs.reshape(1)))
    density=s.boundary_density(full)
    u,_,_=e.flow.advance(zero,tuple(x+y for x,y in zip(source,density)))
    for x,y in zip(u,a.velocity): torch.testing.assert_close(x,y,rtol=2e-12,atol=2e-14)
    def failure(*args,**kwargs): raise RuntimeError('forced true fluid solve failure')
    old=a.pc_mmhg.clone()
    monkeypatch.setattr(e.flow,'advance',failure)
    with pytest.raises(RuntimeError,match='forced'): s.advance(zero,source,a.pc_mmhg)
    torch.testing.assert_close(old,a.pc_mmhg,rtol=0,atol=0)
    e.flow.dt*=.5
    with pytest.raises(RuntimeError,match='rebuild'): s.advance(zero,source,85.)


def test_invalid_boundary_options():
    with pytest.raises(ValueError): MACRCRBoundaryOptions(patch_width_cells=1.)
    with pytest.raises(ValueError): MACRCRBoundaryOptions(valve_resistance_mmhg_s_per_ml=-1.)
    with pytest.raises(ValueError): MACRCRBoundaryOptions(one_way=1)
