"""Phase/controller verification with prescribed data; not a full LV simulation."""
from dataclasses import replace
import pytest
from afsi_torch.ep.lv_circulation import LVCirculation, LVFlowDirectionError
from afsi_torch.ep.windkessel import flow_driven_trial
from afsi_torch.ep.aortic_valve import flow_driven_valved_trial


def accept(m,dt,p=0.,q=0.,**kwargs):
    plan=m.plan(dt,lv_pressure_mmhg=p,**kwargs)
    upstream=None
    if plan.boundary_mode=='ejection':
        _,upstream=flow_driven_valved_trial(plan.state.pc_mmhg,q,plan.dt_s,m.windkessel,m.valve)
    m.commit(plan,outward_flow_ml_s=q,volume_ml=m.state.volume_ml-plan.dt_s*q,upstream_pressure_mmhg=upstream)
    return plan


def preloaded():
    m=LVCirculation(100.)
    a=accept(m,.4)
    assert a.dt_s==pytest.approx(.2) and a.wall_pressure_mmhg==8.
    b=accept(m,.4)
    assert b.dt_s==pytest.approx(.3)
    assert m.state.time_s==pytest.approx(.5) and m.state.pc_mmhg==85.
    return m


def test_preload_stimulus_closed_pc_decay_and_no_pressure_prescription():
    m=preloaded();plan=m.plan(.001,lv_pressure_mmhg=20.)
    assert plan.stimulus and plan.state.beat==1
    assert plan.boundary_mode=='closed' and plan.wall_pressure_mmhg==0.
    assert plan.state.iso_volume_ml==100.
    assert m.state.beat==0  # planning does not advance accepted history
    accept(m,.001,p=20.)
    assert m.state.pc_mmhg==pytest.approx(float(flow_driven_trial(85.,0.,.001,m.windkessel).pc_mmhg))
    assert m.state.pc_mmhg!=20.


def test_ejection_uses_actual_flow_not_lv_pressure_and_pc_continues():
    m=preloaded();plan=accept(m,.001,p=100.,q=10.)
    assert plan.boundary_mode=='ejection'
    assert m.state.volume_ml==pytest.approx(99.99)
    assert m.state.cumulative_out_ml==pytest.approx(.01)
    expected=flow_driven_trial(85.,10.,.001,m.windkessel)
    assert m.state.pc_mmhg==pytest.approx(float(expected.pc_mmhg))


def test_reverse_flow_reject_retry_snapshot_unload_and_next_filling():
    m=preloaded();accept(m,.001,p=100.,q=10.)
    old=m.state;plan=m.plan(.001,lv_pressure_mmhg=100.)
    with pytest.raises(LVFlowDirectionError):m.commit(plan,outward_flow_ml_s=-1.,volume_ml=old.volume_ml+.001)
    assert m.state is old
    closed=accept(m,.001,p=100.,close_outlet=True)
    assert closed.capture_closure_tension and closed.unload_factor==pytest.approx(1-.001/.08)
    assert m.state.iso_volume_ml==old.volume_ml
    last=accept(m,.2,p=5.)
    assert last.dt_s==pytest.approx(.079) and last.unload_factor==pytest.approx(0.)
    assert m.state.phase=='isovolumic_relaxation'
    fill=accept(m,.001,p=5.,q=-2.)
    assert fill.events==('FILLING_START',) and fill.wall_pressure_mmhg==pytest.approx(.04)
    assert m.state.volume_ml==pytest.approx(old.volume_ml+.002)


def test_second_stimulus_after_full_filling_does_not_reset_pc():
    m=preloaded();accept(m,.001,p=100.,q=1.)
    accept(m,.08,p=4.,close_outlet=True)
    accept(m,.2,p=4.)
    accept(m,.3,p=4.)
    previous=m.state.pc_mmhg
    plan=m.plan(.001,lv_pressure_mmhg=20.)
    assert plan.stimulus and plan.state.beat==2
    assert plan.state.pc_mmhg==previous and previous!=85.


def test_closed_flux_mass_error_and_stale_commit_rejected():
    m=preloaded();plan=m.plan(.001,lv_pressure_mmhg=20.);old=m.state
    with pytest.raises(RuntimeError,match='nonzero'):m.commit(plan,outward_flow_ml_s=1.,volume_ml=100.)
    with pytest.raises(RuntimeError,match='mismatch'):m.commit(plan,outward_flow_ml_s=0.,volume_ml=101.)
    assert m.state is old
    m.commit(plan,outward_flow_ml_s=0.,volume_ml=100.)
    with pytest.raises(RuntimeError,match='stale'):m.commit(plan,outward_flow_ml_s=0.,volume_ml=100.)


def test_accumulated_iso_drift_rejected_even_if_each_step_mass_error_small():
    m=preloaded();accept(m,.001,p=20.)
    m.state=replace(m.state,volume_ml=100.0008)
    plan=m.plan(.001,lv_pressure_mmhg=20.)
    with pytest.raises(RuntimeError,match='drift'):m.commit(plan,outward_flow_ml_s=0.,volume_ml=100.0015)


def test_retired_housing_entry_point_creates_no_output(tmp_path):
    from afsi_torch.simulation.lv_dual_port_cycle import run
    target=tmp_path/'not_created'
    with pytest.raises(RuntimeError,match='Retired'):run({},target,device='cpu')
    assert not target.exists()


def test_ejection_requires_actual_pressure_and_source_valve_balance():
    m=preloaded();plan=m.plan(.001,lv_pressure_mmhg=100.);old=m.state
    with pytest.raises(ValueError,match='actual upstream'):
        m.commit(plan,outward_flow_ml_s=10.,volume_ml=99.99)
    with pytest.raises(RuntimeError,match='source valve'):
        m.commit(plan,outward_flow_ml_s=10.,volume_ml=99.99,upstream_pressure_mmhg=100.)
    assert m.state is old
