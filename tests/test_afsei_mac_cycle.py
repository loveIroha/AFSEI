"""CUDA signed inlet, restart, rollback, and accepted phase/event sequencing.

Event tests exercise publication separately; they do not claim a physical
full-cycle trajectory. Real full-cycle validation is a separate server run.
"""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import balanced_path,dynamic_driver
from test_afsei_driver import assert_same,physical_arrays
from afsi_torch.ep.mac_cycle import CardiacStepper,filling_pressure
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.driver import CoupledStepFailure
from afsi_torch.ep.protocol import CycleState


def driver(base):
    ref=dynamic_driver(base)
    d=CardiacStepper(ref.mesh,ref.config,level=2)
    d.port_geometry=ref.port_geometry; d.pressure_guard_cells=1
    return d


def test_gpu_cycle_contraction_and_restart(mac_case,balanced_path,tmp_path):
    path,_=balanced_path; d=driver(mac_case); old=d.initialize_preload(path)
    new,info=d.step(old)
    assert new.phase=='ISOVOLUMIC_CONTRACTION' and new.beat==1
    assert info['cardiac']['cumulative_mass_error_ml']==pytest.approx(0,abs=1e-7)
    assert new.arterial_pc_mmhg<old.arterial_pc_mmhg
    saved=tmp_path/'cycle.npz'; restart.save(saved,d,new,{'ep_steps':2})
    with pytest.raises(ValueError,match='invalid balanced'): restart.load(saved)
    restored,state,progress=restart.load(saved,cardiac=True)
    assert progress['ep_steps']==2 and restored.end_tick==42000
    assert_same(state,new,exact=True)
    a,_=d.step(new); b,_=restored.step(state); assert_same(a,b)


def test_gpu_signed_filling_and_atomic_restart(mac_case,balanced_path,tmp_path,monkeypatch):
    path,_=balanced_path; d=driver(mac_case); old=d.initialize_preload(path)
    old=replace(old,phase='FILLING',mode='filling',cap_mode='rcr',
        fill_start_tick=old.tick,fill_end_tick=26000,fill_start_pressure_mmhg=8.01,
        ep=replace(old.ep,cycle=replace(old.cycle,mode='zero',closure_tick=old.tick-1600)))
    new,info=d.step(old); r=info['balanced']
    assert r['inlet_flow_ml_s']>0 and r['measured_relative_q_ml_s']==0
    assert abs(r['mass_error_ml_s'])<1e-5 and r['transfer_power_relative_error']<1e-7
    assert r['boundary_pressure_mmhg']==pytest.approx(filling_pressure(old,new.tick),abs=1e-7)
    assert new.target_volume_ml>old.target_volume_ml and new.arterial_pc_mmhg<old.arterial_pc_mmhg
    saved=tmp_path/'fill.npz'; restart.save(saved,d,new)
    restored,state,_=restart.load(saved,cardiac=True)
    a,_=d.step(new); b,_=restored.step(state); assert_same(a,b)
    arrays=[v.clone() for v in physical_arrays(new)]; bp=new.cap_pressure.clone()
    def fail(*args,**kwargs): raise RuntimeError('forced cycle flow failure')
    for n in range(1,d.maximum_ticks+1): monkeypatch.setattr(d._fsi_stepper(n).flow,'advance',fail)
    with pytest.raises(CoupledStepFailure): d.step(new)
    assert d._boundary_context is None and torch.equal(new.cap_pressure,bp)
    for a,b in zip(arrays,physical_arrays(new)): assert torch.equal(a,b)


def test_closure_unload_filling_and_pacing_publication(mac_case,balanced_path,tmp_path,monkeypatch):
    path,_=balanced_path; d=driver(mac_case)
    initial=d.initialize_preload(path,arterial_pc_mmhg=7.9)
    ejected,info=d.step(initial)
    assert ejected.phase=='EJECTION'
    d.buffer.stage(ejected.force_load); anchor=d.buffer.closure_snapshot(ejected.fsi.x).clone()
    def event_info(s):
        r=dict(info['balanced']); r.update(inlet_flow_ml_s=0.,measured_relative_q_ml_s=0.,
            volume_ml=s.target_volume_ml,outlet_open=False)
        return dict(balanced=r,accepted_dt_s=.0001)
    closed,_=d.publish_phase(ejected,replace(ejected,outlet_open=False),event_info(ejected))
    assert closed.phase=='ISOVOLUMIC_RELAXATION' and closed.mode=='closed'
    assert torch.equal(closed.cycle.anchor,anchor)
    assert torch.equal(closed.tissue.land,closed.cycle.resting_land)
    assert closed.strain.zs.abs().max()==0 and closed.strain.zw.abs().max()==0
    assert torch.equal(closed.force_load.xs,ejected.force_load.xs)
    assert closed.arterial_pc_mmhg==ejected.arterial_pc_mmhg
    saved=tmp_path/'closure.npz'; restart.save(saved,d,closed)
    restored,copy,_=restart.load(saved,cardiac=True)
    assert_same(copy,closed,exact=True)
    assert torch.equal(copy.cycle.anchor,anchor) and copy.cap_mode=='rcr' and copy.mode=='closed'
    a,_=d.step(closed); b,_=restored.step(copy); assert_same(a,b)
    assert a.cycle.factor(a.tick,d.protocol)<1 and a.phase=='ISOVOLUMIC_RELAXATION'
    monkeypatch.setattr(d,'inlet_pressure',lambda s:7.5)
    def at(s,tick,cycle):
        time=tick*d.protocol.dt_ep_s
        return replace(s,ep=replace(s.ep,tick=tick,cycle=cycle,
            fsi=replace(s.fsi,time=time,force_time=time,pressure_time=time)))
    early=at(closed,closed.tick+1599,closed.cycle)
    state,_=d.publish_phase(closed,early,event_info(closed))
    assert state.phase=='ISOVOLUMIC_RELAXATION'
    off=at(closed,closed.tick+1600,replace(closed.cycle,mode='zero'))
    fill,_=d.publish_phase(closed,off,event_info(closed))
    assert fill.phase=='FILLING' and fill.fill_start_pressure_mmhg==7.5
    assert filling_pressure(fill,fill.tick)==7.5 and filling_pressure(fill,26000)==8.
    assert filling_pressure(fill,(fill.tick+26000)/2)==7.75
    paced=at(fill,26000,CycleState(mode='land',resting_land=fill.cycle.resting_land))
    second,_=d.publish_phase(fill,paced,event_info(fill))
    assert second.beat==2 and second.phase=='ISOVOLUMIC_CONTRACTION' and second.mode=='rcr'
    assert second.arterial_pc_mmhg==fill.arterial_pc_mmhg
    assert second.target_volume_ml==fill.target_volume_ml and torch.equal(second.tissue.gpb,fill.tissue.gpb)
