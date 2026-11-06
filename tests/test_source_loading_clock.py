from types import SimpleNamespace
import os,math
import pytest
import torch
from afsi_torch.two_way.preload import SourceLoadingHistory
from afsi_torch.ep.cycle_policy import FillingClock
from afsi_torch.ep.gao_filling import GaoFillingParameters


def driver():
    device=os.environ.get('AFSEI_TEST_DEVICE','cpu')
    X=torch.ones(2,3,dtype=torch.float64,device=device)
    fiber=X.new_tensor([[1.,0.,0.]])
    calls=[]
    def project(v):calls.append(1);return v.repeat(2)
    d=SimpleNamespace(mesh=SimpleNamespace(X=X,fiber=fiber),
        model=SimpleNamespace(element_gradient=lambda x:torch.eye(3,device=device,dtype=X.dtype)[None]*x[0,0]),
        ep=SimpleNamespace(project_cell_field=project))
    return d,calls


def test_first_200ms_ramp_then_300ms_hold_has_correct_clock_and_analytic_distortion_decay():
    d,calls=driver();h=SourceLoadingHistory(d,8.,.5,100,ramp_s=.2);entries=[]
    for i in range(1,41):
        e={'pressure_mmhg':.2*i};h.observe(e,d.mesh.X*(1+.002*i));entries.append(e)
    zs,zw=h.zs.clone(),h.zw.clone();x=h.x.clone()
    for i in range(60):
        e={'pressure_mmhg':8.};h.observe_held_equilibrium(e);entries.append(e)
    result=h.final()
    assert len(calls)==40 and h.accepted==100
    assert entries[39]['physical_loading_time_s']==.2
    assert entries[-1]['physical_loading_time_s']==.5
    assert all(e['pressure_mmhg']==8. for e in entries[39:])
    torch.testing.assert_close(result.zs,zs*math.exp(-40.14*.3),rtol=1e-13,atol=1e-16)
    torch.testing.assert_close(result.zw,zw*math.exp(-405.86*.3),rtol=1e-13,atol=1e-65)
    assert torch.equal(result.x,x)


def test_hold_cannot_start_before_ramp_or_continue_beyond_500ms():
    d,_=driver();h=SourceLoadingHistory(d,ramp_s=.2)
    with pytest.raises(ValueError,match='completed ramp'):h.observe_held_equilibrium({'pressure_mmhg':8.})
    for i in range(1,41):h.observe({'pressure_mmhg':i*.2},d.mesh.X)
    for i in range(60):h.observe_held_equilibrium({'pressure_mmhg':8.})
    with pytest.raises(ValueError):h.observe_held_equilibrium({'pressure_mmhg':8.})


def test_source_wall_filling_is_200ms_ramp_and_300ms_hold_without_inlet_pressure():
    c=FillingClock(duration_s=.2,inlet_ramp_s=.2,wall_peak_s=.1,hold_s=.3,
                   loading_mode='source-wall',pacing_mode='after-filling')
    p=GaoFillingParameters(wall_peak_mmhg=12.)
    for t,wall in [(0.,0.),(.1,4.),(.2,8.),(.5,8.)]:assert c.pressures(t,p,8.)==(0.,wall)


def test_next_stimulus_occurs_exactly_after_200ms_ramp_and_300ms_hold():
    from test_cut_cycle_upgrade import timing_driver
    d=timing_driver()
    d.filling_clock=FillingClock(duration_s=.2,inlet_ramp_s=.2,wall_peak_s=.1,hold_s=.3,
        loading_mode='source-wall',pacing_mode='after-filling')
    end,changed=d.reserve_filling_window(SimpleNamespace(tick=17836,beat=1))
    assert changed and end==21836
    assert d.protocol.pulse_ticks==(10000,27836)
    end,_=d.reserve_filling_window(SimpleNamespace(tick=43666,beat=2))
    assert d.end_tick==53666 and end==47666


def test_wall_ramp_starts_at_unloading_end_without_waiting_for_a_reservoir_pressure_crossing():
    from dataclasses import replace
    from test_cut_cycle_policy import case,diagnostic
    from test_cut_cycle_upgrade import Config
    d,s=case();d.config=Config(d.protocol)
    d.filling_clock=FillingClock(duration_s=.2,inlet_ramp_s=.2,wall_peak_s=.1,hold_s=.3,
        loading_mode='source-wall',pacing_mode='after-filling')
    d.inlet_pressure=lambda state:20.
    zero=replace(s.cycle,mode='zero',closure_tick=16236)
    off=replace(s,ep=replace(s.ep,tick=17836,cycle=zero),phase='ISOVOLUMIC_RELAXATION',mode='closed')
    fill,info=d.publish_cut_phase(off,off,diagnostic())
    assert fill.phase=='FILLING' and fill.fill_start_tick==17836 and fill.fill_end_tick==21836
    assert fill.mode=='closed' and 'SOURCE_WALL_FILLING_ENTRY' in info['cardiac']['events']
