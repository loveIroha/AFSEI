from dataclasses import asdict,replace
from types import SimpleNamespace as NS
import csv,h5py,numpy as np,pytest,torch
from afsi_torch.ep.mac_source_cycle import SourceCycleParameters,SourceLVStepper,filling_pressure,next_nominal_pace
from afsi_torch.ep.protocol import EPProtocol,ticks
from test_source_windkessel_lv import phase_fixture,DummyConfig


def parameters():
    return SourceCycleParameters(fill_rise_s=.5,fill_hold_s=0.,pacing_mode='fixed',
                                 filling_shape='source-smoothstep',stop_time_s=2.)


def test_fixed_linear_preload_and_source_filling():
    p=EPProtocol(pulse_starts_s=(.5,1.35));cfg=parameters()
    initial=NS(phase='PRELOAD',filling_start_tick=0,filling_end_tick=-1,filling_anchor_mmhg=0.)
    for time,goal in [(0.,0.),(.25,4.),(.5,8.)]:
        assert filling_pressure(initial,ticks(time,p.dt_ep_s),p,cfg)==pytest.approx(goal)
    state=NS(phase='FILLING',filling_start_tick=ticks(.85,p.dt_ep_s),
             filling_end_tick=ticks(1.35,p.dt_ep_s),filling_anchor_mmhg=-.1)
    for time,fraction in [(.85,0.),(.975,.25),(1.1,.5),(1.225,.75),(1.35,1.)]:
        # Independent direct copy of source lv_cycle.h expression.
        goal=-.1+8.1*fraction*fraction*(3.-2.*fraction)
        assert filling_pressure(state,ticks(time,p.dt_ep_s),p,cfg)==pytest.approx(goal)
    assert next_nominal_pace(ticks(.85,p.dt_ep_s),p)==ticks(1.35,p.dt_ep_s)
    assert next_nominal_pace(ticks(1.7,p.dt_ep_s),p)==ticks(2.2,p.dt_ep_s)


def test_fixed_fill_entry_keeps_anchor_and_pacing(monkeypatch):
    d,old=phase_fixture(monkeypatch,'IVR',16600,16602,5.,mode='zero',closure_tick=15000)
    d.circulation=parameters();d.protocol=EPProtocol(pulse_starts_s=(.5,1.35));d.config=DummyConfig(d.protocol)
    new,_=d.step(old)
    assert new.phase=='FILLING' and new.filling_anchor_mmhg==5.
    assert new.filling_end_tick==27000 and d.protocol.pulse_starts_s==(.5,1.35)
    assert not d.completed(NS(beat=2,tick=39999))
    assert d.completed(NS(beat=2,tick=40000))


from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from afsi_torch.ep.active_recording import ActiveStressRecorder
from afsi_torch.ep.checkpoint import save,load


def test_gpu_recorder_and_fixed_checkpoint(mac_case,tmp_path):
    base=precise_driver(mac_case)
    cfg=replace(base.config,start_time_s=0.,protocol=EPProtocol(pulse_starts_s=(.5,1.35)))
    d=SourceLVStepper(base.mesh,cfg);d.set_execution(circulation=asdict(parameters()))
    old=d.initialize();new,info=d.step(old)
    assert new.cavity_pressure_mmhg==pytest.approx(8*new.fsi.time/.5)
    path=tmp_path/'restart.npz';save(path,d,new);restored,again,_=load(path)
    assert restored.circulation==d.circulation and again.filling_anchor_mmhg==new.filling_anchor_mmhg
    # Manufactured nonzero accepted-load buffer for a diagnostic-only recording test.
    field=torch.full_like(old.force_load.xs,.002)
    d.buffer.stage(replace(old.force_load,xs=field,use_land=True))
    expected=d.buffer.tension(old.fsi.x).clone();expected_force=d.buffer.force(old.fsi.x).clone()
    rec=ActiveStressRecorder(tmp_path,d,dict(statistics_every_steps=1,fields_every_steps=1))
    rec.record(old,force=True);rec.record(old,force=True);rec.close()
    torch.testing.assert_close(d.buffer.tension(old.fsi.x),expected,rtol=0,atol=0)
    with h5py.File(tmp_path/'active_load_history.h5') as f:
        assert len(f['time_s'])==1
        np.testing.assert_array_equal(f['Ta_dyn_cm2'][0],expected.cpu().numpy())
        # Repeated GPU nodal force scatter may reorder roundoff-level additions.
        np.testing.assert_allclose(f['active_force_dyn'][0],expected_force.cpu().numpy(),rtol=2e-12,atol=1e-12)
        phi=f['transmural_phi'][:]
        assert np.all(phi[torch.unique(d.mesh.surface(1)).cpu().numpy()]==0)
        assert np.all(phi[torch.unique(d.mesh.surface(2)).cpu().numpy()]==1)
    rows=list(csv.DictReader((tmp_path/'active_tension_layers.csv').open()))
    assert len(rows)==1 and float(rows[0]['middle_Ta_mean_kpa'])>0
