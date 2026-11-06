"""New timing uses real GPU stepping/restart; event publication is tested separately."""
from dataclasses import replace
import pytest
from afsi_torch.ep.mac_cycle import CardiacStepper
from afsi_torch.ep.gao_filling import GaoFillingParameters, filling_fraction
from afsi_torch.ep.protocol import CycleState, ticks
from afsi_torch.ep import mac_balanced_checkpoint as restart
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import balanced_path, dynamic_driver
from test_afsei_driver import assert_same


def test_gpu_hold_preserves_waveform_and_pacing_on_both_beats(mac_case, balanced_path, tmp_path, monkeypatch):
    ref = dynamic_driver(mac_case)
    cfg = replace(ref.config, protocol=replace(ref.protocol, pulse_starts_s=(.5, 1.4)))
    d = CardiacStepper(ref.mesh, cfg, level=2, cycles=2, period_s=.9,
        filling_hold_s=.1, filling_model='gao2017',
        gao_filling=GaoFillingParameters(inlet_ramp_fraction=.1))
    d.port_geometry = ref.port_geometry
    d.pressure_guard_cells = 1
    initial = d.initialize_preload(balanced_path[0])
    accepted, info = d.step(initial)
    assert accepted.phase == 'ISOVOLUMIC_CONTRACTION'
    path = tmp_path/'hold.npz'
    restart.save(path, d, accepted)
    restored, copied, _ = restart.load(path, cardiac=True)
    assert restored.protocol.pulse_ticks == (10000, 28000)
    assert restored.filling_hold_ticks == 2000
    assert restored.end_tick == 46000
    assert_same(accepted, copied, exact=True)
    monkeypatch.setattr(d, 'inlet_pressure', lambda s: 0.)

    def at(s, time_ms, phase, mode, cycle, beat=1):
        tick = ticks(time_ms/1000, d.protocol.dt_ep_s)
        time = tick*d.protocol.dt_ep_s
        return replace(s, phase=phase, mode=mode, beat=beat,
            ep=replace(s.ep, tick=tick, cycle=cycle,
                fsi=replace(s.fsi, time=time, force_time=time, pressure_time=time)))

    def event(s):
        r = dict(info['balanced'])
        r.update(inlet_flow_ml_s=0., measured_relative_q_ml_s=0., supply_pressure_mmhg=8.,
            volume_ml=s.target_volume_ml, outlet_open=False)
        return dict(balanced=r, accepted_dt_s=.0001)

    # Synthetic accepted event times test the controller, not a full physical trajectory.
    zero = CycleState(mode='zero', closure_tick=16236, resting_land=accepted.cycle.resting_land)
    off = at(accepted, 891.8, 'ISOVOLUMIC_RELAXATION', 'closed', zero)
    fill, _ = d.publish_phase(off, off, event(off))
    assert fill.fill_start_tick == 17836 and fill.fill_end_tick == 26000
    for ms, expected in [(1095.9, (8., 12.)), (1300., (8., 0.)),
                         (1350., (8., 0.)), (1399.9, (8., 0.))]:
        f = filling_fraction(fill, ticks(ms/1000, d.protocol.dt_ep_s))
        assert d.gao_filling.pressures(f, 8.) == pytest.approx(expected)
    old_boundary = at(fill, 1300., 'FILLING', 'filling', zero)
    still_fill, _ = d.publish_phase(fill, old_boundary, event(fill))
    assert still_fill.phase == 'FILLING' and still_fill.beat == 1
    paced = at(fill, 1400., 'FILLING', 'filling', replace(zero, mode='land'))
    second, _ = d.publish_phase(fill, paced, event(fill))
    assert second.phase == 'ISOVOLUMIC_CONTRACTION' and second.beat == 2
    zero2 = replace(zero, closure_tick=34236)
    off2 = at(second, 1791.8, 'ISOVOLUMIC_RELAXATION', 'closed', zero2, beat=2)
    fill2, _ = d.publish_phase(off2, off2, event(off2))
    assert fill2.fill_end_tick == 44000  # 2200 ms waveform end, 2300 ms run end.
    f = filling_fraction(fill2, ticks(2.25, d.protocol.dt_ep_s))
    assert d.gao_filling.pressures(f, 8.) == (8., 0.)
