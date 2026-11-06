import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from afsi_torch.ep.cycle_diagnostics import CycleDiagnostics


def test_snapshot_preserves_actual_accepted_history(tmp_path,monkeypatch):
    velocity=tuple(torch.full((4,4,4),float(c+1),dtype=torch.float64) for c in range(3))
    x=torch.zeros((4,3),dtype=torch.float64)
    previous=NS(tick=10,fsi=NS(time=.01,x=x,velocity=velocity))
    state=NS(tick=11,fsi=NS(time=.011),cap_geometry=x.clone(),force_load=NS(dt_s=.001))
    def endpoint(interval):
        assert interval==1
        return NS(flow=NS(advect=lambda u:tuple(.9*v for v in u)))
    driver=NS(protocol=NS(dt_ep_s=.001),_fsi_stepper=endpoint)
    observer=CycleDiagnostics(dict(snapshot_times_ms=[11.,13.]),tmp_path,driver,previous,13)
    assert observer.step_target(10,13)==11
    assert observer.step_target(11,13)==13
    monkeypatch.setattr('afsi_torch.ep.cycle_diagnostics.restart.save',
                        lambda path,*args:path.write_bytes(b'test checkpoint'))
    observer.record(driver,previous,state,{},dict(balanced={'mode':'rcr'}))
    with np.load(tmp_path/'diagnostics/tick_00000011/history.npz') as z:
        meta=json.loads(str(z['metadata']))
        assert meta['previous_tick']==10 and meta['tick']==11
        for c in range(3):
            np.testing.assert_array_equal(z[f'previous_velocity_{c}'],velocity[c].numpy())
            np.testing.assert_allclose(z[f'departure_velocity_{c}'],.9*velocity[c].numpy())
    assert len(observer.index)==1
    # A mismatched old state cannot silently produce a misleading acceleration.
    state.cap_geometry=x+1
    with pytest.raises(ValueError,match='previous accepted'):
        observer.record(driver,previous,state,{},dict(balanced={}))


@pytest.mark.parametrize('times',[[10.],[11.,11.],[12.,11.],[14.],[11.2]])
def test_snapshot_schedule_rejects_ambiguous_clock(tmp_path,times):
    driver=NS(protocol=NS(dt_ep_s=.001));state=NS(tick=10)
    with pytest.raises(ValueError):
        CycleDiagnostics(dict(snapshot_times_ms=times),tmp_path,driver,state,13)
