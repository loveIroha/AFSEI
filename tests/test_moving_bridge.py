import pytest
import torch
from dataclasses import replace
from test_afsei_driver import case,real_case
from afsi_torch.moving_cavity.bridge import publish_outlet_closure,OneWayBridge,OneWayState
from afsi_torch.moving_cavity import checkpoint
from afsi_torch.moving_cavity import ALEStepper,ALEConfig
from afsi_torch.moving_cavity.geometry import from_solid


def test_accepted_closure_matches_existing_source_event(case):
    driver,old=case
    reference,_=driver.attempt(old,2,closure=True)
    accepted,_=driver.attempt(old,2)
    actual=publish_outlet_closure(driver,accepted)
    assert actual.cycle.closure_tick==reference.cycle.closure_tick
    # Two separate nonlinear solves may differ at FP roundoff; the retained
    # force history must reproduce the same closure load without a Z replay.
    torch.testing.assert_close(actual.cycle.anchor,reference.cycle.anchor,rtol=5e-12,atol=1e-9)
    torch.testing.assert_close(actual.tissue.land,reference.tissue.land,rtol=0,atol=0)
    torch.testing.assert_close(actual.tissue.gpb,reference.tissue.gpb,rtol=0,atol=0)
    assert actual.strain.zs.abs().max()==0 and actual.strain.zw.abs().max()==0
    assert actual.cycle.factor(actual.tick+800,driver.protocol)==.5
    assert actual.cycle.factor(actual.tick+1600,driver.protocol)==0
    with pytest.raises(ValueError,match='active beat'): publish_outlet_closure(driver,actual)


def test_bridge_failure_retains_ep_and_fluid_state(case):
    from types import SimpleNamespace
    from afsi_torch.ep.driver import CoupledStepFailure
    driver,old=case
    class FailedBlood:
        config=SimpleNamespace(clock_dt_s=driver.protocol.dt_ep_s)
        def step(self,*args): raise ValueError('forced ALE failure')
    fluid=SimpleNamespace(tick=old.tick,solid_x=old.fsi.x,opened=(True,True))
    pair=OneWayState(old,fluid); bridge=OneWayBridge(driver,FailedBlood())
    saved=old.tissue.gpb.clone(); saved_x=old.fsi.x.clone(); saved_land=old.tissue.land.clone()
    with pytest.raises(CoupledStepFailure,match='all accepted states retained'): bridge.step(pair)
    assert torch.equal(old.tissue.gpb,saved) and torch.equal(old.fsi.x,saved_x) and torch.equal(old.tissue.land,saved_land)
