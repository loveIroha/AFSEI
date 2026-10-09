"""Atomic ONE-WAY EP-FSI geometry -> moving blood flow publication.

The existing EP-FSI dynamics still use their inherited exterior IB fluid.
Computed fitted blood traction is exported, NEVER added on top of that fluid.
Full physical two-way FSI requires replacing the mechanical/fluid block and
adding the closed-phase solid volume/pressure constraint (not implemented).
"""
from dataclasses import dataclass,replace
import torch
from ..ep.protocol import ticks
from ..ep.tissue import TissueState
from ..ep.driver import StrainHistory,CoupledStepFailure


def publish_outlet_closure(driver,state):
    """Capture accepted solved Ta from its PRE-publication force history.

    Reconstructing from state.strain would apply the Z update twice. The
    retained force_load is exactly what the accepted mechanical solve used.
    """
    if state.cycle.mode!='land' or state.cycle.resting_land is None:
        raise ValueError('accepted outlet closure requires an active beat and resting Land snapshot')
    driver.buffer.stage(state.force_load)
    anchor=driver.buffer.closure_snapshot(state.fsi.x)
    cycle=replace(state.cycle,mode='relaxation',closure_tick=state.tick,anchor=anchor,freeze_tick=-1,stable_ticks=0)
    tissue=TissueState(state.tissue.voltage_v,state.tissue.gpb,state.cycle.resting_land.clone())
    strain=StrainHistory(state.strain.stretch,torch.zeros_like(state.strain.zs),torch.zeros_like(state.strain.zw))
    return replace(state,cycle=cycle,tissue=tissue,strain=strain)


@dataclass(frozen=True)
class OneWayState:
    ep: object
    blood: object


class OneWayBridge:
    def __init__(self,ep_driver,blood_driver):
        self.ep_driver=ep_driver; self.blood_driver=blood_driver
        if ep_driver.protocol.dt_ep_s!=blood_driver.config.clock_dt_s: raise ValueError('EP/ALE clocks must agree')

    def initialize(self,ep_state):
        fluid=self.blood_driver.initial_state(ep_state.fsi.x)
        if fluid.tick!=ep_state.tick: raise ValueError('EP/ALE initial clocks must agree')
        return OneWayState(ep_state,fluid)

    @torch.no_grad()
    def step(self,old,target_tick=None):
        e=self.ep_driver; f=self.blood_driver
        if old.ep.tick!=old.blood.tick or not torch.equal(old.ep.fsi.x,old.blood.solid_x): raise ValueError('inconsistent one-way accepted state')
        proposed=e.maximum_ticks
        if target_tick is not None:
            if type(target_tick)!=int or target_tick<=old.ep.tick: raise ValueError('target must follow accepted time')
            proposed=min(proposed,target_tick-old.ep.tick)
        proposed=e.protocol.limit(old.ep.tick,proposed,old.ep.cycle)
        attempts=[]
        for _ in range(e.config.max_retries+1):
            try:
                # Filling status is taken only from the previously accepted
                # independent port state; no guessed time/volume events.
                filling=old.blood.opened==(True,False)
                ep,ediag=e.attempt(old.ep,proposed,filling=filling)
                blood,bdiag,force=f.step(old.blood,ep.fsi.x,proposed*e.protocol.dt_ep_s)
                if blood.tick!=ep.tick: raise RuntimeError('one-way endpoint clocks differ')
                at_pace=ep.tick in e.protocol.pulse_ticks
                if bdiag['accepted_outlet_closure'] and old.ep.cycle.mode=='land' and ep.cycle.mode=='land' and not at_pace:
                    ep=publish_outlet_closure(e,ep)
                if at_pace: blood=replace(blood,has_ejected=False)
                ep=replace(ep,rejected_attempts=old.ep.rejected_attempts+len(attempts))
                return OneWayState(ep,blood),dict(ep_fsi=ediag,blood=bdiag,rejected_attempts=len(attempts)),force
            except (ValueError,RuntimeError,FloatingPointError) as exc:
                attempts.append(dict(step_ticks=proposed,error=str(exc))); e._reset_scratch(old.ep)
                if proposed<=1: break
                proposed=max(1,proposed//2); proposed=e.protocol.limit(old.ep.tick,proposed,old.ep.cycle)
        raise CoupledStepFailure('one-way EP-FSI/ALE step rejected; all accepted states retained',attempts)
