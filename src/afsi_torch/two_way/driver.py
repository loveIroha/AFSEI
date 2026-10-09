"""EP/GPB/Land plus strongly coupled fitted blood and massive myocardium.

Reuses the accepted EP/strain/80-ms unloading protocol. The endpoint adapter
constructs no MAC grid, immersed-boundary transfer, or exterior box fluid.
"""
from dataclasses import dataclass,replace
import torch
from ..ep.driver import AFSEIStepper,CoupledStepFailure
from ..moving_cavity.bridge import publish_outlet_closure
from .mechanics import SolidState,TwoWayMechanics,MechanicalConfig
from .blood import CoupledBlood


@dataclass(frozen=True)
class ThreeFieldState:
    ep: object
    blood: object


class _Endpoint:
    def __init__(self,driver,ticks): self.driver=driver; self.dt=ticks*driver.protocol.dt_ep_s
    def initialize(self,x):
        d=self.driver; time=d.config.start_time_s
        return SolidState(0,time,x.clone(),torch.zeros_like(x),d.model.force(x,time),time,time)
    def physical_step(self,old,time,diagnostics):
        d=self.driver
        s,b,info,force=d.mechanics.step(old,d._trial_blood,self.dt,time)
        info['_blood']=b; info['_interface_force']=force
        return s,info


class ThreeFieldStepper(AFSEIStepper):
    def __init__(self,mesh,ep_config,blood_mesh,blood_config,mechanical_config=None):
        if blood_mesh.reference_solid.shape!=mesh.X.shape or not torch.equal(blood_mesh.reference_solid,mesh.X):
            raise ValueError('blood mapping must refer to the identical solid reference mesh')
        super().__init__(mesh,ep_config)
        self.blood_driver=CoupledBlood(blood_mesh,blood_config)
        self.mechanics=TwoWayMechanics(self.model,self.blood_driver,mechanical_config)
        self._trial_blood=None
        if self.protocol.dt_ep_s!=blood_config.clock_dt_s: raise ValueError('EP/blood clocks differ')

    def _fsi_stepper(self,step_ticks):
        if step_ticks not in self._steppers: self._steppers[step_ticks]=_Endpoint(self,step_ticks)
        return self._steppers[step_ticks]

    def _reset_scratch(self,state):
        self.buffer.stage(state.force_load); self._trial_blood=None

    def initialize_three_fields(self,x=None,tissue=None):
        ep=super().initialize(x,tissue)
        blood=self.blood_driver.initial_state(ep.fsi.x)
        if ep.tick!=blood.tick: raise ValueError('initial clocks differ')
        return ThreeFieldState(ep,blood)

    @torch.no_grad()
    def step_three_fields(self,old,target_tick=None):
        self._check_clock(old.ep)
        if old.ep.tick!=old.blood.tick or not torch.equal(old.ep.fsi.x,old.blood.solid_x): raise ValueError('inconsistent accepted three-field state')
        proposed=self.maximum_ticks
        if target_tick is not None:
            if type(target_tick)!=int or target_tick<=old.ep.tick: raise ValueError('target must follow accepted state')
            proposed=min(proposed,target_tick-old.ep.tick)
        proposed=self.protocol.limit(old.ep.tick,proposed,old.ep.cycle); attempts=[]
        for _ in range(self.config.max_retries+1):
            try:
                self._trial_blood=old.blood
                ep,info=self.attempt(old.ep,proposed,filling=old.blood.opened==(True,False))
                blood=info.pop('_blood'); force=info.pop('_interface_force')
                at_pace=ep.tick in self.protocol.pulse_ticks
                if info['accepted_outlet_closure'] and old.ep.cycle.mode=='land' and ep.cycle.mode=='land' and not at_pace:
                    ep=publish_outlet_closure(self,ep)
                if at_pace: blood=replace(blood,has_ejected=False)
                ep=replace(ep,rejected_attempts=old.ep.rejected_attempts+len(attempts))
                info['rejected_attempts']=len(attempts); self._trial_blood=None
                return ThreeFieldState(ep,blood),info,force
            except (RuntimeError,ValueError,FloatingPointError) as exc:
                attempts.append(dict(ticks=proposed,message=str(exc))); self._reset_scratch(old.ep)
                if proposed<=1: break
                proposed=self.protocol.limit(old.ep.tick,max(1,proposed//2),old.ep.cycle)
        raise CoupledStepFailure('three-field trial rejected; all accepted states retained',attempts)
