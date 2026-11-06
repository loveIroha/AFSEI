"""Bound a validation run by accepted physical events, without changing its model."""
from dataclasses import dataclass
from math import isfinite
from .protocol import ticks

EVENTS=('PHYSICAL_AV_OPENING','OUTLET_CLOSURE_TA_SNAPSHOT','GAO_FILLING_ENTRY',
        'SOURCE_WALL_FILLING_ENTRY','PHYSICAL_MV_OPENING','PACING_END_DIASTOLIC_VOLUME')


@dataclass
class EventWindow:
    name: str
    following_ticks: int
    first_tick: int | None=None

    @classmethod
    def from_config(cls,spec,dt_ep_s):
        if not isinstance(spec,dict) or set(spec)!={'name','advance_ms'}:
            raise ValueError('stop_after_event requires name and advance_ms')
        duration=spec['advance_ms']
        if (spec['name'] not in EVENTS or type(duration) not in (float,int)
                or not isfinite(duration) or duration<0):
            raise ValueError('invalid accepted-event validation window')
        return cls(spec['name'],ticks(duration/1000,dt_ep_s))

    def observe(self,tick,events,upper_tick):
        if self.first_tick is None and self.name in events:self.first_tick=tick
        return upper_tick if self.first_tick is None else min(upper_tick,self.first_tick+self.following_ticks)

    def completed(self,tick):
        return self.first_tick is not None and tick>=self.first_tick+self.following_ticks
