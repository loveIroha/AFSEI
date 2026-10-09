"""Integer EP ticks, hybrid loading and source-aligned recovery controls.

Outlet closure/filling events must come from a physical circulation/port
solver. This module never invents valve events from a time or volume curve.
"""
from dataclasses import dataclass
from math import isfinite
import torch


def ticks(seconds, dt_ep_s):
    value = seconds/dt_ep_s
    result = round(value)
    if not isfinite(value) or abs(value-result) > 1e-7:
        raise ValueError('event time must lie on the integer EP grid')
    return result


@dataclass(frozen=True)
class EPProtocol:
    dt_ep_s: float = 5e-5
    pulse_starts_s: tuple = (.5,1.3)
    pulse_width_s: float = .002
    amplitude_ua_per_cm2: float = 25.
    relaxation_s: float = .080
    freeze_delay_s: float = .200
    stable_s: float = .020
    vm_tolerance_mv: float = 1.
    calcium_rest_relative_tolerance: float = .10
    calcium_drift_relative_tolerance: float = .01
    freeze_enabled: bool = True

    def __post_init__(self):
        object.__setattr__(self,'pulse_starts_s',tuple(self.pulse_starts_s))
        positive = (self.dt_ep_s,self.pulse_width_s,self.amplitude_ua_per_cm2,self.relaxation_s,
                    self.freeze_delay_s,self.stable_s,self.vm_tolerance_mv,
                    self.calcium_rest_relative_tolerance,self.calcium_drift_relative_tolerance)
        if (any(not isfinite(v) or v <= 0 for v in positive) or not self.pulse_starts_s
                or any(not isfinite(v) or v < 0 for v in self.pulse_starts_s)
                or tuple(sorted(set(self.pulse_starts_s))) != self.pulse_starts_s
                or type(self.freeze_enabled) is not bool):
            raise ValueError('invalid EP pacing/recovery controls')
        for t in (*self.pulse_starts_s,self.pulse_width_s,self.relaxation_s,self.freeze_delay_s,self.stable_s):
            ticks(t,self.dt_ep_s)
        if any(b-a < self.pulse_width_s for a,b in zip(self.pulse_starts_s,self.pulse_starts_s[1:])):
            raise ValueError('pacing windows may not overlap')

    @property
    def pulse_ticks(self): return tuple(ticks(t,self.dt_ep_s) for t in self.pulse_starts_s)

    def stimulus(self,left_tick,mask,like):
        # Source midpoint sampling: each 2ms pulse covers exactly 40 EP steps.
        width = ticks(self.pulse_width_s,self.dt_ep_s)
        on = any(t <= left_tick+.5 < t+width for t in self.pulse_ticks)
        return mask.to(dtype=like.dtype)*self.amplitude_ua_per_cm2 if on else torch.zeros_like(like)

    def limit(self,left,proposed,cycle):
        boundaries = [*self.pulse_ticks,*(t+ticks(self.pulse_width_s,self.dt_ep_s) for t in self.pulse_ticks)]
        if cycle.closure_tick >= 0:
            boundaries += [cycle.closure_tick+ticks(self.relaxation_s,self.dt_ep_s),
                           cycle.closure_tick+ticks(self.freeze_delay_s,self.dt_ep_s)]
        future = [t-left for t in boundaries if t > left]
        return min([proposed,*future])


@dataclass(frozen=True)
class CycleState:
    mode: str = 'zero'
    closure_tick: int = -1
    freeze_tick: int = -1
    stable_ticks: int = 0
    anchor: torch.Tensor | None = None
    resting_land: torch.Tensor | None = None

    def factor(self,at_tick,protocol):
        if self.mode != 'relaxation': return 0.
        return max(0.,min(1.,1-(at_tick-self.closure_tick)/ticks(protocol.relaxation_s,protocol.dt_ep_s)))

    def ep_live(self,left,protocol):
        return left >= protocol.pulse_ticks[0] and self.freeze_tick < 0


def recovery_ready(old,new,rest_voltage,rest_calcium_mM,dt_s,protocol):
    # Reduction on actual fields, rather than a single representative cell.
    dt_ms = dt_s*1000
    vm_error = (new.voltage_v-rest_voltage).abs().max()*1000
    vm_rate = (new.voltage_v-old.voltage_v).abs().max()*1000/dt_ms
    ca = new.gpb[:,37]; previous = old.gpb[:,37]
    ca_error = ((ca-rest_calcium_mM).abs()/rest_calcium_mM).max()
    ca_rate = ((ca-previous).abs()/(rest_calcium_mM*dt_ms)).max()
    metrics = torch.stack((vm_error,vm_rate,ca_error,ca_rate))
    values = metrics.tolist()
    ready = (all(isfinite(v) for v in values) and
             values[0] <= protocol.vm_tolerance_mv and
             values[1] <= protocol.vm_tolerance_mv/(protocol.stable_s*1000) and
             values[2] <= protocol.calcium_rest_relative_tolerance and
             values[3] <= protocol.calcium_drift_relative_tolerance/(protocol.stable_s*1000))
    return ready,dict(zip(('vm_error_mv','vm_rate_mv_per_ms','calcium_rest_relative_error',
                          'calcium_rate_relative_per_ms'),values))
