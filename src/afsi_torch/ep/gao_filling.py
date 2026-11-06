"""Gao et al. (2017), section 2.4: separate supply and endocardial loads.

The published 0.8 s filling interval is rescaled to the available AFSEI filling
interval. This is an LV-only adaptation, not a reproduction of anatomical MV
dynamics or the paper's complete cycle. Pressure units here are mmHg.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class GaoFillingParameters:
    inlet_start_mmhg: float = 0.
    wall_peak_mmhg: float = 12.
    wall_peak_fraction: float = .5
    inlet_one_way: bool = True
    # Fraction of the original filling window used to ramp the inlet only.
    inlet_ramp_fraction: float = 1.

    def __post_init__(self):
        for value in (self.inlet_start_mmhg, self.wall_peak_mmhg, self.wall_peak_fraction, self.inlet_ramp_fraction):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError('finite numeric Gao filling parameters required')
        if self.wall_peak_mmhg < 0 or not 0 < self.wall_peak_fraction < 1:
            raise ValueError('nonnegative wall peak and interior peak fraction required')
        if not 0 < self.inlet_ramp_fraction <= 1:
            raise ValueError('inlet ramp fraction must be in (0, 1]')
        if type(self.inlet_one_way) is not bool:
            raise ValueError('Gao inlet_one_way must be boolean')

    def pressures(self, fraction, inlet_end_mmhg):
        if not isfinite(fraction) or not isfinite(inlet_end_mmhg):
            raise ValueError('finite filling phase and inlet endpoint required')
        if inlet_end_mmhg < self.inlet_start_mmhg:
            raise ValueError('Gao inlet pressure must ramp upward')
        x = max(0., min(1., fraction))
        inlet = self.inlet_start_mmhg + (inlet_end_mmhg-self.inlet_start_mmhg)*min(1., x/self.inlet_ramp_fraction)
        triangle = x/self.wall_peak_fraction if x <= self.wall_peak_fraction else (1-x)/(1-self.wall_peak_fraction)
        return inlet, self.wall_peak_mmhg*triangle


def filling_fraction(state, at_tick):
    if state.fill_start_tick < 0 or state.fill_end_tick <= state.fill_start_tick:
        raise ValueError('missing accepted filling anchor')
    return max(0., min(1., (at_tick-state.fill_start_tick)/(state.fill_end_tick-state.fill_start_tick)))


def filling_port_law(outward_speed, port_mmhg, supply_mmhg, one_way):
    """Complementarity: outward Q <= 0, port-supply >= 0, product = 0.

    These are effective cap traction multipliers; they are not pointwise
    absolute fluid pressures on either side of the immersed cap.
    """
    gap = supply_mmhg-port_mmhg
    opened = not one_way or (outward_speed < gap).item()
    return (gap if opened else outward_speed), opened


def filling_port_accepted(q_out, projected_q_out, port_mmhg, supply_mmhg, opened, one_way):
    gap = port_mmhg-supply_mmhg
    return (abs(q_out-projected_q_out) <= 1e-5
            and (abs(gap) <= 1e-7 if opened else abs(q_out) <= 1e-5 and gap >= -1e-7)
            and (not one_way or q_out <= 1e-5))
