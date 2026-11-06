"""Unmodified pressure-driven solid-LV RCR/valve equations (mmHg, mL, s).

Reference: test_ele_soild_heart-test/windkessel.h, commit 5bdde73c.
This is a reduced circulation, NOT a fluid-port boundary condition. All trials
are immutable. Geometry participates in the coupled volume residual, never in
the valve flow law. Raortic is a legacy, unused source parameter.
"""
from dataclasses import dataclass
from math import isfinite
from .aortic_valve import AorticValveParameters
from .windkessel import WindkesselParameters


@dataclass(frozen=True)
class SourceWKTrial:
    pc_mmhg: float
    arterial_mmhg: float
    flow_ml_s: float
    dq_dp_ml_s_mmhg: float
    valve_drop_mmhg: float
    pressure_residual_mmhg: float


def pressure_driven_trial(old_pc, cavity_pressure, dt, parameters=None, valve=None):
    p = parameters or WindkesselParameters()
    valve = valve or AorticValveParameters()
    if not all(isfinite(v) for v in (old_pc,cavity_pressure,dt)) or dt <= 0:
        raise ValueError('finite pressures and positive dt required')
    denom = 1+dt/(p.rp_mmhg_s_per_ml*p.compliance_ml_per_mmhg)
    pc0 = (old_pc+dt*p.venous_mmhg/(p.rp_mmhg_s_per_ml*p.compliance_ml_per_mmhg))/denom
    gain = dt/(p.compliance_ml_per_mmhg*denom)
    impedance = gain+p.rc_mmhg_s_per_ml
    drive = cavity_pressure-pc0
    if drive <= 0:
        return SourceWKTrial(pc0,pc0,0.,0.,drive,0.)
    lo,hi = 0.,drive
    h_lo = drive
    h_hi = -impedance*valve.flow_and_derivative(hi)[0]
    # Source safeguarded secant/bisection. Test pressure residual, not only width.
    for _ in range(100):
        dp = hi-h_hi*(hi-lo)/(h_hi-h_lo)
        if not lo+.1*(hi-lo) < dp < hi-.1*(hi-lo): dp = .5*(lo+hi)
        q,dq = valve.flow_and_derivative(dp)
        h = drive-dp-impedance*q
        if abs(h) <= 1e-10: break
        if h > 0: lo,h_lo = dp,h
        else: hi,h_hi = dp,h
    else:
        raise RuntimeError('source pressure-driven valve root failed')
    pc = pc0+gain*q
    return SourceWKTrial(pc,pc+p.rc_mmhg_s_per_ml*q,q,dq/(1+impedance*dq),dp,h)
