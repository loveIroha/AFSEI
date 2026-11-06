"""Source solid-LV pressure-driven valve, coupled to an actual fluid response.

Units: mmHg, mL, s. The valve and Rc are distinct series elements. No Raortic
legacy parameter is added. A closed valve requires a closed fluid boundary.
"""
from dataclasses import dataclass
from math import exp, isfinite
from .windkessel import flow_driven_trial


@dataclass(frozen=True)
class AorticValveParameters:
    rmin: float = .001
    rmax: float = 1e4
    epsilon_mmhg: float = .1
    smooth: bool = True

    def __post_init__(self):
        if not all(isfinite(v) and v>0 for v in (self.rmin,self.rmax,self.epsilon_mmhg)) or self.rmax<self.rmin:
            raise ValueError('positive ordered valve resistances and smoothing required')

    def flow_and_derivative(self, dp):
        if not isfinite(dp): raise ValueError('finite valve pressure difference required')
        if dp<=0: return 0.,0.
        z=exp(-2*dp/self.epsilon_mmhg) if self.smooth else 0.
        resistance=self.rmin+(self.rmax-self.rmin)*z/(1+z)
        dr=-2*(self.rmax-self.rmin)*z/(self.epsilon_mmhg*(1+z)**2)
        return dp/resistance,1/resistance-dp*dr/resistance**2


class ValveClosureRequired(RuntimeError):
    """Reject trial; repeat the same interval with a physically closed port."""


def flow_driven_valved_trial(old_pc, flow, dt, rcr, valve):
    """Pressure required by the actual forward fluid flow (not a new flow)."""
    if not isfinite(flow) or flow<0:raise ValueError('nonnegative actual flow required')
    wk=flow_driven_trial(old_pc,flow,dt,rcr)
    lo=0.;hi=flow*valve.rmax
    for _ in range(100):
        dp=(lo+hi)/2
        if valve.flow_and_derivative(dp)[0]<flow:lo=dp
        else:hi=dp
        if hi-lo<=2e-14*max(1.,abs(dp)):break
    dp=(lo+hi)/2
    return wk,float(wk.outlet_mmhg)+dp


def solve_fluid_valve(q_free, conductance, old_pc, dt, rcr, valve):
    """Solve Q=q_free+G*p_up AND Q=Phi(p_up-pAO) AND BE RCR.

    G is the negative fluid flux response to one mmHg on the physical port.
    Scalar root solves do not rerun the 3D fluid solver. This is a boundary
    Schur solve, not prescribing a second, independent 0D flow.
    """
    if not all(isfinite(x) for x in (q_free,conductance,old_pc,dt)) or conductance>=0:
        raise ValueError('finite data and negative fluid pressure conductance required')
    closed=flow_driven_trial(old_pc,0.,dt,rcr)
    a=float(closed.outlet_mmhg);z=closed.impedance_mmhg_s_per_ml
    drive=q_free+conductance*a
    if drive<=0: raise ValveClosureRequired('no forward valve flow; recompute with closed port')
    lo=0.;hi=drive/(-conductance)
    # H(dp)=drive+G*dp-(1-G*Z)*Phi(dp), strictly decreasing.
    for _ in range(90):
        dp=(lo+hi)/2;q,dq=valve.flow_and_derivative(dp)
        h=drive+conductance*dp-(1-conductance*z)*q
        if h>0:lo=dp
        else:hi=dp
        if hi-lo<=2e-14*max(1.,abs(dp)):break
    dp=(lo+hi)/2;q,dq=valve.flow_and_derivative(dp)
    wk=flow_driven_trial(old_pc,q,dt,rcr)
    pressure=float(wk.outlet_mmhg)+dp
    residual=q_free+conductance*pressure-q
    if abs(residual)>1e-9*max(1.,abs(q),abs(q_free)):
        raise RuntimeError('fluid/valve scalar solve failed')
    return dict(flow_ml_s=q,upstream_mmhg=pressure,valve_drop_mmhg=dp,
                pc_mmhg=float(wk.pc_mmhg),arterial_mmhg=float(wk.outlet_mmhg),
                tangent_impedance=z+1/dq,fluid_residual_ml_s=residual,
                valve_dissipation_mmhg_ml_s=dp*q)
