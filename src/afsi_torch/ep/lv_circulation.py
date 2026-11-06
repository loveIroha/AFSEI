"""Transactional source-LV phase logic with flow-driven arterial RCR.

This controller supplies no geometry, cap force or pressure correction. A caller
must solve its physical fluid boundary and accept its FSI state atomically with
commit(). Pressures: mmHg, volumes: mL, times: s; outward basal flow is positive.
"""
from dataclasses import dataclass, replace
from math import isfinite
from .windkessel import WindkesselParameters, flow_driven_trial
from .aortic_valve import AorticValveParameters


@dataclass(frozen=True)
class LVCirculationConfig:
    filling_ramp_s: float = .2
    filling_hold_s: float = .3
    filling_wall_mmhg: float = 8.
    inlet_reference_mmhg: float = 0.
    initial_pc_mmhg: float = 85.
    unload_s: float = .08
    volume_tolerance_ml: float = 1e-3
    flow_tolerance_ml_s: float = 1e-8

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isfinite(value): raise ValueError('finite '+name+' required')
        for name in ('filling_ramp_s','unload_s','volume_tolerance_ml','flow_tolerance_ml_s'):
            if getattr(self,name)<=0: raise ValueError('positive '+name+' required')
        if self.filling_hold_s<0 or self.filling_wall_mmhg<0:
            raise ValueError('nonnegative filling hold and wall load required')


@dataclass(frozen=True)
class LVState:
    time_s: float
    volume_ml: float
    pc_mmhg: float
    phase: str = 'filling'
    beat: int = 0
    filling_start_s: float = 0.
    unload_start_s: float | None = None
    iso_volume_ml: float | None = None
    cumulative_in_ml: float = 0.
    cumulative_out_ml: float = 0.


@dataclass(frozen=True)
class LVPlan:
    revision: int
    state: LVState
    dt_s: float
    boundary_mode: str
    wall_pressure_mmhg: float
    unload_factor: float
    stimulus: bool
    capture_closure_tension: bool
    events: tuple


class LVFlowDirectionError(RuntimeError):
    """Reject this trial and resolve the same interval with the port closed."""


class LVCirculation:
    def __init__(self, initial_volume_ml, *, config=None, windkessel=None, valve=None):
        self.config=config or LVCirculationConfig()
        self.windkessel=windkessel or WindkesselParameters()
        self.valve=valve or AorticValveParameters()
        if not isfinite(initial_volume_ml) or initial_volume_ml<=0:
            raise ValueError('positive initial LV volume required')
        self.state=LVState(0.,initial_volume_ml,self.config.initial_pc_mmhg)
        self.revision=0

    def plan(self, dt_s, *, lv_pressure_mmhg, close_outlet=False, close_inlet=False):
        if not isfinite(dt_s) or dt_s<=0 or not isfinite(lv_pressure_mmhg):
            raise ValueError('finite pressure and positive step required')
        c=self.config;s=self.state;events=[];stim=False;capture=False
        fill_duration=c.filling_ramp_s+c.filling_hold_s
        if s.phase=='filling' and s.time_s-s.filling_start_s>=fill_duration-1e-12:
            s=replace(s,phase='isovolumic_contraction',beat=s.beat+1,
                iso_volume_ml=s.volume_ml,unload_start_s=None,
                pc_mmhg=c.initial_pc_mmhg if s.beat==0 else s.pc_mmhg)
            stim=True;events.append('STIMULUS')
        # At a closed outlet use this interval's Q=0 arterial pressure, matching
        # the source BE trial. Never equate the LV pressure to arterial Pc.
        pc_closed=float(flow_driven_trial(s.pc_mmhg,0.,dt_s,self.windkessel).pc_mmhg)
        if s.phase=='isovolumic_contraction' and lv_pressure_mmhg>pc_closed:
            s=replace(s,phase='ejection');events.append('OUTFLOW_OPENING')
        if close_outlet:
            if s.phase!='ejection': raise ValueError('closure retry requires ejection')
            s=replace(s,phase='isovolumic_relaxation',unload_start_s=s.time_s,
                      iso_volume_ml=s.volume_ml)
            capture=True;events.append('OUTFLOW_CLOSURE')
        if (s.phase=='isovolumic_relaxation' and
            s.time_s-s.unload_start_s>=c.unload_s-1e-12 and
            lv_pressure_mmhg<=c.filling_wall_mmhg):
            s=replace(s,phase='filling',filling_start_s=s.time_s,iso_volume_ml=None)
            events.append('FILLING_START')
        mode='closed';wall=0.;factor=0.
        if s.phase=='filling':
            elapsed=s.time_s-s.filling_start_s
            endpoint=c.filling_ramp_s if elapsed<c.filling_ramp_s-1e-12 else fill_duration
            dt_s=min(dt_s,max(endpoint-elapsed,1e-15))
            wall=c.filling_wall_mmhg*min(1.,(elapsed+dt_s)/c.filling_ramp_s)
            mode='closed' if close_inlet else 'filling'
        elif close_inlet: raise ValueError('inlet retry requires filling')
        if s.phase=='ejection': mode='ejection'
        if s.phase=='isovolumic_relaxation':
            remaining=c.unload_s-(s.time_s-s.unload_start_s)
            if remaining>1e-12: dt_s=min(dt_s,remaining)
            factor=max(0.,1-(s.time_s+dt_s-s.unload_start_s)/c.unload_s)
        return LVPlan(self.revision,s,dt_s,mode,wall,factor,stim,capture,tuple(events))

    def commit(self, plan, *, outward_flow_ml_s, volume_ml, upstream_pressure_mmhg=None):
        """Only after FSI convergence; failures leave all accepted state intact."""
        if plan.revision!=self.revision: raise RuntimeError('stale circulation plan')
        q=float(outward_flow_ml_s);v=float(volume_ml);c=self.config;s=plan.state
        if not isfinite(q) or not isfinite(v) or v<=0: raise ValueError('finite flow and positive volume required')
        if ((plan.boundary_mode=='ejection' and q < -c.flow_tolerance_ml_s) or
            (plan.boundary_mode=='filling' and q > c.flow_tolerance_ml_s)):
            raise LVFlowDirectionError('reverse flow: reject and recompute closed; never clip Q')
        if plan.boundary_mode=='closed' and abs(q)>c.flow_tolerance_ml_s:
            raise RuntimeError('closed port has nonzero flow')
        mass_error=v-self.state.volume_ml+plan.dt_s*q
        if abs(mass_error)>c.volume_tolerance_ml:
            raise RuntimeError('LV volume/port flux mismatch')
        if s.iso_volume_ml is not None and plan.boundary_mode=='closed':
            if abs(v-s.iso_volume_ml)>c.volume_tolerance_ml:
                raise RuntimeError('isovolumic volume drift')
        arterial_q=q if plan.boundary_mode=='ejection' else 0.
        trial=flow_driven_trial(s.pc_mmhg,arterial_q,plan.dt_s,self.windkessel)
        valve_error=0.
        if plan.boundary_mode=='ejection':
            if upstream_pressure_mmhg is None or not isfinite(float(upstream_pressure_mmhg)):
                raise ValueError('actual upstream port pressure required to verify source valve')
            expected,_=self.valve.flow_and_derivative(float(upstream_pressure_mmhg)-float(trial.outlet_mmhg))
            valve_error=q-expected
            if abs(valve_error)>c.flow_tolerance_ml_s*max(1.,abs(q)):
                raise RuntimeError('actual fluid flow violates source valve relation')
        # Before first stimulation the arterial circuit is not initialized.
        pc=float(trial.pc_mmhg) if s.beat else c.initial_pc_mmhg
        self.state=replace(s,time_s=s.time_s+plan.dt_s,volume_ml=v,pc_mmhg=pc,
            cumulative_in_ml=s.cumulative_in_ml-(plan.dt_s*q if plan.boundary_mode=='filling' else 0.),
            cumulative_out_ml=s.cumulative_out_ml+(plan.dt_s*q if plan.boundary_mode=='ejection' else 0.))
        self.revision+=1
        return dict(mass_error_ml=mass_error,pc_mmhg=pc,
                    valve_flow_residual_ml_s=valve_error,
                    outlet_mmhg=float(trial.outlet_mmhg) if plan.boundary_mode=='ejection' else None,
                    events=plan.events)
