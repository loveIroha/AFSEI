"""Original LV-only geometry + source pressure-driven circulation + MAC/IB.

One scalar pLV joins the existing implicit displacement solve:
 V(X)-V_blood_old+dt*Q_source(pLV)=0, or prescribed filling pressure.
pLV is an endocardial load, not the mean MAC pressure. The natural open base
is NOT sealed: this reduced model cannot claim anatomical valve flux closure.
No connection walls, tract mesh, virtual cap reactions or fluid-pressure modes.
"""
from dataclasses import dataclass, asdict, replace
from math import isfinite
import torch
from ..config import TimeConfig
from ..nonlinear import normalized_linear_action
from .driver import AFSEIStepper, StrainHistory
from .tissue import TissueState
from .protocol import ticks
from .aortic_valve import AorticValveParameters
from .source_windkessel import pressure_driven_trial
from .mac_isovolumic import (IsovolumicState, IsovolumicProblem,
    IsovolumicEndpointStepper, IsovolumicAFSEIStepper, VOLUME_TOLERANCE_ML)
from .mac_preload import MACEndocardialPressure
from .mac_transport import closed_volume_gradient


@dataclass(frozen=True)
class SourceCycleParameters:
    fill_pressure_mmhg: float = 8.
    fill_rise_s: float = .2
    fill_hold_s: float = .3
    initial_pc_mmhg: float = 85.
    beats: int = 2
    flow_tolerance_ml_s: float = 1e-6
    valve: AorticValveParameters = AorticValveParameters()
    pacing_mode: str = 'after-filling'
    filling_shape: str = 'linear'
    stop_time_s: float | None = None

    def __post_init__(self):
        if (not all(isfinite(v) and v > 0 for v in (self.fill_pressure_mmhg,
                self.fill_rise_s,self.initial_pc_mmhg,self.flow_tolerance_ml_s))
                or not isfinite(self.fill_hold_s) or self.fill_hold_s<0
                or self.pacing_mode not in ('after-filling','fixed')
                or self.filling_shape not in ('linear','source-smoothstep')
                or (self.stop_time_s is not None and (not isfinite(self.stop_time_s) or self.stop_time_s<=0))
                or type(self.beats) is not int or self.beats < 1):
            raise ValueError('invalid source circulation/phase settings')


@dataclass(frozen=True)
class SourceLVState(IsovolumicState):
    phase: str = 'PRELOAD'
    filling_start_tick: int = 0
    beat: int = 0
    cumulative_out_ml: float = 0.
    cumulative_in_ml: float = 0.
    initial_volume_ml: float = 0.
    outlet_flow_ml_s: float = 0.
    arterial_pressure_mmhg: float = 85.
    filling_end_tick: int = -1
    filling_anchor_mmhg: float = 0.


def filling_pressure(control, end_tick, protocol, parameters):
    elapsed = (end_tick-control.filling_start_tick)*protocol.dt_ep_s
    duration=parameters.fill_rise_s
    anchor=0.
    if parameters.pacing_mode=='fixed':
        finish=protocol.pulse_ticks[0] if control.phase=='PRELOAD' else control.filling_end_tick
        if finish<=control.filling_start_tick:raise ValueError('missing fixed filling endpoint')
        # Initial loading has a separate rise and hold even with fixed pacing.
        # Later filling still uses its actual phase start and next stimulus.
        if control.phase!='PRELOAD':
            duration=(finish-control.filling_start_tick)*protocol.dt_ep_s
        anchor=control.filling_anchor_mmhg
    fraction=min(1.,max(0.,elapsed/duration))
    if parameters.filling_shape=='source-smoothstep' and control.phase!='PRELOAD':
        fraction=fraction*fraction*(3-2*fraction)
    return anchor+(parameters.fill_pressure_mmhg-anchor)*fraction


def next_nominal_pace(start,protocol):
    future=[v for v in protocol.pulse_ticks if v>start]
    if future:return min(future)
    pulse=protocol.pulse_ticks
    if len(pulse)<2:raise ValueError('two pacing times required to extrapolate filling clock')
    period=pulse[-1]-pulse[-2]
    return pulse[-1]+((start-pulse[-1])//period+1)*period


class SourceCycleProblem(IsovolumicProblem):
    def __init__(self,*args):
        super().__init__(*args)
        self.filling = self.control.phase in ('PRELOAD','FILLING')
        self.extra_unknowns = 0 if self.filling else 1
        self._wk_key = self._wk = None

    def prescribed_pressure(self):
        owner = self.driver.owner
        return filling_pressure(self.control,self.control.tick+self.driver.step_ticks,
                                owner.protocol,owner.circulation)

    def unpack(self,z):
        if not self.filling: return super().unpack(z)
        y = z.reshape_as(self.state.x)
        return y,self.state.x+y,z.new_tensor(self.prescribed_pressure())

    def trial(self,p):
        key = float(p)
        if key != self._wk_key:
            self._wk = pressure_driven_trial(self.control.arterial_pc_mmhg,key,
                self.time,self.control.windkessel,self.driver.owner.circulation.valve)
            self._wk_key = key
        return self._wk

    def constraint(self,volume,p):
        if self.filling:
            # Prescribed pressure is eliminated exactly, not solved to a
            # displacement-scaled Newton tolerance.
            return volume.new_empty(0)
        return (volume-self.control.target_volume_ml+self.time*self.trial(p).flow_ml_s)/self.volume_scale

    def constraint_derivatives(self,x,p):
        if self.filling: return torch.zeros_like(x).flatten(),x.new_ones(())
        c = closed_volume_gradient(x,self.load.faces,self.load.rim).flatten()/self.volume_scale
        diagonal = x.new_tensor(self.time*self.trial(p).dq_dp_ml_s_mmhg*self.pressure_scale/self.volume_scale)
        return c,diagonal

    def linearization(self,z):
        if not self.filling: return super().linearization(z)
        d = self.driver
        _,x,p = self.unpack(z)
        if d.assembler is None: d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(x,self.time)
        self.tangent_assemblies += 1
        offset = p-d.owner.config.prescribed_endocardial_pressure_mmhg
        def action(v):
            self.jacobian_actions += 1
            dy = v.reshape_as(x)
            df = torch.sparse.mm(tangent,v.reshape(-1,1)).reshape_as(x)
            df = df+self.load.action(x,offset,dy)
            return v+self.response(df).flatten()
        return normalized_linear_action(action)

    def preconditioner(self,z):
        if self.filling: return lambda v:v
        return super().preconditioner(z)


class SourceCycleEndpoint(IsovolumicEndpointStepper):
    def make_problem(self,state,stencil,departure,control):
        return SourceCycleProblem(self,state,stencil,departure,control)

    def volume_error(self,problem,volume,control):
        p = problem.last_data[1]
        if problem.filling:
            error = float(p)-problem.prescribed_pressure()
            if abs(error) > 1e-7: raise ValueError('prescribed filling pressure not converged')
            return 0.
        return float(problem.constraint(volume,p))*problem.volume_scale


class SourceLVStepper(IsovolumicAFSEIStepper):
    def __init__(self,mesh,config):
        if config.mechanics_profile != 'source' or config.prescribed_endocardial_pressure_mmhg != 0:
            raise ValueError('source mechanics and zero duplicate baseline pressure required')
        super().__init__(mesh,config)
        self.circulation = SourceCycleParameters()

    def execution_settings(self):
        return dict(**super().execution_settings(),circulation=asdict(self.circulation))

    def set_execution(self,*,circulation=None,**kwargs):
        if kwargs.get('closed_cap',False): raise ValueError('source LV has no cap reactions')
        super().set_execution(**kwargs)
        if circulation is not None:
            data = dict(circulation)
            data['valve'] = AorticValveParameters(**data.get('valve',{}))
            self.circulation = SourceCycleParameters(**data)
        for t in (self.circulation.fill_rise_s,self.circulation.fill_hold_s):
            ticks(t,self.protocol.dt_ep_s)
        if self.protocol.relaxation_s != .08:
            raise ValueError('source LV requires 80 ms accepted-closure unloading')
        if self.circulation.pacing_mode=='fixed':
            if len(self.protocol.pulse_ticks)!=self.circulation.beats:
                raise ValueError('fixed pacing must specify every requested beat')
            if self.circulation.stop_time_s is None or ticks(self.circulation.stop_time_s,self.protocol.dt_ep_s)<=self.protocol.pulse_ticks[-1]:
                raise ValueError('fixed pacing requires stop time after last stimulus')

    def _fsi_stepper(self,step_ticks):
        if step_ticks not in self._steppers:
            dt = step_ticks*self.protocol.dt_ep_s
            self._steppers[step_ticks] = SourceCycleEndpoint(self,replace(self.config.fsi,time=TimeConfig(dt,dt)))
        return self._steppers[step_ticks]

    @torch.no_grad()
    def initialize(self,x=None,tissue=None):
        if self.config.start_time_s != 0: raise ValueError('fresh source LV starts at zero; use checkpoint to resume')
        first = self.circulation.fill_rise_s+self.circulation.fill_hold_s
        if self.circulation.pacing_mode=='fixed':
            if self.protocol.pulse_ticks[0]!=ticks(first,self.protocol.dt_ep_s):
                raise ValueError('first pacing time must end preload')
        elif self.protocol.pulse_starts_s != (first,):
            raise ValueError('only first stimulus is preset; later stimuli follow accepted filling')
        ep = AFSEIStepper.initialize(self,x,tissue)
        volume = MACEndocardialPressure(self.model).volume(ep.fsi.x).item()
        return SourceLVState(ep,volume,0.,self.circulation.initial_pc_mmhg,
            initial_volume_ml=volume,arterial_pressure_mmhg=self.circulation.initial_pc_mmhg)

    def validate_control(self,state):
        if not isinstance(state,SourceLVState): raise ValueError('source LV circulation checkpoint required')
        super().validate_control(state)
        if (state.phase not in ('PRELOAD','IVC','EJECTION','IVR','FILLING')
                or type(state.beat) is not int or not 0 <= state.beat <= self.circulation.beats
                or not 0 <= state.filling_start_tick <= state.tick
                or state.initial_volume_ml <= 0 or state.cap_pressure is not None
                or not all(isfinite(v) for v in (state.cumulative_in_ml,state.cumulative_out_ml,
                    state.initial_volume_ml,state.outlet_flow_ml_s,state.arterial_pressure_mmhg))):
            raise ValueError('invalid source LV phase/history')
        ledger = state.initial_volume_ml+state.cumulative_in_ml-state.cumulative_out_ml
        if abs(ledger-state.target_volume_ml) > 1e-8:
            raise ValueError('source circulation volume ledger is not conservative')

    def completed(self,state):
        if self.circulation.stop_time_s is not None:
            return state.beat==self.circulation.beats and state.tick>=ticks(self.circulation.stop_time_s,self.protocol.dt_ep_s)
        return (state.beat == self.circulation.beats and state.phase == 'FILLING'
            and state.tick-state.filling_start_tick >= ticks(
                self.circulation.fill_rise_s+self.circulation.fill_hold_s,self.protocol.dt_ep_s))

    def attempt(self,old,step_ticks,**kwargs):
        return AFSEIStepper.attempt(self,old,step_ticks,
            filling=self._iso_context.phase == 'FILLING',**kwargs)

    @torch.no_grad()
    def step(self,old,**kwargs):
        self.validate_control(old)
        if self.completed(old): raise ValueError('requested beats and final filling already complete')
        self._iso_context = old
        filling = old.phase in ('PRELOAD','FILLING')
        # Land/EP already split at pulse and unload events. Also split filling corners.
        events = []
        if filling:
            if self.circulation.pacing_mode=='fixed':
                if old.phase=='PRELOAD':
                    events=[old.filling_start_tick+ticks(self.circulation.fill_rise_s,self.protocol.dt_ep_s),
                            self.protocol.pulse_ticks[0]]
                else:
                    events=[old.filling_end_tick]
            else:
                events = [old.filling_start_tick+ticks(t,self.protocol.dt_ep_s) for t in
                    (self.circulation.fill_rise_s,self.circulation.fill_rise_s+self.circulation.fill_hold_s)]
        if self.circulation.stop_time_s is not None:
            events.append(ticks(self.circulation.stop_time_s,self.protocol.dt_ep_s))
        future = [t for t in events if t > old.tick]
        if future:
            kwargs['target_tick'] = min([*future,kwargs.get('target_tick',min(future))])
        try:
            ep,info = AFSEIStepper.step(self,old.ep,**kwargs)
            if ep.tick in self.protocol.pulse_ticks and not filling:
                raise ValueError('fixed stimulus overlaps unfinished preceding contraction/relaxation')
            dt = info['accepted_dt_s']; report = info.pop('isovolumic')
            p,volume = report['cavity_pressure_mmhg'],report['volume_ml']
            wk = pressure_driven_trial(old.arterial_pc_mmhg,
                -1e9 if filling else p,dt,old.windkessel,self.circulation.valve)
            q = wk.flow_ml_s
            # Source TrialResult::valve_open is strict Q>0 AND pLV>pAO.
            # flow_tol checks balance, never turns small forward flow into closure.
            valve_open = not filling and q > 0. and p > wk.arterial_mmhg
            target = volume if filling else old.target_volume_ml-dt*q
            qin = (target-old.target_volume_ml)/dt if filling else 0.
            phase,beat,start = old.phase,old.beat,old.filling_start_tick
            closure = old.phase == 'EJECTION' and not valve_open
            if closure:
                if ep.cycle.resting_land is None: raise ValueError('missing accepted Land rest state')
                anchor = self.buffer.closure_snapshot(ep.fsi.x)
                cycle = replace(ep.cycle,mode='relaxation',closure_tick=ep.tick,anchor=anchor,
                                freeze_tick=-1,stable_ticks=0)
                ep = replace(ep,cycle=cycle,tissue=TissueState(ep.tissue.voltage_v,ep.tissue.gpb,
                    cycle.resting_land.clone()),strain=StrainHistory(ep.strain.stretch,
                    torch.zeros_like(ep.strain.zs),torch.zeros_like(ep.strain.zw)))
                phase = 'IVR'
            elif valve_open:
                phase = 'EJECTION'
            if filling and ep.tick in self.protocol.pulse_ticks:
                phase,beat = 'IVC',old.beat+1
            if (old.phase == 'IVR' and ep.cycle.mode == 'zero'
                    and not valve_open and p <= self.circulation.fill_pressure_mmhg):
                phase,start = 'FILLING',ep.tick
            pc = self.circulation.initial_pc_mmhg if old.phase == 'PRELOAD' else wk.pc_mmhg
            wk_balance = (old.windkessel.compliance_ml_per_mmhg*(wk.pc_mmhg-old.arterial_pc_mmhg)/dt
                -q+(wk.pc_mmhg-old.windkessel.venous_mmhg)/old.windkessel.rp_mmhg_s_per_ml)
            if abs(wk_balance) > self.circulation.flow_tolerance_ml_s:
                raise ValueError('source Windkessel balance exceeds flow tolerance')
            result = replace(old,ep=ep,target_volume_ml=target,cavity_pressure_mmhg=p,
                arterial_pc_mmhg=pc,arterial_pressure_mmhg=pc if filling else wk.arterial_mmhg,
                accepted_closed_steps=old.accepted_closed_steps+int(not filling and not valve_open),
                phase=phase,beat=beat,filling_start_tick=start,outlet_flow_ml_s=q,
                cumulative_in_ml=old.cumulative_in_ml+dt*qin,
                cumulative_out_ml=old.cumulative_out_ml+dt*q)
            if phase=='FILLING' and old.phase!='FILLING' and self.circulation.pacing_mode=='fixed':
                result=replace(result,filling_end_tick=next_nominal_pace(start,self.protocol),filling_anchor_mmhg=p)
            self.validate_control(result)
            # Commit event-driven stimulation only after geometry/ledger validation.
            if (self.circulation.pacing_mode=='after-filling' and phase == 'FILLING'
                    and old.phase != 'FILLING' and beat < self.circulation.beats):
                next_tick = start+ticks(self.circulation.fill_rise_s+self.circulation.fill_hold_s,
                                       self.protocol.dt_ep_s)
                self.protocol = replace(self.protocol,pulse_starts_s=(*self.protocol.pulse_starts_s,
                                                                     next_tick*self.protocol.dt_ep_s))
                self.config = replace(self.config,protocol=self.protocol)
            info['circulation'] = dict(phase=phase,previous_phase=old.phase,beat=beat,
                volume_ml=volume,blood_volume_ml=target,volume_balance_error_ml=volume-target,
                endocardial_load_mmhg=p,arterial_pressure_mmhg=result.arterial_pressure_mmhg,
                pc_mmhg=pc,q_out_0d_ml_s=q,q_in_volume_ml_s=qin,
                cumulative_in_ml=result.cumulative_in_ml,cumulative_out_ml=result.cumulative_out_ml,
                valve_closed_event=closure,closure_tick=ep.cycle.closure_tick,
                valve_open=valve_open,rcr_balance_ml_s=wk_balance if old.phase != 'PRELOAD' else None,
                filling_start_tick=start,pulse_starts_s=self.protocol.pulse_starts_s,
                filling_end_tick=result.filling_end_tick,filling_anchor_mmhg=result.filling_anchor_mmhg,
                physical_fluid_ports_connected=False,pressure_is_fluid_cavity_mean=False)
            return result,info
        except Exception:
            self._reset_scratch(old.ep)
            raise
        finally:
            self._iso_context = None
