"""Two physical MAC cuts with phase closure and optional source valve + RCR.

The caller supplies a verified fixed-tract domain. This component creates no
housing, immersed cap, volume multiplier, or modification to the FE/IB pair.
It does not establish the geometric connection of the tracts to an LV mesh.
Unlike Gao's anatomical MV, the inflow valve is ideal and closes in systole.
"""
from copy import copy
from math import isfinite
import torch
from ..mac.cut_plane import CutPlaneFlow
from ..mac.sealed_flow import SealedBEFlow
from .windkessel import flow_driven_trial


class DualPortBoundary:
    def __init__(self, base, domain, inlet, outlet, *, fused=False, graphs=False, valve=None):
        inlet.validate(base.grid); outlet.validate(base.grid)
        if inlet.axis==outlet.axis and inlet.side==outlet.side and (inlet.aperture & outlet.aperture).any():
            raise ValueError('inflow and outflow apertures must not overlap')
        for p in (inlet,outlet):
            if not torch.all(domain.masks[p.axis].select(p.axis,p.side)[p.aperture]==1):
                raise ValueError('verified domain must expose both physical apertures')
        self.ports=(inlet,outlet); self.flows={}; self.domain=domain
        for key,opened in [('closed',()),('filling',(inlet,)),('ejection',(outlet,))]:
            d=copy(domain); masks=[m.clone() for m in domain.masks]
            for p in (inlet,outlet):
                masks[p.axis].select(p.axis,p.side)[p.aperture]=0
            for p in opened:
                masks[p.axis].select(p.axis,p.side)[p.aperture]=1
            d.masks=tuple(masks)
            masked=SealedBEFlow(base,d,fused=fused,graphs=graphs)
            if key=='ejection' and valve is not None:
                from ..mac.valved_cut_plane import ValvedCutPlaneFlow
                self.flows[key]=ValvedCutPlaneFlow(masked,opened,valve=valve)
            else:
                self.flows[key]=CutPlaneFlow(masked,opened)
        self.phase=None; self.last_circuit=None; self.pending=None; self.last_committed_step=None

    def __getattr__(self,name):
        if name=='active': raise AttributeError(name)
        return getattr(self.active,name)

    def configure(self,phase,*,old_pc_mmhg,parameters,inlet_mmhg=0.,flow_tolerance_ml_s=1e-8):
        if phase not in ('filling','isovolumic_contraction','ejection','isovolumic_relaxation'):
            raise ValueError('invalid cardiac phase')
        if not isfinite(inlet_mmhg) or not isfinite(flow_tolerance_ml_s) or flow_tolerance_ml_s<0:
            raise ValueError('finite reference pressure and nonnegative flow tolerance required')
        key=phase if phase in ('filling','ejection') else 'closed'
        flow=self.flows[key]
        flow_driven_trial(old_pc_mmhg,0.,flow.dt,parameters)
        self.active=flow; self.phase=phase; self.old_pc=old_pc_mmhg; self.parameters=parameters
        self.flow_tolerance=flow_tolerance_ml_s; self.last_circuit=None; self.pending=None
        if key=='ejection': flow.rcr(old_pc_mmhg,parameters,outlet=0)
        else: flow.prescribe([inlet_mmhg] if key=='filling' else [])

    def advance(self,departure,density,pressure_initial=None):
        if self.phase is None: raise RuntimeError('configure phase before solving')
        self.pending=None; self.last_circuit=None
        u,p,info=self.active.advance(departure,density,pressure_initial)
        q=self.active.flux(u)
        zero=p.new_zeros(())
        qi=q[0] if self.phase=='filling' else zero
        qo=q[0] if self.phase=='ejection' else zero
        circuit=self.active.last_circuit if self.phase=='ejection' else flow_driven_trial(self.old_pc,0.,self.dt,self.parameters)
        direction=bool(qi<=self.flow_tolerance) and bool(qo>=-self.flow_tolerance)
        self.last_circuit=circuit
        self.pending=dict(phase=self.phase,inlet_outward_ml_s=qi,outlet_outward_ml_s=qo,
            phase_direction_valid=direction,trial_pc_mmhg=circuit.pc_mmhg,
            rcr_inflow_ml_s=qo,pressure_application='outflow physical cut face only')
        if self.phase=='ejection':
            self.pending['upstream_port_mmhg']=self.active.last_boundary['pressures_mmhg'][0]
            self.pending['arterial_pressure_mmhg']=circuit.outlet_mmhg
            self.pending['valve_drop_mmhg']=self.active.last_boundary.get('valve_drop_mmhg',zero)
        info['dual_port']=self.pending
        return u,p,info

    def response(self,density):
        if self.phase is None: raise RuntimeError('configure phase before solving')
        return self.active.response(density)

    def commit(self,*,accepted_step):
        """Call only after the full EP/FSI step is accepted, never in Newton."""
        if self.pending is None or not self.pending['phase_direction_valid']:
            raise RuntimeError('missing or reverse-flow trial: locate valve event and re-solve before commit')
        if type(accepted_step) is not int or accepted_step<0 or (self.last_committed_step is not None and accepted_step<=self.last_committed_step):
            raise ValueError('accepted step ledger must increase')
        pc=self.last_circuit.pc_mmhg
        self.last_committed_step=accepted_step; self.pending=None
        return pc
