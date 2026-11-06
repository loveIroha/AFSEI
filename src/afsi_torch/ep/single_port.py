"""One physical MAC box-face port; no immersed cap, tract or transfer correction.

This is a boundary component, not a complete LV driver. The caller must first
provide a mesh whose IB supports are valid at this physical boundary, then
decide phase/one-way events and commit Pc only after accepting the FSI step.
Other box faces retain the original exterior reference-pressure conditions.
"""
from math import isfinite
import torch
from ..mac.sealed_flow import SealedBEFlow
from ..mac.cut_plane import CutPlane, CutPlaneFlow
from .windkessel import flow_driven_trial


class PhysicalPortDomain:
    def __init__(self, grid, aperture, *, axis=2, side=-1, opened=True,
                 pressure_max_iterations=600):
        CutPlane(axis,side,aperture,'basal').validate(grid)
        self.planes=()
        self.config={'pressure_max_iterations':pressure_max_iterations}
        self.masks=list(grid.zeros(device=aperture.device,dtype=torch.float64))
        for m in self.masks:m.fill_(1.)
        # This is a physical box face, never an internal pressure sheet.
        self.masks[axis].select(axis,side).copy_(aperture if opened else torch.zeros_like(aperture))
        self.masks=tuple(self.masks)

    def mask(self,velocity):
        return tuple(m*u for m,u in zip(self.masks,velocity))


class SinglePortBoundary:
    """Cache only open/closed boundary operators for one fixed physical port.

    advance/response are compatible with the unchanged implicit BEIBStepper.
    No branch adds forces to the myocardium or modifies its IB transfer pair.
    """
    def __init__(self, base, aperture, *, axis=2, side=-1, fused=False, graphs=False, valve=None):
        self.plane=CutPlane(axis,side,aperture,'basal')
        self.flows={}
        for opened in (False,True):
            domain=PhysicalPortDomain(base.grid,aperture,axis=axis,side=side,opened=opened)
            masked=SealedBEFlow(base,domain,fused=fused,graphs=graphs)
            if valve is not None and opened:
                from ..mac.valved_cut_plane import ValvedCutPlaneFlow
                self.flows[opened]=ValvedCutPlaneFlow(masked,(self.plane,),valve=valve)
            else:
                self.flows[opened]=CutPlaneFlow(masked,(self.plane,) if opened else ())
        self.mode=None
        self.last_circuit=None
        self._accepted_step=-1
        self.pending=None

    def __getattr__(self,name):
        if name=='active':raise AttributeError(name)
        return getattr(self.active,name)

    def configure(self, mode, *, old_pc_mmhg, parameters, inlet_mmhg=0.):
        if mode not in ('closed','filling','ejection'):
            raise ValueError('single port mode must be closed, filling or ejection')
        if not isfinite(inlet_mmhg):raise ValueError('finite inlet pressure required')
        # Validate old Pc and dt even in the closed/filling phases.
        flow=self.flows[mode!='closed']
        flow_driven_trial(old_pc_mmhg,0.,flow.dt,parameters)
        self.active=flow
        self.mode=mode
        self.old_pc=old_pc_mmhg
        self.parameters=parameters
        self.last_circuit=None
        self.pending=None
        if mode=='ejection':flow.rcr(old_pc_mmhg,parameters,outlet=0)
        else:flow.prescribe([inlet_mmhg] if mode=='filling' else [])

    def advance(self,departure,density,pressure_initial=None):
        if self.mode is None:raise RuntimeError('configure the physical port before solving')
        self.pending=None
        self.last_circuit=None
        u,p,info=self.active.advance(departure,density,pressure_initial)
        q=(self.active.flux(u)[0] if self.mode!='closed' else p.new_zeros(()))
        self.last_circuit=(self.active.last_circuit if self.mode=='ejection' else
            flow_driven_trial(self.old_pc,0.,self.dt,self.parameters))
        self.pending={'mode':self.mode,'outward_flux_ml_s':q,
            'rcr_inflow_ml_s':q if self.mode=='ejection' else q.new_zeros(()),
            'phase_direction_valid':bool(q>=0) if self.mode=='ejection' else
                                   bool(q<=0) if self.mode=='filling' else True,
            'trial_pc_mmhg':self.last_circuit.pc_mmhg,
            'caller_must_check_direction_before_acceptance':True}
        info['single_port']=self.pending
        return u,p,info

    def commit(self, *, accepted_step):
        if self.pending is None or self.last_circuit is None:
            raise RuntimeError('no solved port trial to commit')
        if type(accepted_step) is not int or accepted_step<=self._accepted_step:
            raise RuntimeError('accepted step must increase')
        if not self.pending['phase_direction_valid']:
            raise RuntimeError('reverse-flow trial must be recomputed closed')
        self._accepted_step=accepted_step
        value=self.last_circuit.pc_mmhg
        self.pending=None
        return value

    def response(self,density):
        if self.mode is None:raise RuntimeError('configure the physical port before solving')
        return self.active.response(density)
