"""Four accepted-state phases on the original implicit MAC FSI framework.

The source smooth filling reservoir is signed, not an anatomical MV diode.
Only its cap pressure is prescribed; myocardial force and fluid pressure are
solved. Closure captures the applied spatial Ta before history publication.
"""
from dataclasses import dataclass, replace
from math import isfinite
import torch
from .mac_balanced_dynamic import BalancedAFSEIStepper, BalancedState
from .mac_rcr_coupled import outlet_rows
from .driver import AFSEIStepper
from .protocol import ticks
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..moving_cavity.bridge import publish_outlet_closure

PRODUCER = 'afsei-mac-four-phase-cycle'
PHASES = ('ISOVOLUMIC_CONTRACTION','EJECTION','ISOVOLUMIC_RELAXATION','FILLING')


@dataclass(frozen=True)
class CardiacState(BalancedState):
    phase: str = 'ISOVOLUMIC_CONTRACTION'
    beat: int = 1
    initial_volume_ml: float = 0.
    cumulative_in_ml: float = 0.
    cumulative_out_ml: float = 0.
    edv_ml: float = 0.
    esv_ml: float = 0.
    fill_start_tick: int = -1
    fill_end_tick: int = -1
    fill_start_pressure_mmhg: float = 0.


def filling_pressure(state,at_tick,target=8.):
    if state.fill_start_tick<0 or state.fill_end_tick<=state.fill_start_tick:
        raise ValueError('missing accepted filling anchor')
    x = max(0.,min(1.,(at_tick-state.fill_start_tick)/(state.fill_end_tick-state.fill_start_tick)))
    return state.fill_start_pressure_mmhg+(target-state.fill_start_pressure_mmhg)*x*x*(3-2*x)


class CardiacStepper(BalancedAFSEIStepper):
    def __init__(self,*args,cycles=2,period_s=.8,fill_target_mmhg=8.,**kwargs):
        super().__init__(*args,**kwargs)
        if type(cycles) is not int or cycles<1 or not isfinite(fill_target_mmhg):
            raise ValueError('invalid cardiac run settings')
        self.cycles=cycles; self.period_ticks=ticks(period_s,self.protocol.dt_ep_s)
        if self.period_ticks<=ticks(self.protocol.pulse_width_s,self.protocol.dt_ep_s):
            raise ValueError('invalid cycle period')
        expected=tuple(self.protocol.pulse_ticks[0]+i*self.period_ticks for i in range(cycles))
        if self.protocol.pulse_ticks!=expected:
            raise ValueError('pacing schedule must match requested cycles')
        self.end_tick=expected[0]+cycles*self.period_ticks
        self.fill_target_mmhg=fill_target_mmhg

    def checkpoint_metadata(self,state):
        names=('phase','beat','initial_volume_ml','cumulative_in_ml','cumulative_out_ml',
               'edv_ml','esv_ml','fill_start_tick','fill_end_tick','fill_start_pressure_mmhg')
        return dict(producer=PRODUCER,full_phase_controller=True,
            cardiac={n:getattr(state,n) for n in names},
            cardiac_execution=dict(cycles=self.cycles,period_s=self.period_ticks*self.protocol.dt_ep_s,
                                   fill_target_mmhg=self.fill_target_mmhg))

    def validate_control(self,state):
        super().validate_control(state)
        # Parent initialization builds a BalancedState before conversion.
        if not isinstance(state,CardiacState): return
        values=(state.initial_volume_ml,state.cumulative_in_ml,state.cumulative_out_ml,
                state.edv_ml,state.esv_ml,state.fill_start_pressure_mmhg)
        if (state.phase not in PHASES or type(state.beat) is not int or not 1<=state.beat<=self.cycles
                or not all(isfinite(v) for v in values) or state.initial_volume_ml<=0
                or state.phase=='FILLING' and state.mode!='filling'
                or state.phase=='ISOVOLUMIC_RELAXATION' and state.mode!='closed'
                or state.phase in ('ISOVOLUMIC_CONTRACTION','EJECTION') and state.mode!='rcr'
                or state.tick>self.end_tick):
            raise ValueError('invalid accepted cardiac state')
        if state.phase=='FILLING': filling_pressure(state,state.tick,self.fill_target_mmhg)

    @torch.no_grad()
    def initialize_preload(self,path,**kwargs):
        state=super().initialize_preload(path,mode='rcr',**kwargs)
        if abs(state.windkessel.venous_mmhg-self.fill_target_mmhg)>1e-12:
            raise ValueError('source filling target must equal venous reservoir pressure')
        return CardiacState(**state.__dict__,initial_volume_ml=state.target_volume_ml,
                            edv_ml=state.target_volume_ml)

    def attempt(self,old,step_ticks,**kwargs):
        control=self._boundary_context
        if control is not None and control.mode=='filling':
            self._boundary_context=replace(control,filling_pressure_mmhg=filling_pressure(
                control,old.tick+step_ticks,self.fill_target_mmhg))
            kwargs['filling']=True
        return AFSEIStepper.attempt(self,old,step_ticks,**kwargs)

    def inlet_pressure(self,state):
        """Area average of the accepted cap multiplier over the MV disc."""
        e=self._fsi_stepper(round(state.force_load.dt_s/self.protocol.dt_ep_s))
        mode=state.cap_mode or state.mode
        if mode=='filling': return state.cap_pressure[-1].item()/C
        # Evaluate labels on exactly the same quadrature as the closed rows.
        p=outlet_rows(e,state.cap_geometry,label=1).prepared
        from .mac_cap_coupled import closed_rows
        rows=outlet_rows(e,state.cap_geometry) if mode=='rcr' else closed_rows(e,state.cap_geometry)
        selected=p.cap.labels[rows.selected]==1
        area=p.area_vectors[rows.selected].norm(dim=-1)[selected]
        return (area*state.cap_pressure[rows.patches[selected]]).sum().item()/area.sum().item()/C

    @torch.no_grad()
    def publish_phase(self,old,state,info):
        r=info['balanced']; dt=info['accepted_dt_s']; events=[]
        cin=old.cumulative_in_ml+dt*r['inlet_flow_ml_s']
        cout=old.cumulative_out_ml+dt*r['measured_relative_q_ml_s']
        volume=r['volume_ml']
        if abs(volume-(old.initial_volume_ml+cin-cout))>1e-3:
            raise ValueError('cardiac cumulative blood-volume balance failed')
        state=replace(state,cumulative_in_ml=cin,cumulative_out_ml=cout)
        if old.mode=='rcr':
            if state.outlet_open:
                state=replace(state,phase='EJECTION',target_volume_ml=volume)
            elif old.phase=='EJECTION':
                if state.tick in self.protocol.pulse_ticks:
                    raise ValueError('outlet closure coincides with pacing; unresolved beat')
                ep=publish_outlet_closure(self,state.ep)
                state=replace(state,ep=ep,phase='ISOVOLUMIC_RELAXATION',mode='closed',
                              target_volume_ml=volume,esv_ml=volume)
                events.append('OUTLET_CLOSURE_TA_SNAPSHOT')
            elif abs(volume-old.target_volume_ml)>1e-3:
                raise ValueError('isovolumic contraction volume target failed')
        if state.phase=='ISOVOLUMIC_RELAXATION':
            off=state.cycle.closure_tick+ticks(self.protocol.relaxation_s,self.protocol.dt_ep_s)
            if state.tick>=off and state.cycle.mode=='zero':
                pin=self.inlet_pressure(state)
                if pin<=self.fill_target_mmhg:
                    end=self.protocol.pulse_ticks[0]+((state.tick-self.protocol.pulse_ticks[0])//self.period_ticks+1)*self.period_ticks
                    state=replace(state,phase='FILLING',mode='filling',fill_start_tick=state.tick,
                        fill_end_tick=end,fill_start_pressure_mmhg=pin,filling_pressure_mmhg=pin)
                    events.append('SMOOTH_FILLING_ENTRY')
        if old.phase=='FILLING':
            state=replace(state,target_volume_ml=volume,
                          filling_pressure_mmhg=r['boundary_pressure_mmhg'])
        if state.tick in self.protocol.pulse_ticks and state.tick!=self.protocol.pulse_ticks[0]:
            if old.phase!='FILLING':
                raise ValueError('next stimulus reached before completed unloading/filling')
            state=replace(state,phase='ISOVOLUMIC_CONTRACTION',mode='rcr',outlet_open=False,
                          beat=old.beat+1,target_volume_ml=volume,edv_ml=volume)
            events.append('PACING_END_DIASTOLIC_VOLUME')
        info['cardiac']=dict(phase=state.phase,solved_phase=old.phase,beat=state.beat,events=events,
            volume_ml=volume,edv_ml=state.edv_ml,esv_ml=state.esv_ml,
            cumulative_mass_error_ml=volume-state.initial_volume_ml-cin+cout,
            cumulative_in_ml=cin,cumulative_out_ml=cout,closure_tick=state.cycle.closure_tick,
            unload_factor=state.cycle.factor(state.tick,self.protocol),
            fill_pressure_mmhg=state.filling_pressure_mmhg if state.phase=='FILLING' else None)
        self.validate_control(state)
        return state,info

    @torch.no_grad()
    def step(self,old,*,target_tick=None,**kwargs):
        if not isinstance(old,CardiacState): raise ValueError('formal cardiac state required')
        target=min(target_tick or self.end_tick,self.end_tick)
        try:
            state,info=super().step(old,target_tick=target,**kwargs)
            return self.publish_phase(old,state,info)
        except Exception:
            self._reset_scratch(old.ep)
            raise
