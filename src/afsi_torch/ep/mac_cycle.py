"""Four accepted-state phases on the original implicit MAC FSI framework.

The signed filling port is not an anatomical MV diode. The smooth prescribed
pressure can act at the inlet (legacy) or directly on the endocardium (IBFE
surface external load); the latter has a separate fixed reservoir pressure.
Closure captures applied spatial Ta before history publication.
"""
from dataclasses import dataclass, replace, asdict
from math import isfinite
import torch
from .mac_balanced_dynamic import BalancedAFSEIStepper, BalancedState
from .mac_rcr_coupled import outlet_rows
from .driver import AFSEIStepper
from .protocol import ticks
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..moving_cavity.bridge import publish_outlet_closure
from .gao_filling import GaoFillingParameters, filling_fraction

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
    def __init__(self,*args,cycles=2,period_s=.8,fill_target_mmhg=8.,
                 filling_pressure_location='inlet',filling_reservoir_mmhg=0.,
                 filling_cap_reference=False,filling_model="source",gao_filling=None,filling_hold_s=0.,
                 variable_pacing=False,**kwargs):
        super().__init__(*args,**kwargs)
        if type(cycles) is not int or cycles<1 or not isfinite(fill_target_mmhg):
            raise ValueError('invalid cardiac run settings')
        self.cycles=cycles; self.period_ticks=ticks(period_s,self.protocol.dt_ep_s)
        if self.period_ticks<=ticks(self.protocol.pulse_width_s,self.protocol.dt_ep_s):
            raise ValueError('invalid cycle period')
        expected=tuple(self.protocol.pulse_ticks[0]+i*self.period_ticks for i in range(cycles))
        self.variable_pacing=variable_pacing
        if type(variable_pacing) is not bool:raise ValueError('variable_pacing must be boolean')
        schedule=self.protocol.pulse_ticks
        if ((not variable_pacing and schedule!=expected) or len(schedule)!=cycles
                or variable_pacing and any(b-a<=ticks(self.protocol.pulse_width_s,self.protocol.dt_ep_s)
                                          for a,b in zip(schedule,schedule[1:]))):
            raise ValueError('pacing schedule must match requested cycles')
        self.end_tick=schedule[-1]+self.period_ticks
        self.fill_target_mmhg=fill_target_mmhg
        if filling_pressure_location not in ('inlet','endocardium') or not isfinite(filling_reservoir_mmhg):
            raise ValueError('invalid filling pressure location/reservoir')
        self.filling_pressure_location=filling_pressure_location
        self.filling_reservoir_mmhg=filling_reservoir_mmhg
        if type(filling_cap_reference) is not bool or (filling_cap_reference
                and filling_pressure_location!='endocardium'):
            raise ValueError('cap reference requires endocardial filling pressure')
        self.filling_cap_reference=filling_cap_reference
        if filling_model not in ('source','gao2017'):
            raise ValueError('unknown filling model')
        self.filling_model=filling_model
        self.gao_filling=(GaoFillingParameters(**gao_filling) if isinstance(gao_filling,dict)
                          else GaoFillingParameters() if gao_filling is None else gao_filling)
        if not isinstance(self.gao_filling,GaoFillingParameters):
            raise ValueError('Gao filling parameters required')
        if filling_model=='gao2017':
            if filling_cap_reference or filling_pressure_location!='inlet':
                raise ValueError('Gao uses independent inlet and wall loads, without cap reference')
            self.gao_filling.pressures(0.,fill_target_mmhg)
        if (isinstance(filling_hold_s,bool) or not isinstance(filling_hold_s,(int,float))
                or not isfinite(filling_hold_s) or filling_hold_s<0):
            raise ValueError('finite nonnegative filling hold required')
        self.filling_hold_ticks=ticks(filling_hold_s,self.protocol.dt_ep_s)
        if self.filling_hold_ticks>=self.period_ticks or (self.filling_hold_ticks and filling_model!='gao2017'):
            raise ValueError('filling hold requires Gao loading and must be shorter than the period')
        self.inlet_one_way=filling_model=='gao2017' and self.gao_filling.inlet_one_way

    def checkpoint_metadata(self,state):
        names=('phase','beat','initial_volume_ml','cumulative_in_ml','cumulative_out_ml',
               'edv_ml','esv_ml','fill_start_tick','fill_end_tick','fill_start_pressure_mmhg')
        result=dict(producer=PRODUCER,full_phase_controller=True,
            cardiac={n:getattr(state,n) for n in names},
            cardiac_execution=dict(cycles=self.cycles,period_s=self.period_ticks*self.protocol.dt_ep_s,
                                   fill_target_mmhg=self.fill_target_mmhg,
                                   filling_pressure_location=self.filling_pressure_location,
                                   filling_reservoir_mmhg=self.filling_reservoir_mmhg,
                                   filling_cap_reference=self.filling_cap_reference,
                                   filling_model=self.filling_model,
                                   gao_filling=asdict(self.gao_filling),
                                   filling_hold_s=self.filling_hold_ticks*self.protocol.dt_ep_s))
        if self.variable_pacing:result['cardiac_execution']['variable_pacing']=True
        return result

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
    def initialize_preload(self,path,*,windkessel=None,**kwargs):
        state=super().initialize_preload(path,mode='rcr',**kwargs)
        if windkessel is not None: state=replace(state,windkessel=windkessel)
        if self.filling_model=='source' and abs(state.windkessel.venous_mmhg-self.fill_target_mmhg)>1e-12:
            raise ValueError('source filling target must equal venous reservoir pressure')
        return CardiacState(**state.__dict__,initial_volume_ml=state.target_volume_ml,
                            edv_ml=state.target_volume_ml)

    def attempt(self,old,step_ticks,**kwargs):
        control=self._boundary_context
        if control is not None and control.mode=='filling':
            if self.filling_model=='gao2017':
                inlet,wall=self.gao_filling.pressures(
                    filling_fraction(control,old.tick+step_ticks),self.fill_target_mmhg)
                self._boundary_context=replace(control,wall_pressure_mmhg=wall,
                    cap_reference_mmhg=0.,filling_pressure_mmhg=inlet)
            else:
                prescribed=filling_pressure(control,old.tick+step_ticks,self.fill_target_mmhg)
                wall=self.filling_pressure_location=='endocardium'
                self._boundary_context=replace(control,
                    wall_pressure_mmhg=prescribed if wall else 0.,
                    cap_reference_mmhg=prescribed if wall and self.filling_cap_reference else 0.,
                    filling_pressure_mmhg=self.filling_reservoir_mmhg if wall else prescribed)
            kwargs['filling']=True
        elif control is not None:
            # An interval ending at pacing retains its accepted filling force;
            # the following interval uses the existing closed/RCR boundary.
            self._boundary_context=replace(control,wall_pressure_mmhg=0.,cap_reference_mmhg=0.)
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
                    # Finish the original pressure waveform before a constant-pressure
                    # tail, without stretching it to the delayed pacing boundary.
                    end-=self.filling_hold_ticks
                    if end<=state.tick:
                        raise ValueError('no pressure-waveform interval remains before the filling hold')
                    state=replace(state,phase='FILLING',mode='filling',fill_start_tick=state.tick,
                        fill_end_tick=end,fill_start_pressure_mmhg=pin,filling_pressure_mmhg=pin)
                    events.append('GAO_FILLING_ENTRY' if self.filling_model=='gao2017' else 'SMOOTH_FILLING_ENTRY')
        if old.phase=='FILLING':
            state=replace(state,target_volume_ml=volume,
                          filling_pressure_mmhg=(r['supply_pressure_mmhg'] if self.filling_model=='gao2017'
                              else r['boundary_pressure_mmhg']))
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
            filling_pressure_location=self.filling_pressure_location,
            filling_model=self.filling_model,
            inlet_one_way=self.inlet_one_way,
            prescribed_inlet_pressure_mmhg=(r.get('supply_pressure_mmhg') if old.phase=='FILLING' else None),
            applied_wall_pressure_mmhg=state.wall_pressure_mmhg,
            applied_cap_reference_mmhg=state.cap_reference_mmhg,
            fill_pressure_mmhg=((self.gao_filling.pressures(filling_fraction(state,state.tick),self.fill_target_mmhg)[0]
                if self.filling_model=='gao2017' else filling_pressure(state,state.tick,self.fill_target_mmhg))
                if state.phase=='FILLING' else None))
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
