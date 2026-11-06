"""Opt-in four-phase controller for physical fixed-tract MAC cut boundaries."""
from dataclasses import dataclass,replace,asdict
import json
from pathlib import Path
import numpy as np
import torch
from .mac_cycle import CardiacStepper,CardiacState,PHASES,filling_pressure
from .mac_balanced_dynamic import BalancedAFSEIStepper
from .driver import AFSEIStepper,StrainHistory
from .gao_filling import filling_fraction
from .gao_tracts import prepare_cut_flow,cut_rows
from .mac_preload import MACEndocardialPressure,mesh_binding
from .checkpoint import digest
from .protocol import ticks
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..moving_cavity.bridge import publish_outlet_closure
from .cycle_policy import FillingClock,PeriodicMonitor

PRODUCER='afsei-mac-physical-cut-four-phase-v1'


@dataclass(frozen=True)
class CutCardiacState(CardiacState):
    initial_tract_volume_ml: float=0.
    target_total_volume_ml: float=0.


class CutCardiacStepper(CardiacStepper):
    def __init__(self,*args,filling_clock=None,periodic_criteria=None,**kwargs):
        self.filling_clock=(FillingClock(**filling_clock) if isinstance(filling_clock,dict) else filling_clock)
        if self.filling_clock is not None:kwargs['variable_pacing']=True
        super().__init__(*args,**kwargs)
        if self.spatial_coupling!='legacy-balanced':
            raise ValueError('fixed-tract pressure work currently requires the balanced LV transfer')
        self.gao_cut_boundary=True
        self.cut_cycle_implemented=True
        self.cut_valve_active_set=True
        self.cut_pressure_compact_geometry=False
        self.cut_tract_geometry_gpu=False
        self.cut_pressure_tolerances=None
        self.cut_preload_metadata=None
        self.cut_geometry={}
        self.periodic_monitor=PeriodicMonitor(periodic_criteria) if periodic_criteria is not None else None
        if self.filling_clock is not None:
            if not isinstance(self.filling_clock,FillingClock) or self.filling_model!='gao2017':
                raise ValueError('physical filling clock requires Gao loading')
            for key in ('duration_s','inlet_ramp_s','wall_peak_s','hold_s'):
                ticks(getattr(self.filling_clock,key),self.protocol.dt_ep_s)

    def validate_control(self,state):
        BalancedAFSEIStepper.validate_control(self,state)
        if self.valve_resistance_mmhg_s_per_ml!=0:
            raise ValueError('physical-cut cycle requires zero additional valve resistance')
        if not isinstance(state,CutCardiacState):return
        if (state.phase not in PHASES or state.initial_volume_ml<=0
                or state.initial_tract_volume_ml<=0 or state.target_total_volume_ml<=0
                or not 1<=state.beat<=self.cycles or state.tick>self.end_tick
                or state.phase=='ISOVOLUMIC_CONTRACTION' and state.mode!='closed'
                or state.phase=='ISOVOLUMIC_RELAXATION' and state.mode!='closed'):
            raise ValueError('invalid physical-cut cardiac control')

    def checkpoint_metadata(self,state):
        meta=super().checkpoint_metadata(state)
        meta['producer']=PRODUCER
        meta['cardiac'].update(initial_tract_volume_ml=state.initial_tract_volume_ml,
                              target_total_volume_ml=state.target_total_volume_ml)
        if getattr(self,'filling_clock',None) is not None:
            meta['cardiac_execution']['filling_clock']=asdict(self.filling_clock)
        if getattr(self,'periodic_monitor',None) is not None:
            meta['cardiac_execution']['periodic_criteria']=asdict(self.periodic_monitor.criteria)
        return meta

    @torch.no_grad()
    def initialize_loading_history(self,path,*,arterial_pc_mmhg=85.,windkessel=None):
        with np.load(path,allow_pickle=False) as a:
            data={k:a[k] for k in a.files}
        meta=json.loads(str(data.pop('metadata')));expected=meta.pop('sha256')
        if (meta.get('producer')!='afsei-fixed-tract-loading-history' or meta.get('schema')!=1
                or meta.get('completed') is not True or digest(meta,data)!=expected
                or meta['mesh_binding_sha256']!=mesh_binding(self.mesh)
                or meta['steps']!=100 or meta['duration_s']!=self.config.start_time_s
                or meta.get('ep_ode_advanced') is not False):
            raise ValueError('invalid fixed-tract loading history')
        source_config=meta['config'];current_config=json.loads(json.dumps(asdict(self.config)))
        if self.variable_pacing:
            # No EP was advanced during loading. Future paced beats do not
            # affect this history; all other coefficients still match exactly.
            source_config=json.loads(json.dumps(source_config))
            source_config['protocol']['pulse_starts_s']=source_config['protocol']['pulse_starts_s'][:1]
            current_config['protocol']['pulse_starts_s']=current_config['protocol']['pulse_starts_s'][:1]
        if source_config!=current_config:
            raise ValueError('loading history config differs from fresh cardiac driver')
        schedule=meta.get('loading_schedule')
        if (meta.get('geometry',{}).get('neck_corner_trim_cells',0)
                !=self.cut_geometry.get('neck_corner_trim_cells',0)):
            raise ValueError('loading history uses a different fixed tract neck geometry')
        if schedule is not None:
            nominal=[r for r in meta['loading_records'] if r.get('physical_loading_endpoint')]
            if len(nominal)!=meta['steps']:raise ValueError('loading schedule is incomplete')
            for i,r in enumerate(nominal,1):
                t=meta['duration_s']*i/meta['steps']
                expected_pressure=meta['pressure_mmhg']*min(1.,t/schedule['ramp_s'])
                if abs(r['physical_loading_time_s']-t)>1e-10 or abs(r['pressure_mmhg']-expected_pressure)>1e-10:
                    raise ValueError('loading pressure/history clock differs from schedule')
        tensor=lambda k:torch.as_tensor(data[k],device=self.mesh.X.device)
        x=tensor('x');self.gao_tract_anchor=tensor('geometry_anchor')
        self.cut_preload_metadata=dict(sha256=expected,steps=meta['steps'],duration_s=meta['duration_s'],
            mesh_binding_sha256=meta['mesh_binding_sha256'],pressure_mmhg=meta['pressure_mmhg'])
        if schedule is not None:self.cut_preload_metadata['loading_schedule']=schedule
        ep=self.initialize(x=x)
        strain=StrainHistory(*(tensor(k) for k in ('stretch','zs','zw')))
        torch.testing.assert_close(strain.stretch,ep.strain.stretch,rtol=1e-10,atol=1e-10)
        self.ode._validate(ep.tissue,strain.stretch,strain.zs,strain.zw,ep.fsi.time)
        load=replace(ep.force_load,use_land=False,anchor=None,ramp_factor=0.,
                     zs_old=strain.zs.clone(),zw_old=strain.zw.clone())
        self.buffer.stage(load)
        ep=replace(ep,strain=strain,force_load=load)
        e=self._fsi_stepper(self.maximum_ticks)
        v=float(MACEndocardialPressure(self.model).volume(x))
        s=CutCardiacState(ep,v,x.new_zeros(1),x.clone(),mode='closed',
            arterial_pc_mmhg=arterial_pc_mmhg,initial_volume_ml=v,edv_ml=v)
        if windkessel is not None:s=replace(s,windkessel=windkessel)
        prepare_cut_flow(e,s.fsi,s)
        tract=e.flow.domain.storage.volume(x)
        rows=cut_rows(e,x);bp=torch.full_like(rows.area,meta['pressure_mmhg']*C);bp[-2:]=0.
        s=replace(s,cap_pressure=bp,initial_tract_volume_ml=tract,target_total_volume_ml=v+tract)
        s=replace(s,ep=replace(s.ep,fsi=replace(s.fsi,force=self.force_for_state(s))))
        stencil=e.transfer.prepare(x)
        density,_,_=e.transfer.relative_parts(s.fsi.force,bp,stencil,rows=rows)
        _,p,_=e.flow.advance(e.grid.zeros(device=x.device,dtype=x.dtype),density)
        s=replace(s,ep=replace(s.ep,fsi=replace(s.fsi,pressure=p)))
        self.validate_control(s)
        if self.periodic_monitor is not None:self.periodic_monitor.capture(s)
        return s

    def cut_pressure(self,state,label):
        e=self._fsi_stepper(round(state.force_load.dt_s/self.protocol.dt_ep_s))
        if not hasattr(e,'_gao_material_cap'):
            prepare_cut_flow(e,state.fsi,state)
        domain=e.flow.domain
        values=state.fsi.pressure.select(domain.axis,-1)
        return float(values[domain.apertures[label-1]].mean()/C)

    def inlet_pressure(self,state):return self.cut_pressure(state,1)

    def filling_loads(self,state,at_tick):
        if getattr(self,'filling_clock',None) is not None:
            return self.filling_clock.pressures((at_tick-state.fill_start_tick)*self.protocol.dt_ep_s,
                                               self.gao_filling,self.fill_target_mmhg)
        if self.filling_model=='gao2017':
            return self.gao_filling.pressures(filling_fraction(state,at_tick),self.fill_target_mmhg)
        return filling_pressure(state,at_tick,self.fill_target_mmhg),0.

    def reserve_filling_window(self,state):
        """Called only at accepted unloading completion, never in Newton trials."""
        clock=getattr(self,'filling_clock',None)
        if clock is None:
            end=self.protocol.pulse_ticks[0]+((state.tick-self.protocol.pulse_ticks[0])//self.period_ticks+1)*self.period_ticks
            return end-self.filling_hold_ticks,False
        end=state.tick+ticks(clock.duration_s,self.protocol.dt_ep_s)
        ready=end+ticks(clock.hold_s,self.protocol.dt_ep_s)
        exact=clock.pacing_mode=='after-filling'
        changed=self.delay_pacing(state.beat,ready,allow_advance=exact)
        if state.beat==self.cycles:self.end_tick=ready if exact else max(self.end_tick,ready)
        return end,changed

    def delay_pacing(self,index,ready,*,allow_advance=False):
        pulses=list(self.protocol.pulse_ticks)
        if index>=len(pulses) or pulses[index]==ready or not allow_advance and pulses[index]>ready:return False
        delta=ready-pulses[index];pulses[index:]=[p+delta for p in pulses[index:]]
        self.protocol=replace(self.protocol,pulse_starts_s=tuple(p*self.protocol.dt_ep_s for p in pulses))
        self.config=replace(self.config,protocol=self.protocol);self.end_tick+=delta
        return True

    def protect_unready_pacing(self,state):
        clock=getattr(self,'filling_clock',None)
        if clock is None or state.phase=='FILLING':return False
        proposed=state.tick+self.maximum_ticks
        ready=proposed+ticks(self.protocol.relaxation_s+clock.duration_s+clock.hold_s,self.protocol.dt_ep_s)
        pulses=self.protocol.pulse_ticks
        if state.beat<len(pulses) and pulses[state.beat]<=proposed:
            return self.delay_pacing(state.beat,ready)
        if state.beat==self.cycles and self.end_tick<=proposed:self.end_tick=ready
        return False

    def attempt(self,old,step_ticks,**kwargs):
        control=self._boundary_context
        if control is not None:
            fill=control.phase=='FILLING'
            pin,wall=self.filling_loads(control,old.tick+step_ticks) if fill else (control.filling_pressure_mmhg,0.)
            self._boundary_context=replace(control,filling_pressure_mmhg=pin,wall_pressure_mmhg=wall,
                                           cap_reference_mmhg=0.)
            kwargs['filling']=fill
        return AFSEIStepper.attempt(self,old,step_ticks,**kwargs)

    @torch.no_grad()
    def publish_cut_phase(self,old,state,info,*,valve_closed=False):
        r=info['balanced'];dt=info['accepted_dt_s'];events=[]
        upstream=None
        solved_mode=r.get('mode','closed' if valve_closed else old.mode)
        if solved_mode=='closed':
            upstream=self.cut_pressure(state,1 if old.phase=='FILLING' else 2)
            r.update(boundary_pressure_mmhg=upstream,prescribed_cut_pressure_mmhg=None,
                boundary_pressure_kind='closed-upstream-fluid-trace')
        else:
            r.update(prescribed_cut_pressure_mmhg=r.get('boundary_pressure_mmhg'),
                boundary_pressure_kind='prescribed-physical-cut')
        qout=r.get('physical_outlet_flow_ml_s',0.)
        qin=r['inlet_flow_ml_s']
        cin=old.cumulative_in_ml+dt*qin;cout=old.cumulative_out_ml+dt*qout
        v=r['volume_ml'];tract=r['tract_volume_ml'];total=v+tract
        mass=total-old.initial_volume_ml-old.initial_tract_volume_ml-cin+cout
        if abs(mass)>1e-3:raise ValueError('physical-cut cumulative blood-volume balance failed')
        state=replace(state,cumulative_in_ml=cin,cumulative_out_ml=cout)
        if old.phase in ('ISOVOLUMIC_CONTRACTION','ISOVOLUMIC_RELAXATION'):
            if abs(total-old.target_total_volume_ml)>1e-3:
                raise ValueError('closed physical-cut total blood volume drift')
        if valve_closed and old.phase=='EJECTION':
            ep=publish_outlet_closure(self,state.ep)
            state=replace(state,ep=ep,phase='ISOVOLUMIC_RELAXATION',mode='closed',outlet_open=False,
                target_volume_ml=v,target_total_volume_ml=total,esv_ml=v)
            events.append('OUTLET_CLOSURE_TA_SNAPSHOT')
        elif old.phase=='ISOVOLUMIC_CONTRACTION':
            if upstream>state.arterial_pc_mmhg+1e-5:
                state=replace(state,phase='EJECTION',mode='rcr',outlet_open=True)
                events.append('PHYSICAL_AV_OPENING')
        if state.phase=='ISOVOLUMIC_RELAXATION':
            off=state.cycle.closure_tick+ticks(self.protocol.relaxation_s,self.protocol.dt_ep_s)
            if state.tick>=off and state.cycle.mode=='zero':
                pin=self.inlet_pressure(state)
                wall_driven=(getattr(self,'filling_clock',None) is not None
                             and self.filling_clock.loading_mode=='source-wall')
                if pin<=self.fill_target_mmhg or wall_driven:
                    end,delayed=self.reserve_filling_window(state)
                    if end<=state.tick:raise ValueError('no filling waveform interval before pacing')
                    state=replace(state,phase='FILLING',mode='closed',fill_start_tick=state.tick,
                        fill_end_tick=end,fill_start_pressure_mmhg=pin,filling_pressure_mmhg=pin)
                    events.append('SOURCE_WALL_FILLING_ENTRY' if wall_driven else 'GAO_FILLING_ENTRY')
                    if delayed:events.append('PACING_RESCHEDULED_AFTER_FILLING' if getattr(self,'filling_clock',None)
                        and self.filling_clock.pacing_mode=='after-filling' else 'PACING_DELAYED_FOR_FULL_FILLING')
        if state.phase=='FILLING':
            pin,_=self.filling_loads(state,state.tick)
            if state.mode=='closed' and self.inlet_pressure(state)<pin-1e-5:
                state=replace(state,mode='filling')
                events.append('PHYSICAL_MV_OPENING')
            state=replace(state,target_volume_ml=v,target_total_volume_ml=total)
        if state.tick in self.protocol.pulse_ticks and state.tick!=self.protocol.pulse_ticks[0]:
            if old.phase!='FILLING':raise ValueError('pacing reached before filling/unloading')
            state=replace(state,phase='ISOVOLUMIC_CONTRACTION',mode='closed',outlet_open=False,
                beat=old.beat+1,target_volume_ml=v,target_total_volume_ml=total,edv_ml=v)
            events.append('PACING_END_DIASTOLIC_VOLUME')
        r['basal_measured_relative_q_ml_s']=r['measured_relative_q_ml_s']
        r['measured_relative_q_ml_s']=qout
        info['cardiac']=dict(phase=state.phase,solved_phase=old.phase,solved_mode=solved_mode,beat=state.beat,events=events,
            volume_ml=v,edv_ml=state.edv_ml,esv_ml=state.esv_ml,cumulative_mass_error_ml=mass,
            cumulative_in_ml=cin,cumulative_out_ml=cout,closure_tick=state.cycle.closure_tick,
            unload_factor=state.cycle.factor(state.tick,self.protocol),filling_model=self.filling_model,
            filling_pressure_location='physical-cut',inlet_one_way=self.inlet_one_way,
            applied_wall_pressure_mmhg=state.wall_pressure_mmhg,applied_cap_reference_mmhg=0.,
            total_blood_volume_ml=total,tract_volume_ml=tract,
            prescribed_inlet_pressure_mmhg=r.get('supply_pressure_mmhg'))
        self.validate_control(state)
        if ('PACING_END_DIASTOLIC_VOLUME' in events
                and getattr(self,'periodic_monitor',None) is not None):
            self.periodic_monitor.capture(state)
        if getattr(self,'periodic_monitor',None) is not None:
            info['cardiac']['periodicity']=self.periodic_monitor.summary()
        info['cardiac']['scheduled_pulse_starts_ms']=[p*1000 for p in self.protocol.pulse_starts_s]
        info['cardiac']['effective_filling_model']=(self.filling_clock.loading_mode
            if getattr(self,'filling_clock',None) is not None else self.filling_model)
        return state,info

    @torch.no_grad()
    def step(self,old,*,target_tick=None,**kwargs):
        target=min(target_tick or self.end_tick,self.end_tick)
        clock=(self.protocol,getattr(self,'config',None),self.end_tick)
        try:
            deferred=self.protect_unready_pacing(old)
            state,info=BalancedAFSEIStepper.step(self,old,target_tick=target,**kwargs)
            r=info['balanced']
            reverse=(old.mode=='rcr' and r.get('physical_outlet_flow_ml_s',0.) < -1e-5
                     or old.mode=='filling' and r['inlet_flow_ml_s'] < -1e-5)
            if reverse:
                trial_time=state.fsi.time
                self._reset_scratch(old.ep)
                state,info=BalancedAFSEIStepper.step(self,replace(old,mode='closed'),target_tick=state.tick,**kwargs)
                if state.fsi.time!=trial_time:raise ValueError('valve active-set attempts differ in time')
                info['cut_valve_trial_discarded']=True
            state,info=self.publish_cut_phase(old,state,info,valve_closed=bool(reverse and old.mode=='rcr'))
            if deferred:info['cardiac']['events'].append('PACING_DELAYED_UNTIL_RELAXATION')
            return state,info
        except Exception:
            self.protocol,self.config,self.end_tick=clock
            self._reset_scratch(old.ep)
            raise
