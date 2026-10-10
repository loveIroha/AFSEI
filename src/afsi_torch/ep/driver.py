"""Unified EP/GPB/Land -> implicit BE fluid/solid -> accepted strain feedback.

This is partitioned electromechanical coupling, with two EP steps per default
mechanical step. It is not a monolithic three-field Newton solve. Physical
AV/MV ports, preload and circulation events are not manufactured here.
"""
from dataclasses import dataclass, replace
from contextlib import contextmanager
from time import perf_counter
import torch
from ..paper_lv import PaperLVSolid
from ..config import TimeConfig
from ..mac.paper_coupling import BEIBStepper, PaperState
from .solid import EPDrivenPaperSolid
from .buffered_load import BufferedLandLoad,ActiveLoadState
from .monodomain import Monodomain
from .tissue import GPBLand,TissueState,initial_state
from .mapping import source_p1_fields
from .protocol import CycleState,ticks,recovery_ready
from .land import distortion_step


@dataclass(frozen=True)
class StrainHistory:
    stretch: torch.Tensor
    zs: torch.Tensor
    zw: torch.Tensor


@dataclass(frozen=True)
class CoupledState:
    tick: int
    fsi: PaperState
    tissue: TissueState
    strain: StrainHistory
    cycle: CycleState
    force_load: ActiveLoadState
    ionic_current: torch.Tensor
    rejected_attempts: int = 0


class CoupledStepFailure(RuntimeError):
    def __init__(self,message,attempts):
        super().__init__(message)
        self.attempts = attempts


class _EndpointStepper(BEIBStepper):
    """Reuse the unchanged BE solve with a local clock, then publish EP ticks.

    EP loads are frozen tensors during the solve, and have no dependence on
    the local time supplied by BEProblem. The fluid RHS is autonomous. This
    adapter avoids altering upstream fixed-dt checkpoint/clock semantics.
    """
    @torch.no_grad()
    def physical_step(self,state,new_time,diagnostics):
        previous_x = state.previous_x
        if previous_x is not None and state.previous_dt is not None:
            previous_x = state.x-(state.x-previous_x)*(self.flow.dt/state.previous_dt)
        local = replace(state,step=0,time=0.,force_time=0.,pressure_time=0.,previous_x=previous_x)
        accepted,info = super().step(local,diagnostics=diagnostics)
        return replace(accepted,step=state.step+1,time=new_time,force_time=new_time,
                       pressure_time=new_time,previous_dt=self.flow.dt,previous_x=state.x.clone()),info


class AFSEIStepper:
    def __init__(self,mesh,config):
        if not mesh.X.is_cuda or mesh.X.dtype != torch.float64:
            raise ValueError('AFSEI GPB/Land time integration requires actual CUDA float64 tensors')
        self.config,self.mesh = config,mesh
        self.protocol = config.protocol
        passive = PaperLVSolid(mesh,config.fsi)
        self.buffer = BufferedLandLoad(passive.geometry,mesh.fiber)
        # Start with a zero load so every execution factory sees initialized buffers.
        n = len(mesh.X); zero = mesh.X.new_zeros(n)
        self.buffer.stage(ActiveLoadState(mesh.X,zero,zero,zero,zero,config.fsi.time.dt,False))
        self.model = EPDrivenPaperSolid(mesh,config.fsi,self.buffer,
            source_quadrature=config.mechanics_profile == 'source',
            prescribed_pressure_mmhg=config.prescribed_endocardial_pressure_mmhg)
        if config.direction_mapping == 'source-serial-cg1':
            fiber,sheet,self.stimulus_mask,self.mapping_owner = source_p1_fields(mesh)
            self.ep = Monodomain(mesh.X,mesh.cells,fiber,sheet,directions='nodal',parameters=config.ep)
        else:
            self.ep = Monodomain(mesh.X,mesh.cells,mesh.fiber,mesh.sheet,directions='cell',parameters=config.ep)
            self.stimulus_mask = torch.zeros(n,dtype=torch.bool,device=mesh.X.device)
            self.stimulus_mask[torch.unique(self.model.endo_faces)] = True
            self.mapping_owner = None
        self.ode = GPBLand(config.protocol.dt_ep_s,config.dt_ode_ms)
        resting = initial_state(mesh.X[:,0])
        self.rest_voltage = resting.voltage_v
        self.rest_calcium = resting.gpb[:,37].clone()
        self.maximum_ticks = ticks(config.fsi.time.dt,self.protocol.dt_ep_s)
        self._steppers = {}
        self._ep_steps = self._ode_substeps = 0
        # Runtime instrumentation is deliberately absent from physical checkpoints.
        self.profile = False
        self._last_attempt_timings = {}

    @contextmanager
    def _stage(self,name):
        if not self.profile:
            yield
            return
        torch.cuda.synchronize(self.mesh.X.device)
        started = perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize(self.mesh.X.device)
            self._last_attempt_timings[name] = self._last_attempt_timings.get(name,0.)+perf_counter()-started

    def _fsi_stepper(self,step_ticks):
        if step_ticks not in self._steppers:
            dt = step_ticks*self.protocol.dt_ep_s
            # Independent per-dt solvers retain correct Helmholtz diagonals,
            # captured graph coefficients and preconditioner dt after a retry.
            cfg = replace(self.config.fsi,time=TimeConfig(dt,dt))
            self._steppers[step_ticks] = _EndpointStepper(self.model,cfg,self.mesh.X.device)
        return self._steppers[step_ticks]

    @torch.no_grad()
    def initialize(self,x=None,tissue=None):
        x = self.mesh.X if x is None else x
        self.model.validate(x)
        tissue = initial_state(x[:,0]) if tissue is None else tissue
        F = self.model.element_gradient(x)
        cell_stretch = torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,self.mesh.fiber),dim=-1)
        stretch = self.ep.project_cell_field(cell_stretch)
        zero = torch.zeros_like(stretch)
        self.ode._validate(tissue,stretch,zero,zero,self.config.start_time_s)
        start = ticks(self.config.start_time_s,self.protocol.dt_ep_s)
        paced = start == self.protocol.pulse_ticks[0]
        cycle = CycleState(mode='land' if paced else 'zero',
                           resting_land=tissue.land.clone() if paced else None)
        load = ActiveLoadState(x.clone(),tissue.land[:,0].clone(),tissue.land[:,1].clone(),
                               zero.clone(),zero.clone(),self.config.fsi.time.dt,paced)
        self.buffer.stage(load)
        fsi = self._fsi_stepper(self.maximum_ticks).initialize(x)
        time = start*self.protocol.dt_ep_s
        fsi = replace(fsi,time=time,force_time=time,pressure_time=time)
        return CoupledState(start,fsi,tissue,StrainHistory(stretch,zero.clone(),zero.clone()),
                            cycle,load,zero.clone())

    def _check_clock(self,state):
        if (type(state.tick) is not int or state.tick < 0 or type(state.fsi.step) is not int
                or state.fsi.step < 0 or state.fsi.time != state.tick*self.protocol.dt_ep_s
                or state.fsi.force_time != state.fsi.time or state.fsi.pressure_time != state.fsi.time):
            raise ValueError('coupled state must share the exact integer EP clock')

    def _reset_scratch(self,state):
        self.buffer.stage(state.force_load)
        for driver in self._steppers.values():
            driver.transfer.reset_warm_start()

    @torch.no_grad()
    def attempt(self,old,step_ticks,*,closure=False,filling=False,diagnostics=True):
        """Trial one interval. Events require a future physical port solver.

        closure identifies closure at the accepted endpoint; filling asserts
        filling throughout this interval. A rejected trial publishes neither.
        """
        self._check_clock(old)
        if type(step_ticks) is not int or not 1 <= step_ticks <= self.maximum_ticks:
            raise ValueError('invalid mechanical EP tick count')
        if self.protocol.limit(old.tick,step_ticks,old.cycle) != step_ticks:
            raise ValueError('mechanical interval may not straddle a pacing/ramp event')
        if closure and old.cycle.mode != 'land':
            raise ValueError('outlet closure event requires an active accepted beat')
        end = old.tick+step_ticks
        dt = step_ticks*self.protocol.dt_ep_s
        tissue = old.tissue
        current = old.ionic_current
        live = old.cycle.ep_live(old.tick,self.protocol)
        ep_steps = 0; max_ratio = 0.; ep_iterations = 0
        self._last_attempt_timings = {}
        try:
            for left in range(old.tick,end):
                if not live: continue
                stim = self.protocol.stimulus(left,self.stimulus_mask,tissue.voltage_v)
                if self.profile:
                    with self._stage('gpb_land_seconds'):
                        ode,current = self.ode.advance_ode(tissue,old.strain.stretch,old.strain.zs,
                            old.strain.zw,left*self.protocol.dt_ep_s,advance_land=old.cycle.mode == 'land')
                    with self._stage('monodomain_seconds'):
                        voltage,ep_info = self.ep.step(tissue.voltage_v,current,stim,self.protocol.dt_ep_s)
                        tissue = TissueState(voltage,ode.gpb,ode.land)
                else:
                    tissue,current,ep_info = self.ode.step(self.ep,tissue,old.strain.stretch,old.strain.zs,
                        old.strain.zw,left*self.protocol.dt_ep_s,stim,advance_land=old.cycle.mode == 'land')
                ep_steps += 1
                ep_iterations += ep_info.iterations
                max_ratio = max(max_ratio,ep_info.residual_norm/ep_info.tolerance)
            load = ActiveLoadState(old.fsi.x,tissue.land[:,0],tissue.land[:,1],old.strain.zs,old.strain.zw,
                dt,old.cycle.mode == 'land',old.cycle.anchor,old.cycle.factor(end,self.protocol))
            with self._stage('load_staging_seconds'):
                self.buffer.stage(load)
            with self._stage('implicit_fsi_seconds'):
                driver = self._fsi_stepper(step_ticks)
                fsi,info = driver.physical_step(old.fsi,end*self.protocol.dt_ep_s,diagnostics)
            # Record closure Ta BEFORE publishing projected strain histories.
            anchor = self.buffer.closure_snapshot(fsi.x) if closure else old.cycle.anchor
            cycle = old.cycle
            refresh = old.tick < self.protocol.pulse_ticks[0] or live or end in self.protocol.pulse_ticks
            strain = old.strain
            if refresh:
                with self._stage('strain_feedback_seconds'):
                    F = self.model.element_gradient(fsi.x)
                    cell_stretch = torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,self.mesh.fiber),dim=-1)
                    stretch = self.ep.project_cell_field(cell_stretch)
                    if (stretch <= 0).any():
                        raise FloatingPointError('nonpositive projected Land stretch')
                    advance_z = old.tick < self.protocol.pulse_ticks[0] or old.cycle.mode == 'land'
                    delta = stretch-old.strain.stretch
                    zs = distortion_step(old.strain.zs,delta,dt,40.14) if advance_z else torch.zeros_like(stretch)
                    zw = distortion_step(old.strain.zw,delta,dt,405.86) if advance_z else torch.zeros_like(stretch)
                    strain = StrainHistory(stretch,zs,zw)
            if closure:
                if cycle.resting_land is None:
                    raise ValueError('missing first pre-stimulus Land snapshot')
                cycle = replace(cycle,mode='relaxation',closure_tick=end,anchor=anchor,
                                freeze_tick=-1,stable_ticks=0)
                tissue = TissueState(tissue.voltage_v,tissue.gpb,cycle.resting_land.clone())
                strain = StrainHistory(strain.stretch,torch.zeros_like(strain.zs),torch.zeros_like(strain.zw))
            recovery = {}
            earliest = cycle.closure_tick+ticks(self.protocol.freeze_delay_s,self.protocol.dt_ep_s)
            if (self.protocol.freeze_enabled and cycle.freeze_tick < 0 and cycle.closure_tick >= 0
                    and old.tick >= earliest and filling and not closure and live):
                ready,recovery = recovery_ready(old.tissue,tissue,self.rest_voltage,self.rest_calcium,dt,self.protocol)
                stable = cycle.stable_ticks+step_ticks if ready else 0
                frozen = end if stable >= ticks(self.protocol.stable_s,self.protocol.dt_ep_s) else -1
                cycle = replace(cycle,stable_ticks=stable,freeze_tick=frozen)
            elif cycle.freeze_tick < 0:
                cycle = replace(cycle,stable_ticks=0)
            if cycle.mode == 'relaxation' and cycle.factor(end,self.protocol) == 0:
                cycle = replace(cycle,mode='zero')
            # The interval ENDING at a pace used its previous mode/gate. Resume
            # only the next interval and retain voltage/GPB/Ca, as in main.cpp.
            if end in self.protocol.pulse_ticks:
                resting_land = tissue.land.clone() if cycle.resting_land is None else cycle.resting_land
                cycle = CycleState(mode='land',resting_land=resting_land)
            new = CoupledState(end,fsi,tissue,strain,cycle,load,current,old.rejected_attempts)
            info['ep'] = dict(steps=ep_steps,ode_substeps=ep_steps*self.ode.substeps,
                pcg_iterations=ep_iterations,max_residual_to_tolerance=max_ratio,
                land_advanced=live and old.cycle.mode == 'land',ep_live=live,recovery=recovery)
            info['accepted_dt_s'] = dt
            return new,info
        except Exception:
            self._reset_scratch(old)
            raise

    @torch.no_grad()
    def step(self,old,*,target_tick=None,diagnostics=True):
        proposed = self.maximum_ticks
        if target_tick is not None:
            if type(target_tick) is not int or target_tick <= old.tick:
                raise ValueError('target tick must follow the accepted state')
            proposed = min(proposed,target_tick-old.tick)
        proposed = self.protocol.limit(old.tick,proposed,old.cycle)
        attempts = []
        timings = {}
        def collect_timing():
            for key,value in self._last_attempt_timings.items():
                timings[key] = timings.get(key,0.)+value
        while True:
            try:
                new,info = self.attempt(old,proposed,diagnostics=diagnostics)
                collect_timing()
                info['rejected_attempts'] = attempts
                if self.profile: info['timings'] = timings
                return replace(new,rejected_attempts=old.rejected_attempts+len(attempts)),info
            except (RuntimeError,FloatingPointError,ValueError) as exc:
                collect_timing()
                attempts.append(dict(ticks=proposed,type=type(exc).__name__,message=str(exc)))
                if proposed == 1 or len(attempts) > self.config.max_retries:
                    raise CoupledStepFailure('minimum/allowed coupled step failed; accepted state restored',attempts) from exc
                proposed = max(1,proposed//2)
