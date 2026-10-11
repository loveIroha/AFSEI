"""Global closed-volume validation inside the original implicit MAC response.

This adds one pressure unknown, not an outer loop of complete FSI solves.
The source follower load acts only on endocardium; the moving basal fan is
included in the volume differential. Global volume does not seal that fan.
"""
from dataclasses import dataclass, replace, asdict
from math import isfinite
import torch
from ..config import TimeConfig
from ..nonlinear import newton, normalized_linear_action, NonlinearFailure
from ..mac.paper_coupling import BEIBStepper, PaperState
from ..mac.grid import divergence
from ..mac.open_boundary import face_weights
from .driver import AFSEIStepper, CoupledState
from .mac_preload import MACEndocardialPressure
from .windkessel import WindkesselParameters, flow_driven_trial
from .mac_transport import closed_volume_gradient

VOLUME_TOLERANCE_ML = 1e-3


@dataclass(frozen=True)
class IsovolumicState:
    ep: CoupledState
    target_volume_ml: float
    cavity_pressure_mmhg: float
    arterial_pc_mmhg: float = 85.
    windkessel: WindkesselParameters = WindkesselParameters()
    accepted_closed_steps: int = 0
    cap_pressure: torch.Tensor | None = None
    cap_geometry: torch.Tensor | None = None

    def __getattr__(self, name):
        return getattr(self.ep, name)


class IsovolumicProblem:
    extra_unknowns = 1
    def __init__(self, driver, state, stencil, departure, control):
        self.driver, self.state, self.stencil, self.departure = driver, state, stencil, departure
        self.control = control
        self.load = MACEndocardialPressure(driver.model)
        self.time = driver.flow.dt
        self.pressure_initial = state.pressure
        self.evaluations = self.jacobian_actions = self.tangent_assemblies = 0
        self.preconditioner_applications = 0
        self.last_z = self.last_residual = self.last_data = None
        self.cache_hits = 0
        self._last_state_version = None
        self.volume_scale = torch.linalg.vector_norm(closed_volume_gradient(state.x,self.load.faces,self.load.rim)).item()
        scale = driver.pressure_scale_cache if driver.owner.response_optimizations else None
        if scale is None:
            mobility = self.response(self.load.force(state.x, 1.))
            size = mobility.norm().item()
        else:
            size = 1./scale
        if not isfinite(size) or size <= 0 or not isfinite(self.volume_scale) or self.volume_scale <= 0:
            raise ValueError('degenerate isovolumic pressure mobility or volume gradient')
        self.pressure_scale = 1./size
        if driver.owner.response_optimizations:
            # Units/conditioning only. Every Newton pressure column is still
            # evaluated at the current geometry with the current IB stencil.
            driver.pressure_scale_cache = self.pressure_scale
            driver.owner._pressure_scales[driver.step_ticks] = self.pressure_scale

    def response(self, force):
        d = self.driver
        d.transfer.reset_warm_start()
        density, _ = d.transfer.spread(force, self.stencil)
        velocity = d.flow.response(density)
        nodal, _ = d.transfer.interpolate(velocity, self.stencil)
        return -d.flow.dt*nodal

    def unpack(self, z):
        y = z[:-1].reshape_as(self.state.x)
        p = self.control.cavity_pressure_mmhg+self.pressure_scale*z[-1]
        return y, self.state.x+y, p

    def validate(self, z):
        _, x, p = self.unpack(z)
        if not torch.isfinite(p): raise FloatingPointError('nonfinite cavity pressure')
        self.driver.solid.validate(x)

    def residual(self, z):
        state_version = None if torch.is_inference(self.state.x) else self.state.x._version
        if (self.driver.owner.response_optimizations and self.last_z is not None
                and state_version is not None and state_version == self._last_state_version
                and torch.equal(z,self.last_z)):
            self.cache_hits += 1
            return self.last_residual.clone()
        self.validate(z)
        y, x, p = self.unpack(z)
        d = self.driver
        baseline = d.owner.config.prescribed_endocardial_pressure_mmhg
        force = d.solid.force(x, self.time)+self.load.force(x, p-baseline)
        density, spread = d.transfer.spread(force, self.stencil)
        velocity, pressure, flow = d.flow.advance(self.departure, density, self.pressure_initial)
        nodal, interpolation = d.transfer.interpolate(velocity, self.stencil)
        volume = self.load.volume(x)
        residual = torch.cat(((y-d.flow.dt*nodal).flatten(),
                              ((volume-self.control.target_volume_ml)/self.volume_scale).reshape(1)))
        if not torch.isfinite(residual).all(): raise FloatingPointError('nonfinite isovolumic residual')
        self.pressure_initial = pressure
        self.last_z, self.last_residual = z.clone(), residual.clone()
        self._last_state_version = state_version
        self.last_data = x, p, volume, force, velocity, pressure, flow, nodal, density, spread, interpolation
        self.evaluations += 1
        return residual

    def accepted(self, z):
        if self.last_z is None or not torch.equal(z, self.last_z): self.residual(z)
        return self.last_residual

    def linearization(self, z):
        d = self.driver
        _, x, p = self.unpack(z)
        if d.assembler is None: d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(x, self.time)
        self.tangent_assemblies += 1
        c = closed_volume_gradient(x,self.load.faces,self.load.rim).flatten()/self.volume_scale
        b = self.response(self.load.force(x, 1.)).flatten()*self.pressure_scale
        self.c, self.b = c, b
        offset = p-d.owner.config.prescribed_endocardial_pressure_mmhg
        def action(v):
            self.jacobian_actions += 1
            dy = v[:-1].reshape_as(x)
            df = torch.sparse.mm(tangent, dy.reshape(-1, 1)).reshape_as(x)
            df = df+self.load.action(x, offset, dy)
            mechanical = dy.flatten()+self.response(df).flatten()+b*v[-1]
            return torch.cat((mechanical, (c@dy.flatten()).reshape(1)))
        return normalized_linear_action(action)

    def preconditioner(self, z):
        # Signed approximate Schur complement, with A approximately identity.
        # This changes only convergence; acceptance uses the full true residual.
        c, b = self.c, self.b
        denominator = c@b
        if not torch.isfinite(denominator) or denominator.abs() <= 1e-14:
            raise ValueError('singular isovolumic Schur complement')
        def apply(v):
            self.preconditioner_applications += 1
            q = (c@v[:-1]-v[-1])/denominator
            return torch.cat((v[:-1]-b*q, q.reshape(1)))
        return apply


class IsovolumicEndpointStepper(BEIBStepper):
    def __init__(self, owner, config):
        super().__init__(owner.model, config, owner.mesh.X.device)
        self.owner = owner
        self.step_ticks = round(config.time.dt/owner.protocol.dt_ep_s)
        self.pressure_scale_cache = owner._pressure_scales.get(self.step_ticks)
        if owner.pressure_backend == 'spectral':
            from ..mac.spectral_dirichlet import SpectralDirichletPressure
            self.flow.pressure_solver = SpectralDirichletPressure(self.grid,
                device=owner.mesh.X.device,options=config.pressure_solver)

    def make_problem(self, state, stencil, departure, control):
        if self.owner.closed_cap:
            from .mac_cap_coupled import CapIsovolumicProblem
            return CapIsovolumicProblem(self, state, stencil, departure, control)
        return IsovolumicProblem(self, state, stencil, departure, control)

    def volume_error(self, problem, volume, control):
        return (volume-control.target_volume_ml).item()

    @torch.no_grad()
    def physical_step(self, state, new_time, diagnostics):
        control = self.owner._iso_context
        if control is None: raise ValueError('missing immutable isovolumic trial control')
        self.solid.validate(state.x)
        stencil = self.transfer.prepare(state.x)
        departure = self.flow.advect(state.velocity)
        before = (self.flow.calls, self.flow.pressure_cycles, self.flow.helmholtz_sweeps,
                  self.mass_calls, self.mass_iterations)
        direct_before = getattr(self.flow.pressure_solver,'direct_solves',0)
        problem = self.make_problem(state, stencil, departure, control)
        initial = torch.zeros_like(state.x)
        if state.previous_x is not None:
            initial = (state.x-state.previous_x)*(self.flow.dt/(state.previous_dt or self.flow.dt))
        z = torch.cat((initial.flatten(), initial.new_zeros(problem.extra_unknowns)))
        r0 = problem.residual(z)
        settings = self.config.nonlinear
        # Retain the original mechanical tolerance, independent of constraint scaling.
        tolerance = max(settings.atol, settings.rtol*r0[:state.x.numel()].norm().item())
        options = replace(settings, rtol=0., atol=tolerance)
        result = newton(problem.residual, z, validate=problem.validate,
            linearization_factory=problem.linearization, preconditioner_factory=problem.preconditioner,
            options=options)
        residual = problem.accepted(result.x)
        x, p, volume, force, velocity, pressure, flow, nodal, density, spread, interpolation = problem.last_data
        error = self.volume_error(problem, volume, control)
        if residual.norm().item() > tolerance or abs(error) > VOLUME_TOLERANCE_ML:
            raise NonlinearFailure('isovolumic true residual/volume exceeds target', result)
        self.check_support(x)
        courant = self.flow.dt*sum(u.abs().max()/h for u,h in zip(velocity,self.grid.spacing)).item()
        if not isfinite(courant) or courant > self.config.max_courant:
            raise ValueError('isovolumic characteristic CFL exceeds configured limit')
        accepted = PaperState(state.step+1, new_time, x, velocity, pressure, force, new_time,
            pressure_time=new_time, previous_x=state.x.clone(), previous_dt=self.flow.dt)
        linear = [h['linear'] for h in result.history if 'linear' in h]
        info = dict(nonlinear=dict(iterations=result.iterations,residual_norm=residual.norm().item(),
            tolerance=tolerance,history=result.history,solver='augmented-newton',
            residual_evaluations=problem.evaluations,jacobian_actions=problem.jacobian_actions,
            residual_cache_hits=problem.cache_hits,pressure_backend=self.flow.pressure_solver.backend,
            pressure_direct_solves=getattr(self.flow.pressure_solver,'direct_solves',0)-direct_before,
            tangent_assemblies=problem.tangent_assemblies,preconditioner_applications=problem.preconditioner_applications,
            fluid_solves=self.flow.calls-before[0],pressure_cycles=self.flow.pressure_cycles-before[1],
            helmholtz_sweeps=self.flow.helmholtz_sweeps-before[2],mass_solves=self.mass_calls-before[3],
            mass_iterations=self.mass_iterations-before[4],gmres_iterations=sum(h['iterations'] for h in linear)),
            flow=flow,force_mass=asdict(spread),velocity_mass=asdict(interpolation),courant=courant,
            isovolumic=dict(target_volume_ml=control.target_volume_ml,volume_ml=volume.item(),
                volume_error_ml=error,cavity_pressure_mmhg=p.item(),volume_tolerance_ml=VOLUME_TOLERANCE_ML,
                pressure_scale_mmhg_per_cm=problem.pressure_scale,validation_only=True))
        if diagnostics:
            fp = sum((u*f*face_weights(self.grid,c,x)).sum() for c,(u,f) in enumerate(zip(velocity,density)))
            sp = (nodal*force).sum()
            info.update(divergence_l2=(self.grid.volume*divergence(velocity,self.grid.spacing).square().sum()).sqrt().item(),
                solid_power=sp.item(),fluid_power=fp.item(),power_error=abs((fp-sp).item()))
        if self.owner.closed_cap:
            report = problem.report()
            # Physical checks supplement the original full displacement residual.
            if (report['constraint_residual_norm_ml_s'] > 1e-5
                    or report['interpolated_velocity_constraint_norm_ml_s'] > 1e-5):
                raise NonlinearFailure('coupled cap true relative flux exceeds target',result)
            info['cap'] = report
            info['_cap_state'] = (problem.last_cap_pressure.clone(),state.x.clone())
            if diagnostics:
                # Fluid cap pressure power is distinct from IB solid power.
                info['power_error'] = abs(fp.item()-sp.item()-report['cap_fluid_power_erg_s'])
        return accepted, info


class IsovolumicAFSEIStepper(AFSEIStepper):
    def __init__(self, mesh, config):
        super().__init__(mesh, config)
        self._iso_context = None
        self.pressure_backend = 'spectral'
        self.response_optimizations = True
        self._pressure_scales = {}
        self.closed_cap = False
        self.cap_patch_width_cells = 3.

    @classmethod
    def from_driver(cls, driver):
        result = cls.__new__(cls)
        result.__dict__.update(driver.__dict__)
        result._steppers = {}
        result._iso_context = None
        result.pressure_backend = 'spectral'
        result.response_optimizations = True
        result._pressure_scales = {}
        result.closed_cap = False
        result.cap_patch_width_cells = 3.
        return result

    def execution_settings(self):
        return dict(pressure_backend=self.pressure_backend,response_optimizations=self.response_optimizations,
                    pressure_scales={str(k):v for k,v in self._pressure_scales.items()},
                    closed_cap=self.closed_cap,cap_patch_width_cells=self.cap_patch_width_cells)

    def set_execution(self, *, pressure_backend=None,response_optimizations=None,pressure_scales=None,
                      closed_cap=None,cap_patch_width_cells=None):
        if pressure_backend is not None and pressure_backend not in ('spectral','multigrid'):
            raise ValueError('invalid isovolumic pressure backend')
        if response_optimizations is not None and type(response_optimizations) is not bool:
            raise ValueError('response optimizations must be boolean')
        if closed_cap is not None and type(closed_cap) is not bool:
            raise ValueError('closed cap must be boolean')
        if cap_patch_width_cells is not None and (not isfinite(cap_patch_width_cells) or cap_patch_width_cells < 2.):
            raise ValueError('cap patch width >=2 required')
        if pressure_scales is not None:
            if not isinstance(pressure_scales,dict): raise ValueError('pressure scales must be a map')
            scales = {}
            for key,value in pressure_scales.items():
                k=int(key)
                if str(k)!=key or not 1<=k<=self.maximum_ticks or not isfinite(value) or value<=0:
                    raise ValueError('invalid pressure unit scaling')
                scales[k]=value
        if pressure_backend is not None: self.pressure_backend = pressure_backend
        if response_optimizations is not None: self.response_optimizations = response_optimizations
        if pressure_scales is not None: self._pressure_scales = scales
        if closed_cap is not None: self.closed_cap = closed_cap
        if cap_patch_width_cells is not None: self.cap_patch_width_cells = cap_patch_width_cells
        self._steppers = {}

    def _fsi_stepper(self, step_ticks):
        if step_ticks not in self._steppers:
            dt = step_ticks*self.protocol.dt_ep_s
            cfg = replace(self.config.fsi,time=TimeConfig(dt,dt))
            self._steppers[step_ticks] = IsovolumicEndpointStepper(self,cfg)
        return self._steppers[step_ticks]

    def lock(self, state, *, arterial_pc_mmhg=85.):
        if self.config.mechanics_profile != 'source': raise ValueError('source mechanics required')
        result = IsovolumicState(state,MACEndocardialPressure(self.model).volume(state.fsi.x).item(),
            self.config.prescribed_endocardial_pressure_mmhg,arterial_pc_mmhg)
        self.validate_control(result)
        return result

    def validate_control(self, state):
        if not isinstance(state,IsovolumicState): raise ValueError('isovolumic state required')
        if (not all(isfinite(v) for v in (state.target_volume_ml,state.cavity_pressure_mmhg,state.arterial_pc_mmhg))
                or state.target_volume_ml <= 0 or type(state.accepted_closed_steps) is not int
                or state.accepted_closed_steps < 0): raise ValueError('invalid isovolumic state')
        volume = MACEndocardialPressure(self.model).volume(state.fsi.x).item()
        if abs(volume-state.target_volume_ml) > VOLUME_TOLERANCE_ML:
            raise ValueError('saved isovolumic geometry violates fixed target')
        if (state.cap_pressure is None) != (state.cap_geometry is None):
            raise ValueError('cap pressure and frozen geometry must be saved together')
        if state.cap_pressure is not None:
            if (not self.closed_cap or state.cap_geometry.shape != state.fsi.x.shape
                    or state.cap_pressure.ndim != 1
                    or any(v.dtype != state.fsi.x.dtype or v.device != state.fsi.x.device
                           for v in (state.cap_pressure,state.cap_geometry))
                    or not torch.isfinite(state.cap_geometry).all() or not torch.isfinite(state.cap_pressure).all()):
                raise ValueError('invalid coupled cap state')
            if state.fsi.previous_x is None or not torch.equal(state.cap_geometry,state.fsi.previous_x):
                raise ValueError('cap reaction must retain the accepted pre-step geometry')

    def force_for_state(self, state):
        force = self.model.force(state.fsi.x,state.fsi.time)+MACEndocardialPressure(self.model).force(
            state.fsi.x,state.cavity_pressure_mmhg-self.config.prescribed_endocardial_pressure_mmhg)
        if state.cap_pressure is not None:
            from .mac_cap_coupled import closed_rows
            ticks = round(state.force_load.dt_s/self.protocol.dt_ep_s)
            rows = closed_rows(self._fsi_stepper(ticks),state.cap_geometry)
            if rows.count != len(state.cap_pressure): raise ValueError('cap pressure partition mismatch')
            force = force+rows.reaction(state.cap_pressure)
        return force

    @torch.no_grad()
    def step(self, old, **kwargs):
        self.validate_control(old)
        self._iso_context = old
        try:
            accepted, info = super().step(old.ep, **kwargs)
            trial = flow_driven_trial(old.arterial_pc_mmhg,0.,info['accepted_dt_s'],old.windkessel)
            result = replace(old,ep=accepted,cavity_pressure_mmhg=info['isovolumic']['cavity_pressure_mmhg'],
                arterial_pc_mmhg=trial.pc_mmhg,accepted_closed_steps=old.accepted_closed_steps+1)
            if self.closed_cap:
                pressure,geometry = info.pop('_cap_state')
                result = replace(result,cap_pressure=pressure,cap_geometry=geometry)
            self.validate_control(result)
            info['isovolumic'].update(arterial_pc_mmhg=result.arterial_pc_mmhg,
                assumed_closed_outlet_flow_ml_s=0.,outlet_opening_pressure_reached=
                result.cavity_pressure_mmhg > result.arterial_pc_mmhg)
            return result, info
        finally:
            self._iso_context = None
