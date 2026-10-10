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

VOLUME_TOLERANCE_ML = 1e-3


@dataclass(frozen=True)
class IsovolumicState:
    ep: CoupledState
    target_volume_ml: float
    cavity_pressure_mmhg: float
    arterial_pc_mmhg: float = 85.
    windkessel: WindkesselParameters = WindkesselParameters()
    accepted_closed_steps: int = 0

    def __getattr__(self, name):
        return getattr(self.ep, name)


class IsovolumicProblem:
    def __init__(self, driver, state, stencil, departure, control):
        self.driver, self.state, self.stencil, self.departure = driver, state, stencil, departure
        self.control = control
        self.load = MACEndocardialPressure(driver.model)
        self.time = driver.flow.dt
        self.pressure_initial = state.pressure
        self.evaluations = self.jacobian_actions = self.tangent_assemblies = 0
        self.preconditioner_applications = 0
        self.last_z = self.last_residual = self.last_data = None
        self.volume_scale = torch.linalg.vector_norm(torch.func.grad(self.load.volume)(state.x)).item()
        mobility = self.response(self.load.force(state.x, 1.))
        size = mobility.norm().item()
        if not isfinite(size) or size <= 0 or not isfinite(self.volume_scale) or self.volume_scale <= 0:
            raise ValueError('degenerate isovolumic pressure mobility or volume gradient')
        self.pressure_scale = 1./size

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
        c = torch.func.grad(self.load.volume)(x).flatten()/self.volume_scale
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

    @torch.no_grad()
    def physical_step(self, state, new_time, diagnostics):
        control = self.owner._iso_context
        if control is None: raise ValueError('missing immutable isovolumic trial control')
        self.solid.validate(state.x)
        stencil = self.transfer.prepare(state.x)
        departure = self.flow.advect(state.velocity)
        before = (self.flow.calls, self.flow.pressure_cycles, self.flow.helmholtz_sweeps,
                  self.mass_calls, self.mass_iterations)
        problem = IsovolumicProblem(self, state, stencil, departure, control)
        initial = torch.zeros_like(state.x)
        if state.previous_x is not None:
            initial = (state.x-state.previous_x)*(self.flow.dt/(state.previous_dt or self.flow.dt))
        z = torch.cat((initial.flatten(), initial.new_zeros(1)))
        r0 = problem.residual(z)
        settings = self.config.nonlinear
        # Retain the original mechanical tolerance, independent of constraint scaling.
        tolerance = max(settings.atol, settings.rtol*r0[:-1].norm().item())
        options = replace(settings, rtol=0., atol=tolerance)
        result = newton(problem.residual, z, validate=problem.validate,
            linearization_factory=problem.linearization, preconditioner_factory=problem.preconditioner,
            options=options)
        residual = problem.accepted(result.x)
        x, p, volume, force, velocity, pressure, flow, nodal, density, spread, interpolation = problem.last_data
        error = (volume-control.target_volume_ml).item()
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
        return accepted, info


class IsovolumicAFSEIStepper(AFSEIStepper):
    def __init__(self, mesh, config):
        super().__init__(mesh, config)
        self._iso_context = None

    @classmethod
    def from_driver(cls, driver):
        result = cls.__new__(cls)
        result.__dict__.update(driver.__dict__)
        result._steppers = {}
        result._iso_context = None
        return result

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

    def force_for_state(self, state):
        return self.model.force(state.fsi.x,state.fsi.time)+MACEndocardialPressure(self.model).force(
            state.fsi.x,state.cavity_pressure_mmhg-self.config.prescribed_endocardial_pressure_mmhg)

    @torch.no_grad()
    def step(self, old, **kwargs):
        self.validate_control(old)
        self._iso_context = old
        try:
            accepted, info = super().step(old.ep, **kwargs)
            trial = flow_driven_trial(old.arterial_pc_mmhg,0.,info['accepted_dt_s'],old.windkessel)
            result = replace(old,ep=accepted,cavity_pressure_mmhg=info['isovolumic']['cavity_pressure_mmhg'],
                arterial_pc_mmhg=trial.pc_mmhg,accepted_closed_steps=old.accepted_closed_steps+1)
            self.validate_control(result)
            info['isovolumic'].update(arterial_pc_mmhg=result.arterial_pc_mmhg,
                assumed_closed_outlet_flow_ml_s=0.,outlet_opening_pressure_reached=
                result.cavity_pressure_mmhg > result.arterial_pc_mmhg)
            return result, info
        finally:
            self._iso_context = None
