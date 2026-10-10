"""Closed-cap pressure reactions inside the original BE--BE FSI Newton solve.

Unknowns are displacement, source cavity pressure and weak cap pressures.
Cap geometry and IB transfer are frozen at the accepted old configuration,
as in the original MAC scheme. There is no outer complete-FSI iteration.
"""
from time import perf_counter
import torch
from .mac_isovolumic import IsovolumicProblem
from .mac_ports import MACBasalPorts
from .mac_seal import prepare_local_cap, MACCapProjection
from ..nonlinear import normalized_linear_action
from ..mac.open_boundary import face_weights


def closed_rows(endpoint, x):
    if not hasattr(endpoint, '_material_cap'):
        endpoint._material_cap = MACBasalPorts(endpoint.model.mesh.X,
            endpoint.model.endo_faces, endpoint.grid, with_ports=False)
    return prepare_local_cap(endpoint._material_cap.prepare(x),
        patch_width_cells=endpoint.owner.cap_patch_width_cells, regions=(0,1,2))


class CapIsovolumicProblem(IsovolumicProblem):
    def __init__(self, *args):
        super().__init__(*args)
        self.rows = closed_rows(self.driver, self.state.x)
        self.mechanical_size = self.state.x.numel()
        # Convert a cap residual to cm; choose pressure units giving an
        # approximately unit kinetic diagonal. Acceptance checks physical flux.
        self.cap_scale = self.driver.flow.rho*self.rows.area/(
            self.driver.flow.dt**2*self.rows.diagonal)
        self.last_cap_pressure = None
        self.setup_seconds = 0.
        self.setup_pressure_calls = 0
        self.preconditioner_rebuilt = False
        self._prepare_preconditioner()

    @property
    def extra_unknowns(self): return 1+self.rows.count

    def unpack(self, z):
        y = z[:self.mechanical_size].reshape_as(self.state.x)
        p = self.control.cavity_pressure_mmhg+self.pressure_scale*z[self.mechanical_size]
        return y, self.state.x+y, p

    def cap_pressure(self, z):
        return self.cap_scale*z[self.mechanical_size+1:]

    def cap_density(self, pressure):
        """L p = S C^T p - W^-1 B^T p, the adjoint of relative flux."""
        d = self.driver
        d.transfer.reset_warm_start()
        spread, _ = d.transfer.spread(self.rows.reaction(pressure), self.stencil)
        return tuple(a-b for a,b in zip(spread,self.rows.lift(pressure)))

    def sensitivity(self, force, pressure=None):
        d = self.driver
        d.transfer.reset_warm_start()
        if pressure is not None:
            force = force+self.rows.reaction(pressure)
        density, _ = d.transfer.spread(force, self.stencil)
        if pressure is not None:
            # One spread also prevents aliasing a transfer workspace by a
            # second spread before the first density has been consumed.
            density = tuple(a-b for a,b in zip(density,self.rows.lift(pressure)))
        velocity = d.flow.response(density)
        nodal, _ = d.transfer.interpolate(velocity,self.stencil)
        return velocity, nodal

    def _prepare_preconditioner(self):
        """Cached approximate Schur columns; only true BE residuals accept steps.

        Build with the kinetic pressure projection, without repeated solid,
        ODE or complete-FSI solves. Reuse while material points move <0.1 cell
        and the patches retain a one-to-one substantial overlap. Small
        quadrature bin changes remap the cached columns by area overlap;
        the true current constraint rows remain unchanged. Units rescale.
        """
        d, rows = self.driver, self.rows
        cache = getattr(d,'_cap_schur_cache',None)
        reusable = (cache is not None and cache['patches'].shape == rows.patches.shape
            and cache['h'].shape == (rows.count,rows.count)
            and torch.equal(cache['selected'],rows.selected)
            and (self.state.x-cache['x']).abs().max().item() < .1*min(d.grid.spacing))
        if reusable:
            if torch.equal(cache['patches'],rows.patches):
                ratio = self.cap_scale/cache['scale']
                self.cap_b = cache['b']*ratio[None,:]
                self.cap_h = cache['h']*ratio[None,:]
                return
            overlap = rows.area.new_zeros((rows.count,rows.count))
            point_area = rows.prepared.area_vectors[rows.selected].norm(dim=-1)
            overlap.view(-1).index_add_(0,rows.patches*rows.count+cache['patches'],point_area)
            restriction = overlap/overlap.sum(1)[:,None]
            matched = restriction.argmax(1)
            if ((restriction.max(1).values > .8).all()
                    and len(torch.unique(matched)) == rows.count):
                prolongation = overlap.T/overlap.sum(0)[:,None]
                units = prolongation*self.cap_scale[None,:]/cache['scale'][:,None]
                self.cap_b = cache['b']@units
                self.cap_h = restriction@cache['h']@units
                return
        started = perf_counter()
        projector = MACCapProjection(rows,d.flow.pressure_solver)
        columns, h = [], []
        unit = rows.area.new_zeros(rows.count)
        for j in range(rows.count):
            unit.zero_(); unit[j]=self.cap_scale[j]
            density = self.cap_density(unit)
            projected, _ = projector.divergence_project(density)
            velocity = tuple(d.flow.dt/d.flow.rho*u for u in projected)
            nodal, _ = d.transfer.interpolate(velocity,self.stencil)
            columns.append((-d.flow.dt*nodal).flatten())
            h.append(d.flow.dt*(rows.flow(velocity)-rows.material_flow(nodal))/rows.area)
        self.cap_b = torch.stack(columns,1)
        self.cap_h = torch.stack(h,1)
        d._cap_schur_cache = dict(x=self.state.x.clone(),patches=rows.patches.clone(),
            selected=rows.selected.clone(),scale=self.cap_scale.clone(),
            b=self.cap_b,h=self.cap_h)
        if self.state.x.is_cuda: torch.cuda.synchronize(self.state.x.device)
        self.setup_seconds = perf_counter()-started
        self.setup_pressure_calls = projector.pressure_calls
        self.preconditioner_rebuilt = True

    def validate(self,z):
        super().validate(z)
        if not torch.isfinite(self.cap_pressure(z)).all():
            raise FloatingPointError('nonfinite cap pressure')

    def residual(self,z):
        state_version = None if torch.is_inference(self.state.x) else self.state.x._version
        if (self.driver.owner.response_optimizations and self.last_z is not None
                and state_version is not None and state_version == self._last_state_version
                and torch.equal(z,self.last_z)):
            self.cache_hits += 1
            return self.last_residual.clone()
        self.validate(z)
        d, rows = self.driver, self.rows
        y,x,p = self.unpack(z); cap_p = self.cap_pressure(z)
        base = d.owner.config.prescribed_endocardial_pressure_mmhg
        force = d.solid.force(x,self.time)+self.load.force(x,p-base)+rows.reaction(cap_p)
        d.transfer.reset_warm_start()
        density,spread = d.transfer.spread(force,self.stencil)
        density = tuple(a-b for a,b in zip(density,rows.lift(cap_p)))
        velocity,pressure,flow = d.flow.advance(self.departure,density,self.pressure_initial)
        nodal,interpolation = d.transfer.interpolate(velocity,self.stencil)
        volume = self.load.volume(x)
        cap_residual = (d.flow.dt*rows.flow(velocity)-rows.material_flow(y))/rows.area
        residual = torch.cat(((y-d.flow.dt*nodal).flatten(),
            ((volume-self.control.target_volume_ml)/self.volume_scale).reshape(1),cap_residual))
        if not torch.isfinite(residual).all(): raise FloatingPointError('nonfinite coupled cap residual')
        self.pressure_initial = pressure
        self.last_z,self.last_residual = z.clone(),residual.clone()
        self._last_state_version = state_version
        self.last_cap_pressure = cap_p.clone()
        self.last_data = x,p,volume,force,velocity,pressure,flow,nodal,density,spread,interpolation
        self.evaluations += 1
        return residual

    def linearization(self,z):
        d,rows = self.driver,self.rows
        _,x,p = self.unpack(z)
        if d.assembler is None: d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(x,self.time)
        self.tangent_assemblies += 1
        self.c = torch.func.grad(self.load.volume)(x).flatten()/self.volume_scale
        up,wp = self.sensitivity(self.load.force(x,1.)*self.pressure_scale)
        self.b = (-d.flow.dt*wp).flatten()
        self.hp = d.flow.dt*(rows.flow(up)-rows.material_flow(wp))/rows.area
        offset = p-d.owner.config.prescribed_endocardial_pressure_mmhg
        def action(v):
            self.jacobian_actions += 1
            dy = v[:self.mechanical_size].reshape_as(x)
            df = torch.sparse.mm(tangent,dy.reshape(-1,1)).reshape_as(x)
            df = df+self.load.action(x,offset,dy)
            # The source cavity pressure has its own current-geometry column.
            df += self.load.force(x,1.)*(self.pressure_scale*v[self.mechanical_size])
            du,dw = self.sensitivity(df,self.cap_scale*v[self.mechanical_size+1:])
            cap = (d.flow.dt*rows.flow(du)-rows.material_flow(dy))/rows.area
            return torch.cat(((dy-d.flow.dt*dw).flatten(),
                (self.c@dy.flatten()).reshape(1),cap))
        return normalized_linear_action(action)

    def preconditioner(self,z):
        c,b,rows = self.c,self.b,self.rows
        h = rows.area.new_empty((rows.count+1,rows.count+1))
        h[0,0] = -c@b; h[0,1:] = -c@self.cap_b
        h[1:,0] = self.hp; h[1:,1:] = self.cap_h
        lu,pivots = torch.linalg.lu_factor(h)
        def apply(v):
            self.preconditioner_applications += 1
            y = v[:self.mechanical_size]
            lower = torch.cat(((v[self.mechanical_size]-c@y).reshape(1),
                v[self.mechanical_size+1:]+rows.material_flow(y.reshape_as(self.state.x))/rows.area))
            q = torch.linalg.lu_solve(lu,pivots,lower[:,None]).flatten()
            dy = y-b*q[0]-self.cap_b@q[1:]
            return torch.cat((dy,q))
        return apply

    def report(self):
        d,rows = self.driver,self.rows
        x,_,_,_,u,_,_,nodal,_,_,_ = self.last_data
        geometric_velocity = (x-self.state.x)/d.flow.dt
        # Check both the equation velocity and the accepted displacement velocity.
        flow_residual = rows.flow(u)-rows.material_flow(geometric_velocity)
        interpolation_residual = rows.flow(u)-rows.material_flow(nodal)
        lift = rows.lift(self.last_cap_pressure)
        reaction = rows.reaction(self.last_cap_pressure)
        fluid_power = -sum((a*b*face_weights(d.grid,c,x)).sum()
            for c,(a,b) in enumerate(zip(u,lift)))
        solid_power = (reaction*nodal).sum()
        report = dict(rows.diagnostics(u,geometric_velocity),
            patches=rows.count,complete_cap_closed_in_patch_average=True,
            anatomical_valves_present=False,pointwise_impermeability_established=False,
            constraint_residual_norm_ml_s=flow_residual.norm().item(),
            interpolated_velocity_constraint_norm_ml_s=interpolation_residual.norm().item(),
            reaction_norm_dyn=reaction.norm().item(),pressure_max_abs_dyn_cm2=self.last_cap_pressure.abs().max().item(),
            cap_fluid_power_erg_s=fluid_power.item(),cap_solid_power_erg_s=solid_power.item(),
            cap_total_power_erg_s=(fluid_power+solid_power).item(),
            cap_power_relative_error=((fluid_power+solid_power).abs()/
                (fluid_power.abs()+solid_power.abs()).clamp_min(1e-30)).item(),
            coupled_solid_advanced=True,prescribed_material_motion=False,
            frozen_cap_geometry=True,preconditioner_rebuilt=self.preconditioner_rebuilt,
            preconditioner_setup_seconds=self.setup_seconds,
            preconditioner_setup_pressure_calls=self.setup_pressure_calls)
        return report
