"""Active EP/Land and measured-flow RCR on the pressure-balanced MAC branch.

Outlet-interval validation, not the four-phase cardiac controller. The source
PK1 and unload law are unchanged. An optional prescribed endocardial follower
traction is a boundary load, not a wall-pressure feedback controller. No outer
full-FSI iteration or fitted fluid mesh is used.
"""
from dataclasses import dataclass, replace
from math import isfinite
import torch
from .driver import AFSEIStepper, CoupledState, StrainHistory
from .mac_isovolumic import IsovolumicEndpointStepper
from .mac_balanced_preload import (BalancedCapProblem, EndocardialAndCapPressure,
                                  read_result, reuse_validated_loading)
from .mac_transport import closed_volume_gradient
from .mac_pressure_mode import PressureModeTransfer
from .mac_cap_coupled import closed_rows
from .mac_rcr_coupled import outlet_rows
from .mac_preload import MACEndocardialPressure
from .mac_pressure_audit import cavity_masks, pressure_statistics
from .windkessel import WindkesselParameters, flow_driven_trial, MMHG_TO_DYN_CM2 as C
from .tissue import TissueState
from ..config import TimeConfig
from ..mac.paper_coupling import BEProblem, PaperState
from ..mac.grid import divergence
from ..mac.open_boundary import face_weights
from ..nonlinear import newton, normalized_linear_action, NonlinearFailure
from .mac_compact_cap import cap_reaction,add_cap_lift
from ..krylov_workspace import KrylovWorkspace
from .gao_filling import filling_port_law, filling_port_accepted


@dataclass(frozen=True)
class BalancedState:
    ep: CoupledState
    target_volume_ml: float
    cap_pressure: torch.Tensor
    cap_geometry: torch.Tensor
    mode: str = 'closed'
    arterial_pc_mmhg: float = 85.
    windkessel: WindkesselParameters = WindkesselParameters()
    outlet_open: bool = False
    accepted_port_steps: int = 0
    cap_mode: str | None = None
    filling_pressure_mmhg: float = 8.
    # Load used in the last accepted physical solve, including at phase changes.
    wall_pressure_mmhg: float = 0.
    # Accepted pressure-reference completion on the same virtual cap. This is
    # a prescribed force, not a new pressure unknown or kinematic constraint.
    cap_reference_mmhg: float = 0.

    def __getattr__(self, name): return getattr(self.ep, name)


class BalancedDynamicProblem(BalancedCapProblem):
    def __init__(self, endpoint, state, stencil, departure, control):
        BEProblem.__init__(self, endpoint, state, stencil, departure)
        self.control = control
        self.load = MACEndocardialPressure(endpoint.model)
        self.rows = (outlet_rows(endpoint,state.x,label=1 if control.mode=='filling' else 2)
                     if control.mode != 'closed' else stencil.rows)
        if getattr(endpoint,'cut_boundary',False):
            self.rows=stencil.rows
            self.cut_storage=endpoint.flow.domain.storage
        self.mechanical_size = state.x.numel()
        self.cap_scale = endpoint.flow.rho*self.rows.area/(endpoint.flow.dt**2*self.rows.diagonal)
        self.last_z = self.last_residual = self.last_cap_pressure = None
        self.cache_hits = 0; self.setup_seconds = 0.; self.setup_pressure_calls = 0
        self.preconditioner_rebuilt = False
        self._prepare_preconditioner()
        self.last_open = False
        self.last_inlet_open = False
        self._cap_lu_cache = {}
        self.current_stencil = stencil
        self.closed_load = EndocardialAndCapPressure(endpoint.model)
        self.old_volume_gradient = closed_volume_gradient(state.x,self.load.faces,self.load.rim)
        unit=getattr(stencil,'cap_unit_pressure',None)
        if unit is None:unit=torch.full_like(stencil.rows.area,C)
        self.old_cap_covector = stencil.rows.reaction(unit)/C
        self.impedance = flow_driven_trial(control.arterial_pc_mmhg, 0., endpoint.flow.dt,
            control.windkessel).impedance_mmhg_s_per_ml+endpoint.owner.valve_resistance_mmhg_s_per_ml

    def cap_row_units(self, value, *, inverse=False):
        # The RCR pressure-law row has its own units; only closed rows are scaled.
        area = (self.rows.area[:-2] if getattr(self.driver,'cut_boundary',False) else
                self.rows.area[:-1] if self.control.mode != 'closed' else self.rows.area)
        n = len(area); w = area/area.norm()
        result = value.clone(); k = -15/16 if inverse else 15.
        result[:n] += k*w*(w*value[:n]).sum()
        return result

    def pressure_mode(self,x):
        # Exact integral of the quadratic P1 volume gradient along this BE
        # displacement. Only the numerical transfer mode changes: source
        # material forces and the frozen paired cap reaction are untouched.
        middle = closed_volume_gradient((self.state.x+x)/2,self.load.faces,self.load.rim)
        current = closed_volume_gradient(x,self.load.faces,self.load.rim)
        average = (self.old_volume_gradient+4*middle+current)/6
        if getattr(self.driver,'cut_boundary',False):
            average=average+self.cut_storage.discrete(self.state.x,x)
        mode = -C*(average-self.old_cap_covector)
        return replace(self.stencil,mode=mode,dual=mode/mode.square().sum())

    def sensitivity(self, force, pressure=None):
        d = self.driver; pressure = torch.zeros_like(self.rows.area) if pressure is None else pressure
        d.transfer.reset_warm_start()
        remainder, _, _ = d.transfer.relative_parts(force+cap_reaction(self.rows,pressure,check_finite=False),
            pressure, self.stencil, rows=self.rows,check_finite=False)
        u = d.flow.response(remainder)
        nodal, _ = d.transfer.interpolate(u, self.stencil,check_finite=False)
        return u, nodal

    def residual(self, z):
        self.validate(z)
        version=BEProblem._version(self.state.x)
        if (self.driver.owner.response_optimizations and version is not None and version==self._last_state_version
                and self.last_z is not None and torch.equal(z,self.last_z)):
            self.cache_hits += 1
            return self.last_residual.clone()
        d, rows = self.driver, self.rows
        y, x, _ = self.unpack(z); bp = self.cap_pressure(z)
        stencil = self.pressure_mode(x)
        self.current_stencil = stencil
        # The mass RHS and BE/pressure true residuals validate all resulting
        # fields. Avoid intermediate CPU barriers on cap scalars and velocity.
        force = d.solid.force(x, self.time)+cap_reaction(rows,
            bp+self.control.cap_reference_mmhg*C,check_finite=False)
        if self.control.wall_pressure_mmhg:
            force=force+self.load.force(x,self.control.wall_pressure_mmhg)
        d.transfer.reset_warm_start()
        remainder, amplitude, spread = d.transfer.relative_parts(force, bp, stencil, rows=rows,check_finite=False)
        known = amplitude*stencil.potential
        u, p, flow = d.flow.advance(self.departure, remainder, self.pressure_initial-known)
        p = p+known
        density = tuple(f+amplitude*g for f,g in zip(remainder,stencil.gradient))
        nodal, interpolation = d.transfer.interpolate(u, stencil,check_finite=False)
        rates = rows.flow(u)-rows.material_flow(nodal)
        lower = rates/rows.area
        self.last_open = False; self.last_inlet_open = False
        self.last_circuit = None; self.last_gap = None
        if self.control.mode == 'rcr':
            circuit = flow_driven_trial(self.control.arterial_pc_mmhg, rates[-1], d.flow.dt,
                                       self.control.windkessel)
            gap = circuit.outlet_mmhg+d.owner.valve_resistance_mmhg_s_per_ml*rates[-1]-bp[-1]/C
            self.last_open = not d.owner.one_way or (lower[-1]>gap).item()
            lower[-1] = gap if self.last_open else lower[-1]
            self.last_circuit, self.last_gap = circuit, gap
        elif self.control.mode == 'filling':
            lower[-1],self.last_inlet_open = filling_port_law(lower[-1],bp[-1]/C,
                self.control.filling_pressure_mmhg,getattr(d.owner,'inlet_one_way',False))
        if getattr(d,'cut_boundary',False):
            # The basal opening is an interior connection with no prescribed
            # pressure sheet. Circulation acts at the physical tract cut.
            lower[-2:] = -bp[-2:]/C
            self.last_circuit=d.flow.last_circuit
            self.last_open=self.control.mode=='rcr'
            self.last_inlet_open=self.control.mode=='filling'
        residual = torch.cat(((y-d.flow.dt*nodal).flatten(), self.cap_row_units(lower)))
        if not torch.isfinite(residual).all(): raise FloatingPointError('nonfinite balanced dynamic residual')
        self.pressure_initial = p
        self.last_z, self.last_residual = z.clone(), residual.clone()
        self._last_state_version=version
        self.last_cap_pressure = bp.clone()
        self.last_data = x, 0., self.load.volume(x), force, u, p, flow, nodal, density, spread, interpolation
        self.evaluations += 1
        return residual

    def linearization(self, z, *, reuse_tangent=False):
        self.accepted(z)
        d, rows = self.driver, self.rows; x = self.unpack(z)[1]
        if d.assembler is None: d.assembler = d.model.tangent_factory(d.owner.balanced_tangent_chunk_size)
        reusable=(reuse_tangent and self._tangent is not None
            and self._tangent_open==(self.last_open,self.last_inlet_open)
            and (x-self._tangent_x).abs().max().item()<.05*min(d.grid.spacing))
        if not reusable:
            self._tangent = d.assembler.assemble(x,self.time)
            self._tangent_x=x.clone(); self._tangent_open=(self.last_open,self.last_inlet_open)
            self.tangent_assemblies += 1
        tangent=self._tangent
        self.linear_open = self.last_open
        self.linear_inlet_open = self.last_inlet_open
        stencil = self.current_stencil
        force,u = self.last_data[3],self.last_data[4]
        d.transfer.reset_warm_start()
        from ..mac.assembled_transfer import AssembledP1Transfer
        base_nodal,_ = (d.transfer.base.interpolate(u,stencil.original,check_finite=False)
            if isinstance(d.transfer.base,AssembledP1Transfer)
            else d.transfer.base.interpolate(u,stencil.original))
        work = d.transfer.target_power(u,stencil.target)-(stencil.mode*base_nodal).sum()
        amplitude = (stencil.dual*force).sum()
        # The P1 boundary force derivative is affine in x. This evaluates
        # (2*D F(midpoint)+D F(endpoint))/6 exactly with ONE boundary action.
        derivative_geometry=(self.state.x+2*x)/3
        tract_derivative=(self.cut_storage.derivative(self.state.x,x)
                          if getattr(d,'cut_boundary',False) else None)
        def action(v):
            self.jacobian_actions += 1
            dy = v[:self.mechanical_size].reshape_as(x)
            df = torch.sparse.mm(tangent, dy.reshape(-1,1)).reshape_as(x)
            if self.control.wall_pressure_mmhg:
                df=df+self.load.action(x,self.control.wall_pressure_mmhg,dy)
            dbp = self.cap_scale*v[self.mechanical_size:]
            df = df+cap_reaction(rows,dbp,check_finite=False)
            dm = d.mode_derivative(derivative_geometry,dy)
            if tract_derivative is not None:dm=dm-C*tract_derivative(dy)
            da = (dm-2*stencil.dual*(stencil.mode*dm).sum())/stencil.mode.square().sum()
            dalpha = (stencil.dual*df).sum()+(da*force).sum()
            d.transfer.reset_warm_start()
            remainder,_ = d.transfer.base.spread(
                df-dalpha*stencil.mode-amplitude*dm,stencil.original)
            unit=C if stencil.cap_unit_pressure is None else stencil.cap_unit_pressure
            density=add_cap_lift(rows,dalpha*unit-dbp,remainder,check_finite=False)
            if getattr(d,'cut_boundary',False):
                density=tuple(f+dalpha*(g+b) for f,g,b in zip(density,stencil.gradient,stencil.cut_unit_lift))
            du = d.flow.response(density)
            dw,_ = d.transfer.interpolate(du,stencil,check_finite=False)
            dw = dw+da*work-stencil.dual*(dm*base_nodal).sum()
            dq = rows.flow(du)-rows.material_flow(dw)
            lower = dq/rows.area
            if self.control.mode == 'rcr' and self.linear_open:
                lower[-1] = self.impedance*dq[-1]-dbp[-1]/C
            elif self.control.mode == 'filling' and self.linear_inlet_open:
                lower[-1] = -dbp[-1]/C
            if getattr(d,'cut_boundary',False):
                lower[-2:] = -dbp[-2:]/C
            return torch.cat(((dy-d.flow.dt*dw).flatten(), self.cap_row_units(lower)))
        return normalized_linear_action(action)

    def preconditioner(self, z):
        d, rows = self.driver, self.rows
        key=(self.linear_open,self.linear_inlet_open)
        if key not in self._cap_lu_cache:
            h = self.cap_h/d.flow.dt
            if self.control.mode == 'rcr' and self.linear_open:
                h = h.clone(); h[-1] = self.impedance*rows.area[-1]/d.flow.dt*self.cap_h[-1]
                h[-1,-1] -= self.cap_scale[-1]/C
            elif self.control.mode == 'filling' and self.linear_inlet_open:
                h = h.clone(); h[-1] = 0.; h[-1,-1] = -self.cap_scale[-1]/C
            if getattr(d,'cut_boundary',False):
                h=h.clone();h[-2:]=0.
                h[-2,-2]=-self.cap_scale[-2]/C;h[-1,-1]=-self.cap_scale[-1]/C
            # M H = H + 15*w*(w^T H), without launching one loop per column.
            area=(rows.area[:-2] if getattr(d,'cut_boundary',False) else
                  rows.area[:-1] if self.control.mode!='closed' else rows.area)
            n=len(area); w=area/area.norm()
            h[:n] += 15*w[:,None]*(w@h[:n])[None,:]
            self._cap_lu_cache[key]=torch.linalg.lu_factor(h)
        lu,pivots=self._cap_lu_cache[key]
        def apply(v):
            self.preconditioner_applications += 1
            q = torch.linalg.lu_solve(lu,pivots,v[self.mechanical_size:,None]).flatten()
            return torch.cat((v[:self.mechanical_size]-self.cap_b@q,q))
        return apply


class BalancedEndpoint(IsovolumicEndpointStepper):
    def __init__(self, owner, config):
        if config.execution.mass_backend=='graph':
            config=replace(config,mass_solver=replace(config.mass_solver,
                check_every=owner.mass_graph_check_every))
        config=replace(config,flow=replace(config.flow,initial_sweeps=owner.helmholtz_initial_sweeps))
        super().__init__(owner, config)
        if owner.fluid_solver == 'be-stokes-fgmres':
            from ..mac.be_stokes import BEStokesFlow
            self.flow = BEStokesFlow(self.grid,dt=config.time.dt,rho=config.fluid.rho,mu=config.fluid.mu,
                device=owner.mesh.X.device,dtype=owner.mesh.X.dtype,pressure_options=config.pressure_solver,
                options=config.flow,backend=config.execution.execution_backend,
                pressure_backend=config.execution.pressure_backend)
        if owner.spatial_coupling == 'griffith-luo-unified':
            from .mac_standard_ports import StandardPortTransfer
            self.transfer = StandardPortTransfer(self, self.transfer)
        else:
            self.transfer = PressureModeTransfer(self, level=owner.pressure_mode_level)
        from ..mac.execution import tensor_kernel
        load=EndocardialAndCapPressure(self.model)
        self.mode_derivative=tensor_kernel(lambda x,v: -.5*load.action(x,1.,v),self.model.mesh.X.device)
        self.krylov_workspace=KrylovWorkspace()

    @torch.no_grad()
    def physical_step(self, old, new_time, diagnostics):
        d = self; control = d.owner._boundary_context
        if control is None: raise ValueError('immutable balanced boundary context required')
        if getattr(self.owner,'gao_cut_boundary',False):
            from .gao_tracts import prepare_cut_flow
            prepare_cut_flow(self,old,control)
        stencil = self.transfer.prepare(old.x)
        before = self.flow.calls
        problem_class = BalancedDynamicProblem
        if self.owner.spatial_coupling == 'griffith-luo-unified':
            from .mac_standard_ports import StandardPortProblem
            problem_class = StandardPortProblem
        problem = problem_class(self, old, stencil, self.flow.advect(old.velocity), control)
        if getattr(self,'cut_boundary',False): self._last_cut_problem=problem
        initial = torch.zeros_like(old.x)
        if old.previous_x is not None:
            initial = (old.x-old.previous_x)*(self.flow.dt/old.previous_dt)
        bp = control.cap_pressure
        if len(bp) != problem.rows.count or control.cap_mode not in (None,control.mode):
            bp = torch.full_like(problem.rows.area, control.cap_pressure.mean())
        z = torch.cat((initial.flatten(),bp/problem.cap_scale))
        r0 = problem.residual(z); settings = self.config.nonlinear
        tolerance = max(settings.atol, settings.rtol*r0[:old.x.numel()].norm().item())
        tangent_factory=lambda z: problem.linearization(z,reuse_tangent=self.owner.reuse_step_tangent)
        fallback=False
        options=replace(settings,rtol=0.,atol=tolerance)
        try:
            result = newton(problem.residual,z,validate=problem.validate,
                linearization_factory=tangent_factory,preconditioner_factory=problem.preconditioner,
                options=options,krylov_workspace=self.krylov_workspace)
        except NonlinearFailure as exc:
            if not self.owner.reuse_step_tangent: raise
            fallback=True
            result = newton(problem.residual,exc.result.x,validate=problem.validate,
                linearization_factory=problem.linearization,preconditioner_factory=problem.preconditioner,
                options=options,krylov_workspace=self.krylov_workspace)
            result=replace(result,iterations=exc.result.iterations+result.iterations,
                history=exc.result.history+[dict(h,iteration=exc.result.iterations+h['iteration']) for h in result.history])
        residual = problem.accepted(result.x)
        x, _, volume, force, u, p, flow, nodal, density, spread, interpolation = problem.last_data
        rows, bp = problem.rows, problem.last_cap_pressure
        self.check_support(x)
        if getattr(self,'cut_boundary',False):
            # Reject an invalid candidate before it becomes the next accepted
            # starting geometry; reducing dt can then operate from a valid state.
            self.flow.domain.validate_wall(self._gao_material_cap,x)
        courant = self.flow.dt*sum(v.abs().max()/h for v,h in zip(u,self.grid.spacing)).item()
        if residual.norm().item() > tolerance or not isfinite(courant) or courant > self.config.max_courant:
            raise NonlinearFailure('balanced dynamic residual/CFL failed',result)
        geometric = (x-old.x)/self.flow.dt
        actual = rows.flow(u)-rows.material_flow(geometric)
        projected = rows.flow(u)-rows.material_flow(nodal)
        cut=getattr(self,'cut_boundary',False)
        nclosed=(rows.count-2 if cut else rows.count-1 if control.mode != 'closed' else rows.count)
        if max(actual[:nclosed].norm().item(),projected[:nclosed].norm().item()) > 1e-5:
            raise NonlinearFailure('balanced true closed-cap flux failed',result)
        q = actual[-1] if control.mode != 'closed' else volume.new_zeros(())
        if cut and control.mode=='filling':q=actual[-2]
        old_volume = problem.load.volume(old.x)
        mass = (volume-old_volume)/self.flow.dt+q
        if cut:
            tract_old=problem.cut_storage.volume(old.x)
            tract_new=problem.cut_storage.volume(x)
            fixed_q=self.flow.last_boundary['outward_flux_ml_s'].sum().item()
            mass=(volume-old_volume+tract_new-tract_old)/self.flow.dt+fixed_q
        if abs(mass.item()) > 1e-5:
            raise NonlinearFailure('balanced finite-step mass balance failed',result)
        if control.mode == 'closed' and not cut and abs(volume.item()-control.target_volume_ml) > 1e-3:
            raise NonlinearFailure('balanced closed-volume target failed',result)
        lift = rows.lift(bp); reaction = rows.reaction(bp)
        fluid_boundary_power = -sum((a*b*face_weights(self.grid,c,x)).sum() for c,(a,b) in enumerate(zip(u,lift)))
        material_boundary_power = (reaction*nodal).sum()
        expected = -(bp*projected).sum()
        work = abs((fluid_boundary_power+material_boundary_power-expected).item())
        work_scale = max(abs(fluid_boundary_power.item())+abs(material_boundary_power.item())+abs(expected.item()),1.)
        if work/work_scale > 1e-7: raise NonlinearFailure('balanced boundary work failed',result)
        circuit = flow_driven_trial(control.arterial_pc_mmhg,
            q.item() if control.mode=='rcr' else 0.,self.flow.dt,control.windkessel)
        gap = 0.
        boundary_q=(self.flow.last_boundary['outward_flux_ml_s'][0].item()
                    if cut and control.mode!='closed' else q.item())
        if cut and control.mode!='closed':
            self._cut_budget=self.flow.domain.budget(u,rows.prepared,geometric)
            if (bp[-2:]/C).abs().max().item()>1e-7:
                raise NonlinearFailure('old internal pressure sheet is not zero',result)
            if abs(self._cut_budget['fixed_tract_mass_error_ml_s'].item())>1e-5:
                raise NonlinearFailure('physical fixed tract FV mass balance failed',result)
            if control.mode=='rcr':
                circuit=flow_driven_trial(control.arterial_pc_mmhg,boundary_q,self.flow.dt,control.windkessel)
                gap=self.flow.last_boundary['rcr_gap_mmhg'].item()
                if abs(gap)>1e-7 or (d.owner.one_way and boundary_q < -1e-5
                        and not getattr(d.owner,'cut_valve_active_set',False)):
                    raise NonlinearFailure('physical cut RCR law/flow direction failed',result)
            elif (getattr(d.owner,'inlet_one_way',False) and boundary_q>1e-5
                    and not getattr(d.owner,'cut_valve_active_set',False)):
                raise NonlinearFailure('physical inlet reverse flow',result)
        elif control.mode == 'rcr':
            circuit = flow_driven_trial(control.arterial_pc_mmhg,q.item(),self.flow.dt,control.windkessel)
            gap = circuit.outlet_mmhg+d.owner.valve_resistance_mmhg_s_per_ml*q.item()-bp[-1].item()/C
            if (abs(q.item()-projected[-1].item()) > 1e-5
                    or (problem.last_open and abs(gap) > 1e-7)
                    or (d.owner.one_way and q < -1e-5)
                    or (not problem.last_open and (abs(q) > 1e-5 or gap < -1e-7))):
                raise NonlinearFailure('balanced RCR true flow/pressure law failed',result)
        elif control.mode == 'filling' and not cut:
            gap = bp[-1].item()/C-control.filling_pressure_mmhg
            if not filling_port_accepted(q.item(),projected[-1].item(),bp[-1].item()/C,
                    control.filling_pressure_mmhg,problem.last_inlet_open,getattr(d.owner,'inlet_one_way',False)):
                raise NonlinearFailure('balanced filling reservoir/complementarity law failed',result)
        cached = getattr(self,'_pressure_core',None)
        if cached is None or (x-cached[0]).abs().max().item() > .25*min(d.grid.spacing):
            _, core = cavity_masks(x,d.model.endo_faces,d.grid,guard_cells=d.owner.pressure_guard_cells)
            self._pressure_core = (x.clone(),core)
        else: core = cached[1]
        stats = pressure_statistics(p,core)
        info = dict(nonlinear=dict(iterations=result.iterations,residual_norm=residual.norm().item(),
            solver='modified-newton-gmres' if self.owner.reuse_step_tangent else 'newton-gmres',
            reuse_step_tangent=self.owner.reuse_step_tangent,tangent_refresh_fallback=fallback,
            residual_cache_hits=problem.cache_hits,
            tolerance=tolerance,history=result.history,fluid_solves=self.flow.calls-before,
            jacobian_actions=problem.jacobian_actions,tangent_assemblies=problem.tangent_assemblies,
            preconditioner_rebuilt=problem.preconditioner_rebuilt,preconditioner_setup_seconds=problem.setup_seconds),
            balanced=dict(mode=control.mode,volume_ml=volume.item(),
                volume_target_error_ml=volume.item()-control.target_volume_ml,
                mass_error_ml_s=mass.item(),closed_flux_norm_ml_s=actual[:nclosed].norm().item(),
                measured_relative_q_ml_s=q.item() if control.mode=='rcr' else 0.,
                outlet_open=problem.last_open if control.mode=='rcr' else False,
                inlet_open=problem.last_inlet_open if control.mode=='filling' else False,
                supply_pressure_mmhg=control.filling_pressure_mmhg if control.mode=='filling' else None,
                inlet_pressure_gap_mmhg=gap if control.mode=='filling' else None,
                inlet_reservoir_power_erg_s=(-control.filling_pressure_mmhg*C*q.item()
                    if control.mode=='filling' else 0.),
                inlet_flow_ml_s=-q.item() if control.mode=='filling' else 0.,
                projected_relative_q_ml_s=projected[-1].item() if control.mode!='closed' else 0.,
                pressure_core=stats,boundary_pressure_mmhg=bp[-1].item()/C,
                wall_pressure_mmhg=control.wall_pressure_mmhg,
                cap_reference_mmhg=control.cap_reference_mmhg,
                wall_load_power_erg_s=((problem.load.force(x,control.wall_pressure_mmhg)*nodal).sum().item()
                    if control.wall_pressure_mmhg else 0.),
                cap_reference_power_erg_s=((cap_reaction(rows,
                    torch.full_like(bp,control.cap_reference_mmhg*C),check_finite=False)*nodal).sum().item()
                    if control.cap_reference_mmhg else 0.),
                arterial_pc_old_mmhg=control.arterial_pc_mmhg,
                arterial_pc_mmhg=circuit.pc_mmhg if circuit else control.arterial_pc_mmhg,
                boundary_work_relative_error=work/work_scale,
                cap_fluid_power_erg_s=fluid_boundary_power.item(),cap_material_power_erg_s=material_boundary_power.item(),
                transfer_setup_seconds=stencil.report['setup_seconds'],
                volume_transfer=self.owner.volume_transfer,
                spatial_coupling=self.owner.spatial_coupling,
                pressure_mode_correction=self.owner.spatial_coupling=='legacy-balanced',
                pressure_backend=self.flow.pressure_solver.backend,
                fluid_solver=self.owner.fluid_solver,
                source_wall_pressure_unknown_added=False,full_fsi_outer_iterations_added=False,
                pressure_field_shifted=False,actual_port_flow_used=control.mode!='closed'),
            flow=flow,courant=courant,_cap_state=(bp.clone(),old.x.clone()))
        if circuit:
            q = (volume.new_tensor(boundary_q) if cut and control.mode=='rcr'
                 else q if control.mode=='rcr' else volume.new_zeros(()))
            params = control.windkessel; dp = circuit.pc_mmhg-control.arterial_pc_mmhg
            balance = params.compliance_ml_per_mmhg*dp/self.flow.dt-q.item()+(
                circuit.pc_mmhg-params.venous_mmhg)/params.rp_mmhg_s_per_ml
            storage = .5*params.compliance_ml_per_mmhg*((circuit.pc_mmhg-params.venous_mmhg)**2-
                (control.arterial_pc_mmhg-params.venous_mmhg)**2)*C/self.flow.dt
            loss = .5*params.compliance_ml_per_mmhg*dp**2*C/self.flow.dt
            dissipated = ((circuit.pc_mmhg-params.venous_mmhg)**2/params.rp_mmhg_s_per_ml+
                (params.rc_mmhg_s_per_ml+d.owner.valve_resistance_mmhg_s_per_ml)*q.item()**2)*C
            power = (circuit.outlet_mmhg+d.owner.valve_resistance_mmhg_s_per_ml*q.item()-params.venous_mmhg)*q.item()*C
            energy_error=abs(power-storage-loss-dissipated)/max(abs(power)+abs(storage)+abs(loss)+abs(dissipated),1.)
            if abs(balance)>1e-8 or energy_error>1e-8:
                raise NonlinearFailure('balanced RCR balance/energy failed',result)
            info['balanced'].update(rcr_balance_ml_s=balance,
                outlet_law_gap_mmhg=gap,
                rcr_energy_relative_error=energy_error)
        if cut:
            info['balanced'].update(circulation_interface='physical-MAC-tract-cut',
                internal_opening_pressure_sheet_mmhg=bp[-1].item()/C if control.mode!='closed' else None,
                boundary_pressure_mmhg=(self.flow.last_boundary['pressures_mmhg'][0].item()
                    if control.mode!='closed' else bp[-1].item()/C),
                cut_outward_flux_ml_s=boundary_q if control.mode!='closed' else 0.,
                cut_minus_basal_flux_ml_s=boundary_q-actual[-1].item() if control.mode!='closed' else 0.,
                cut_boundary_power_erg_s=self.flow.last_boundary['power_into_fluid_erg_s'].item(),
                tract_geometry=self.flow.domain.metadata,
                open_box_hydrostatic_shortcut=False)
            if control.mode!='closed':
                active=-2 if control.mode=='filling' else -1
                info['balanced'].update(
                    basal_relative_q_ml_s=actual[active].item(),
                    projected_relative_q_ml_s=projected[active].item(),
                    cut_minus_basal_flux_ml_s=boundary_q-actual[active].item(),
                    physical_inlet_flow_ml_s=-boundary_q if control.mode=='filling' else 0.,
                    physical_outlet_flow_ml_s=boundary_q if control.mode=='rcr' else 0.,
                    inlet_flow_ml_s=-boundary_q if control.mode=='filling' else 0.,
                    inlet_reservoir_power_erg_s=(self.flow.last_boundary['power_into_fluid_erg_s'].item()
                        if control.mode=='filling' else 0.))
            info['balanced'].update(tract_volume_old_ml=tract_old,tract_volume_ml=tract_new,
                tract_volume_rate_ml_s=(tract_new-tract_old)/self.flow.dt,
                cavity_plus_tract_mass_error_ml_s=mass.item(),
                internal_pressure_sheets_mmhg=(bp[-2:]/C).tolist(),
                tract_fv_mass_error_ml_s=(self._cut_budget['fixed_tract_mass_error_ml_s'].item()
                    if control.mode!='closed' else 0.),
                tract_fv_control_volume=('shared-collar-union' if self.flow.domain.shared_top_index is not None
                    else 'single-tract'))
        if diagnostics:
            fp = sum((v*f*face_weights(self.grid,c,x)).sum() for c,(v,f) in enumerate(zip(u,density)))
            sp = (force*nodal).sum()
            transfer_error = abs((fp-sp-fluid_boundary_power).item())
            transfer_relative = transfer_error/max(abs(fp.item())+abs(sp.item())+abs(fluid_boundary_power.item()),1.)
            if transfer_relative > 1e-7:
                raise NonlinearFailure('balanced full transfer work failed',result)
            info['balanced']['transfer_power_relative_error'] = transfer_relative
            info.update(divergence_l2=(self.grid.volume*divergence(u,self.grid.spacing).square().sum()).sqrt().item(),
                power_error=transfer_error,
                max_velocity_cm_s=max(v.abs().max().item() for v in u))
        return PaperState(old.step+1,new_time,x,u,p,force,new_time,pressure_time=new_time,
                          previous_x=old.x.clone(),previous_dt=self.flow.dt),info


class BalancedAFSEIStepper(AFSEIStepper):
    def __init__(self,mesh,config,*,level=3,spatial_coupling='legacy-balanced',pressure_backend='spectral',
                 fluid_solver='be-projection',cap_integration_rule='centroid'):
        if cap_integration_rule not in ('centroid','gauss3','gauss7','gauss12'):
            raise ValueError('unknown cap integration rule')
        self.cap_integration_rule=cap_integration_rule
        if config.mechanics_profile != 'source' or config.prescribed_endocardial_pressure_mmhg != 0:
            raise ValueError('balanced source branch requires zero prescribed wall traction')
        if spatial_coupling not in ('legacy-balanced','griffith-luo-unified'):
            raise ValueError('unknown MAC spatial coupling')
        if pressure_backend not in ('spectral','multigrid'):
            raise ValueError('MAC pressure backend must be spectral or multigrid')
        if fluid_solver not in ('be-projection','be-stokes-fgmres'):
            raise ValueError('unknown BE fluid solver')
        if fluid_solver=='be-stokes-fgmres' and (spatial_coupling!='griffith-luo-unified' or pressure_backend!='multigrid'):
            raise ValueError('coupled BE Stokes requires standard IB and geometric multigrid')
        self.fluid_solver=fluid_solver
        self.spatial_coupling = spatial_coupling
        super().__init__(mesh,config)
        self.pressure_mode_level = level
        self.pressure_guard_cells = 2
        self._boundary_context = None
        self._pressure_scales = {}; self.pressure_backend = pressure_backend
        self.response_optimizations = True; self.cap_patch_width_cells = 3.
        self.port_geometry = None; self.one_way = True
        self.compact_cap_operators = True
        self.mass_graph_check_every = 16
        self.balanced_tangent_chunk_size = 8192
        self.helmholtz_initial_sweeps = (config.fsi.flow.check_every
            if config.fsi.flow.max_sweeps>=config.fsi.flow.check_every else 0)
        self.reuse_step_tangent = True
        self.valve_resistance_mmhg_s_per_ml = 0.

    @property
    def volume_transfer(self):
        if getattr(self,'gao_cut_boundary',False):
            return 'LV-plus-fixed-tract-discrete-gradient-v1'
        return ('griffith-luo-unified-p1-v1' if self.spatial_coupling=='griffith-luo-unified'
                else 'integrated-p1-gradient-v1')

    def _fsi_stepper(self,step_ticks):
        if step_ticks not in self._steppers:
            dt = step_ticks*self.protocol.dt_ep_s
            self._steppers[step_ticks] = BalancedEndpoint(self,replace(self.config.fsi,time=TimeConfig(dt,dt)))
        return self._steppers[step_ticks]

    def validate_control(self,state):
        if (type(self.reuse_step_tangent) is not bool
                or type(self.balanced_tangent_chunk_size) is not int or self.balanced_tangent_chunk_size<1
                or type(self.helmholtz_initial_sweeps) is not int or self.helmholtz_initial_sweeps<0
                or self.helmholtz_initial_sweeps>self.config.fsi.flow.max_sweeps
                or self.helmholtz_initial_sweeps%self.config.fsi.flow.check_every):
            raise ValueError('invalid balanced execution controls')
        if not isinstance(state,BalancedState):
            raise ValueError('pressure-balanced state required')
        self._check_clock(state)
        if (state.mode not in ('closed','rcr','filling')
                or state.cap_mode not in (None,'closed','rcr','filling')
                or not isfinite(state.filling_pressure_mmhg)
                or not isfinite(state.wall_pressure_mmhg)
                or not isfinite(state.cap_reference_mmhg)
                or not isfinite(state.arterial_pc_mmhg) or not isfinite(state.target_volume_ml)
                or state.target_volume_ml <= 0 or type(state.outlet_open) is not bool
                or type(state.accepted_port_steps) is not int or state.accepted_port_steps < 0
                or state.cap_geometry.shape != state.fsi.x.shape
                or state.cap_geometry.dtype != state.fsi.x.dtype or state.cap_geometry.device != state.fsi.x.device
                or not torch.isfinite(state.cap_geometry).all()
                or state.cap_pressure.ndim != 1 or not state.cap_pressure.numel()
                or state.cap_pressure.dtype != state.fsi.x.dtype or state.cap_pressure.device != state.fsi.x.device
                or not torch.isfinite(state.cap_pressure).all()):
            raise ValueError('invalid pressure-balanced boundary state')
        if state.fsi.step and not torch.equal(state.cap_geometry,state.fsi.previous_x):
            raise ValueError('paired reaction must retain pre-step geometry')

    def force_for_state(self,state):
        e = self._fsi_stepper(round(state.force_load.dt_s/self.protocol.dt_ep_s))
        mode = state.cap_mode or state.mode
        rows = (outlet_rows(e,state.cap_geometry,label=1 if mode=='filling' else 2)
                if mode!='closed' else closed_rows(e,state.cap_geometry))
        if getattr(self,'gao_cut_boundary',False):
            from .gao_tracts import cut_rows
            rows=cut_rows(e,state.cap_geometry)
        force=self.model.force(state.fsi.x,state.fsi.time)+rows.reaction(
            state.cap_pressure+state.cap_reference_mmhg*C)
        if state.wall_pressure_mmhg:
            force=force+MACEndocardialPressure(self.model).force(state.fsi.x,state.wall_pressure_mmhg)
        return force

    @torch.no_grad()
    def initialize_preload(self,path,*,mode='closed',arterial_pc_mmhg=85.):
        metadata,data = read_result(path)
        result,history = reuse_validated_loading(self,path)
        tensor = lambda n: torch.as_tensor(data[n],device=self.mesh.X.device)
        tissue = TissueState(*(tensor(n) for n in ('voltage_v','gpb','land')))
        ep = self.initialize(x=result.x,tissue=tissue)
        strain = StrainHistory(*(self.prolong_ep(v) for v in (history.stretch,history.zs,history.zw)))
        load = replace(ep.force_load,zs_old=strain.zs.clone(),zw_old=strain.zw.clone())
        self.buffer.stage(load)
        e = self._fsi_stepper(self.maximum_ticks)
        rows = outlet_rows(e,result.x) if mode=='rcr' else closed_rows(e,result.x)
        bp = torch.full_like(rows.area,result.pressure_mmhg*C)
        force = self.model.force(result.x,ep.fsi.time)+rows.reaction(bp)
        state = BalancedState(replace(ep,strain=strain,force_load=load,
            fsi=replace(ep.fsi,pressure=tensor('pressure'),force=force)),result.volume_ml,bp,result.x.clone(),
            mode=mode,arterial_pc_mmhg=arterial_pc_mmhg)
        self.validate_control(state)
        return state

    @torch.no_grad()
    def step(self,old,**kwargs):
        self.validate_control(old); self._boundary_context = old
        try:
            ep,info = AFSEIStepper.step(self,old.ep,**kwargs)
            bp,geom = info.pop('_cap_state'); r = info['balanced']
            result = replace(old,ep=ep,cap_pressure=bp,cap_geometry=geom,
                cap_mode=old.mode,
                arterial_pc_mmhg=r['arterial_pc_mmhg'],outlet_open=r['outlet_open'],
                wall_pressure_mmhg=r['wall_pressure_mmhg'],
                cap_reference_mmhg=r['cap_reference_mmhg'],
                accepted_port_steps=old.accepted_port_steps+int(old.mode=='rcr'))
            self.validate_control(result)
            return result,info
        finally: self._boundary_context = None
