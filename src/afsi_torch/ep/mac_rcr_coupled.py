"""Measured relative AV flow and RCR inside the original MAC FSI Newton step.

Validation of the outlet interval, not a complete cardiac phase controller.
MV/non-port cap averages are closed; AV has one uniform pressure multiplier.
The source current-endocardial follower force is unchanged. A separate wall
pressure unknown enforces finite-step volume/port balance, not local sealing.
Geometry/IB operators are frozen at the accepted old state, as upstream.
"""
from dataclasses import dataclass, replace, asdict
from math import isfinite
import torch
from .driver import AFSEIStepper
from .mac_isovolumic import IsovolumicState, IsovolumicAFSEIStepper, IsovolumicEndpointStepper
from .mac_cap_coupled import CapIsovolumicProblem, closed_rows
from .mac_ports import MACBasalPorts
from .mac_seal import LocalCapRows, prepare_local_cap
from .mac_transport import closed_volume_gradient
from .mac_preload import MACEndocardialPressure
from .windkessel import flow_driven_trial, MMHG_TO_DYN_CM2 as C
from ..config import TimeConfig
from ..nonlinear import normalized_linear_action
from ..mac.open_boundary import face_weights

_UNSET = object()


@dataclass(frozen=True)
class RCRState(IsovolumicState):
    outlet_open: bool = False
    accepted_port_steps: int = 0
    boundary_history: str = 'closed'


def outlet_rows(endpoint, x, *, label=2):
    """Retain local closed patches, plus ONE row for the entire idealized AV."""
    if not hasattr(endpoint, '_outlet_cap'):
        endpoint._outlet_cap = MACBasalPorts(endpoint.model.mesh.X,
            endpoint.model.endo_faces, endpoint.grid, ports=endpoint.owner.port_geometry)
    p = endpoint._outlet_cap.prepare(x)
    closed = prepare_local_cap(p, patch_width_cells=endpoint.owner.cap_patch_width_cells,
                              regions=tuple(k for k in (0,1,2) if k != label))
    av = p.cap.labels == label
    area = p.area_vectors[av].norm(dim=-1).sum()
    patches = torch.empty_like(p.cap.labels)
    patches[closed.selected] = closed.patches
    patches[av] = closed.count
    components = []
    diagonal = area.new_zeros(())
    for local, port in zip(closed.components, p.rows.components):
        r, index, coefficient = local
        labels, pi, pc = port
        keep = labels == label
        coefficient_av = pc[keep]
        components.append((torch.cat((r, torch.full_like(pi[keep], closed.count))),
            torch.cat((index, pi[keep])), torch.cat((coefficient, coefficient_av))))
        diagonal += coefficient_av.square().sum()/endpoint.grid.volume
    return LocalCapRows(p, torch.arange(len(p.points), device=x.device), patches,
        torch.cat((closed.area, area.reshape(1))), tuple(components),
        torch.cat((closed.diagonal, diagonal.reshape(1))))


class RCRCoupledProblem(CapIsovolumicProblem):
    def prepare_rows(self):
        return outlet_rows(self.driver, self.state.x)

    def __init__(self, *args):
        super().__init__(*args)
        self.old_volume = self.load.volume(self.state.x)
        self.impedance = flow_driven_trial(self.control.arterial_pc_mmhg, 0.,
            self.driver.flow.dt, self.control.windkessel).impedance_mmhg_s_per_ml
        self.impedance += self.driver.owner.valve_resistance_mmhg_s_per_ml

    def circuit(self, q):
        return flow_driven_trial(self.control.arterial_pc_mmhg, q,
                                 self.driver.flow.dt, self.control.windkessel)

    def port_equation(self, q, pressure):
        trial = self.circuit(q)
        gap = (trial.outlet_mmhg+
               self.driver.owner.valve_resistance_mmhg_s_per_ml*q)*C-pressure
        flow_row = self.driver.flow.dt*q/self.rows.area[-1]
        # The kinetic cap pressure scale is very large at dt=0.1 ms.
        # Using it here hides a physical mmHg error under a small Newton
        # norm. Keep the open pressure-law row in mmHg; the border inverse
        # supplies its corresponding units. Positive row scalings preserve
        # the same one-way complementarity condition.
        gap_row = gap/C
        closed = self.driver.owner.one_way and flow_row.item() <= gap_row.item()
        # min(Q, Pao-Pboundary)=0: Q >=0, gap >=0, Q*gap=0.
        return (flow_row if closed else gap_row), not closed, trial, gap

    def residual(self, z):
        state_version = None if torch.is_inference(self.state.x) else self.state.x._version
        if (self.driver.owner.response_optimizations and self.last_z is not None
                and state_version is not None and state_version == self._last_state_version
                and torch.equal(z, self.last_z)):
            self.cache_hits += 1
            return self.last_residual.clone()
        self.validate(z)
        d, rows = self.driver, self.rows
        y, x, p = self.unpack(z); bp = self.cap_pressure(z)
        base = d.owner.config.prescribed_endocardial_pressure_mmhg
        force = d.solid.force(x, self.time)+self.load.force(x, p-base)+rows.reaction(bp)
        d.transfer.reset_warm_start()
        density, spread = d.transfer.spread(force, self.stencil)
        density = tuple(a-b for a, b in zip(density, rows.lift(bp)))
        velocity, pressure, flow = d.flow.advance(self.departure, density, self.pressure_initial)
        nodal, interpolation = d.transfer.interpolate(velocity, self.stencil)
        volume = self.load.volume(x)
        relative = rows.flow(velocity)-rows.material_flow(y/d.flow.dt)
        q = relative[-1]
        port, opened, circuit, gap = self.port_equation(q, bp[-1])
        self.last_mass_error = volume-self.old_volume+d.flow.dt*q
        residual = torch.cat(((y-d.flow.dt*nodal).flatten(),
            (self.last_mass_error/self.volume_scale).reshape(1),
            d.flow.dt*relative[:-1]/rows.area[:-1], port.reshape(1)))
        if not torch.isfinite(residual).all():
            raise FloatingPointError('nonfinite coupled RCR residual')
        self.pressure_initial = pressure
        self.last_z, self.last_residual = z.clone(), residual.clone()
        self._last_state_version = state_version
        self.last_cap_pressure = bp.clone()
        self.last_data = x, p, volume, force, velocity, pressure, flow, nodal, density, spread, interpolation
        self.last_q, self.last_open, self.last_circuit, self.last_gap = q, opened, circuit, gap
        self.evaluations += 1
        return residual

    def linearization(self, z):
        self.accepted(z)
        d, rows = self.driver, self.rows
        _, x, p = self.unpack(z)
        if d.assembler is None: d.assembler = d.model.tangent_factory()
        tangent = d.assembler.assemble(x, self.time)
        self.tangent_assemblies += 1
        self.c = closed_volume_gradient(x, self.load.faces, self.load.rim).flatten()/self.volume_scale
        up, wp = self.sensitivity(self.load.force(x, 1.)*self.pressure_scale)
        self.b = (-d.flow.dt*wp).flatten()
        self.hp = d.flow.dt*(rows.flow(up)-rows.material_flow(wp))/rows.area
        self.linear_open = self.last_open
        offset = p-d.owner.config.prescribed_endocardial_pressure_mmhg
        def action(v):
            self.jacobian_actions += 1
            dy = v[:self.mechanical_size].reshape_as(x)
            df = torch.sparse.mm(tangent, dy.reshape(-1, 1)).reshape_as(x)
            df += self.load.action(x, offset, dy)
            df += self.load.force(x, 1.)*(self.pressure_scale*v[self.mechanical_size])
            du, dw = self.sensitivity(df, self.cap_scale*v[self.mechanical_size+1:])
            rates = rows.flow(du)-rows.material_flow(dy/d.flow.dt)
            mass = self.c@dy.flatten()+d.flow.dt*rates[-1]/self.volume_scale
            last = ((self.impedance*C*rates[-1]-self.cap_scale[-1]*v[-1])/C
                    if self.linear_open else d.flow.dt*rates[-1]/rows.area[-1])
            return torch.cat(((dy-d.flow.dt*dw).flatten(),
                mass.reshape(1), d.flow.dt*rates[:-1]/rows.area[:-1], last.reshape(1)))
        return normalized_linear_action(action)

    def preconditioner(self, z):
        """Use the existing cached kinetic border; never rebuild full BE mobility.

        The last row changes algebraically between open/closed laws. Global
        volume and actual outlet rate are independent rows in the true solve.
        This preconditioner uses the same cap-column remapping as phase 18.
        """
        c, b, rows = self.c, self.b, self.rows
        n = rows.count
        h = rows.area.new_empty((n+1, n+1))
        factor = rows.area[-1]/self.volume_scale
        h[0, 0] = -c@b+factor*self.hp[-1]
        h[0, 1:] = -c@self.cap_b+factor*self.cap_h[-1]
        h[1:, 0] = self.hp
        h[1:, 1:] = self.cap_h
        law_factor = self.impedance*rows.area[-1]/self.driver.flow.dt
        if self.linear_open:
            h[-1] *= law_factor
            h[-1, -1] -= self.cap_scale[-1]/C
        lu, pivots = torch.linalg.lu_factor(h)
        def apply(v):
            self.preconditioner_applications += 1
            y = v[:self.mechanical_size]
            material = rows.material_flow(y.reshape_as(self.state.x))/rows.area
            mass = v[self.mechanical_size]-c@y+material[-1]*factor
            lower = torch.cat((mass.reshape(1), v[self.mechanical_size+1:]+material))
            if self.linear_open:
                lower[-1] = v[-1]+law_factor*material[-1]
            q = torch.linalg.lu_solve(lu, pivots, lower[:, None]).flatten()
            return torch.cat((y-b*q[0]-self.cap_b@q[1:], q))
        return apply

    def report(self):
        d, rows = self.driver, self.rows
        x, p, volume, _, u, _, _, nodal, _, _, _ = self.last_data
        geometric = (x-self.state.x)/d.flow.dt
        rates = rows.flow(u)-rows.material_flow(geometric)
        interpolated = rows.flow(u)-rows.material_flow(nodal)
        bp, q = self.last_cap_pressure, rates[-1]
        # Local closure and one-way law checked independently of Newton scaling.
        gap = self.last_gap/C
        law_error = abs(gap.item()) if self.last_open else 0.
        if (law_error > 1e-7 or (d.owner.one_way and q.item() < -1e-5)
                or (not self.last_open and (abs(q.item()) > 1e-5 or gap.item() < -1e-7))
                or abs(self.last_mass_error.item()/d.flow.dt) > 1e-5
                or abs((q-interpolated[-1]).item()) > 1e-5):
            raise RuntimeError('true RCR pressure/one-way/volume-port balance failed: '
                f'Q={q.item():.9g} ml/s, gap={gap.item():.9g} mmHg, '
                f'law_error={law_error:.9g} mmHg, mass={self.last_mass_error.item()/d.flow.dt:.9g} ml/s, '
                f'port_motion_error={(q-interpolated[-1]).item():.9g} ml/s')
        circuit = self.last_circuit
        dp = circuit.pc_mmhg-self.control.arterial_pc_mmhg
        params = self.control.windkessel
        balance = (params.compliance_ml_per_mmhg*dp/d.flow.dt-q+
            (circuit.pc_mmhg-params.venous_mmhg)/params.rp_mmhg_s_per_ml)
        storage = .5*params.compliance_ml_per_mmhg*((circuit.pc_mmhg-params.venous_mmhg)**2-
            (self.control.arterial_pc_mmhg-params.venous_mmhg)**2)*C/d.flow.dt
        capacitor_loss = .5*params.compliance_ml_per_mmhg*dp**2*C/d.flow.dt
        resistance_loss = ((circuit.pc_mmhg-params.venous_mmhg)**2/params.rp_mmhg_s_per_ml+
            (params.rc_mmhg_s_per_ml+d.owner.valve_resistance_mmhg_s_per_ml)*q**2)*C
        arterial_power = (circuit.outlet_mmhg+d.owner.valve_resistance_mmhg_s_per_ml*q-
                          params.venous_mmhg)*q*C
        energy_error = (arterial_power-storage-capacitor_loss-resistance_loss).abs()
        energy_relative = energy_error/(arterial_power.abs()+storage.abs()+
            capacitor_loss.abs()+resistance_loss.abs()).clamp_min(1.)
        lift = rows.lift(bp); reaction = rows.reaction(bp)
        fluid_power = -sum((a*b*face_weights(d.grid, c, x)).sum()
                          for c, (a, b) in enumerate(zip(u, lift)))
        solid_power = (reaction*nodal).sum()
        expected = -(bp*interpolated).sum()
        work_error = abs((fluid_power+solid_power-expected).item())
        work_relative = work_error/max(abs(fluid_power.item())+abs(solid_power.item())+abs(expected.item()), 1.)
        if work_relative > 1e-7:
            raise RuntimeError('coupled boundary work-adjoint check failed')
        report = dict(patches=rows.count-1, outlet_pressure_unknowns=1,
            complete_cap_closed_in_patch_average=False,
            constraint_residual_norm_ml_s=rates[:-1].norm().item(),
            interpolated_velocity_constraint_norm_ml_s=interpolated[:-1].norm().item(),
            cap_fluid_power_erg_s=fluid_power.item(), cap_solid_power_erg_s=solid_power.item(),
            cap_total_power_erg_s=(fluid_power+solid_power).item(),
            cap_power_relative_error=work_relative, boundary_work_error_erg_s=work_error,
            coupled_solid_advanced=True, frozen_cap_geometry=True,
            anatomical_valves_present=False, pointwise_impermeability_established=False,
            preconditioner_rebuilt=self.preconditioner_rebuilt,
            preconditioner_setup_seconds=self.setup_seconds,
            preconditioner_setup_pressure_calls=self.setup_pressure_calls)
        closed_mask = rows.patches < rows.count-1
        closed = replace(rows, selected=rows.selected[closed_mask], patches=rows.patches[closed_mask],
            area=rows.area[:-1], diagonal=rows.diagonal[:-1],
            components=tuple((r[r<rows.count-1], i[r<rows.count-1], a[r<rows.count-1])
                             for r,i,a in rows.components))
        report['closed_cap_slip'] = closed.diagnostics(u, geometric)
        report['rcr'] = dict(outlet_open=self.last_open, measured_relative_q_ml_s=q.item(),
            interpolated_relative_q_ml_s=interpolated[-1].item(),
            volume_old_ml=self.old_volume.item(), volume_ml=volume.item(),
            volume_rate_ml_s=((volume-self.old_volume)/d.flow.dt).item(),
            mass_error_ml_s=(self.last_mass_error/d.flow.dt).item(),
            current_volume_rate_ml_s=(closed_volume_gradient(x, self.load.faces, self.load.rim)*nodal).sum().item(),
            cavity_pressure_mmhg=p.item(), boundary_pressure_mmhg=(bp[-1]/C).item(),
            arterial_pc_old_mmhg=self.control.arterial_pc_mmhg,
            arterial_pc_mmhg=circuit.pc_mmhg.item(), rcr_outlet_mmhg=circuit.outlet_mmhg.item(),
            valve_gap_mmhg=gap.item(), open_pressure_law_error_mmhg=law_error,
            rcr_balance_ml_s=balance.item(), finite_step_volume_balance_enforced=True,
            rcr_energy_error_erg_s=energy_error.item(),rcr_energy_relative_error=energy_relative.item(),
            actual_port_flow_used=True, fixed_volume_target=False,
            source_wall_pressure_is_validation_control=True,
            physical_cavity_pressure_identification_pending=True,
            local_wall_sealing_established=False, full_cardiac_cycle_implemented=False)
        return report


class RCREndpointStepper(IsovolumicEndpointStepper):
    def make_problem(self, state, stencil, departure, control):
        return RCRCoupledProblem(self, state, stencil, departure, control)

    def volume_error(self, problem, volume, control):
        return problem.last_mass_error.item()

    def physical_step(self, state, new_time, diagnostics):
        accepted, info = super().physical_step(state, new_time, diagnostics)
        info.pop('isovolumic')
        info['rcr'] = info['cap'].pop('rcr')
        info['nonlinear']['solver'] = 'original-mac-augmented-newton-rcr'
        return accepted, info


class RCRAFSEIStepper(IsovolumicAFSEIStepper):
    @classmethod
    def from_driver(cls, driver):
        result = super().from_driver(driver)
        result.closed_cap = True
        result.port_geometry = None
        result.one_way = True
        result.valve_resistance_mmhg_s_per_ml = 0.
        result.cap_patch_width_cells = driver.cap_patch_width_cells if hasattr(driver, 'cap_patch_width_cells') else 3.
        return result

    def __init__(self, mesh, config):
        super().__init__(mesh, config)
        self.closed_cap = True
        self.port_geometry = None
        self.one_way = True
        self.valve_resistance_mmhg_s_per_ml = 0.

    def _fsi_stepper(self, step_ticks):
        if step_ticks not in self._steppers:
            dt = step_ticks*self.protocol.dt_ep_s
            cfg = replace(self.config.fsi, time=TimeConfig(dt, dt))
            self._steppers[step_ticks] = RCREndpointStepper(self, cfg)
        return self._steppers[step_ticks]

    def connect(self, state):
        self.buffer.stage(state.force_load)
        if isinstance(state, RCRState):
            self.validate_control(state)
            return state
        if not isinstance(state, IsovolumicState):
            state = IsovolumicState(state, MACEndocardialPressure(self.model).volume(state.fsi.x).item(),
                                   self.config.prescribed_endocardial_pressure_mmhg)
        result = RCRState(**state.__dict__)
        self.validate_control(result)
        return result

    def execution_settings(self):
        return dict(super().execution_settings(), one_way=self.one_way,
            valve_resistance_mmhg_s_per_ml=self.valve_resistance_mmhg_s_per_ml,
            port_geometry=asdict(self.port_geometry) if self.port_geometry is not None else None)

    def set_execution(self, *, one_way=None, valve_resistance_mmhg_s_per_ml=None,
                      port_geometry=_UNSET, **kwargs):
        from ..circulation.geometry import PortGeometry
        one_way = self.one_way if one_way is None else one_way
        valve_resistance_mmhg_s_per_ml = (self.valve_resistance_mmhg_s_per_ml
            if valve_resistance_mmhg_s_per_ml is None else valve_resistance_mmhg_s_per_ml)
        if type(one_way) is not bool or not isfinite(valve_resistance_mmhg_s_per_ml) or valve_resistance_mmhg_s_per_ml < 0:
            raise ValueError('invalid one-way/resistance setting')
        if kwargs.get('closed_cap') is False:
            raise ValueError('RCR requires the coupled boundary rows')
        self.one_way, self.valve_resistance_mmhg_s_per_ml = one_way, valve_resistance_mmhg_s_per_ml
        if port_geometry is not _UNSET:
            self.port_geometry = PortGeometry(**port_geometry) if port_geometry is not None else None
        super().set_execution(**kwargs)

    def validate_control(self, state):
        super().validate_control(state)
        if (not isinstance(state, RCRState) or state.boundary_history not in ('closed', 'rcr')
                or type(state.outlet_open) is not bool or type(state.accepted_port_steps) is not int
                or state.accepted_port_steps < 0 or type(self.one_way) is not bool
                or not isfinite(self.valve_resistance_mmhg_s_per_ml)
                or self.valve_resistance_mmhg_s_per_ml < 0 or not self.closed_cap):
            raise ValueError('invalid coupled RCR control/history')

    def force_for_state(self, state):
        force = self.model.force(state.fsi.x, state.fsi.time)+MACEndocardialPressure(self.model).force(
            state.fsi.x, state.cavity_pressure_mmhg-self.config.prescribed_endocardial_pressure_mmhg)
        if state.cap_pressure is not None:
            ticks = round(state.force_load.dt_s/self.protocol.dt_ep_s)
            endpoint = self._fsi_stepper(ticks)
            rows = (outlet_rows(endpoint, state.cap_geometry) if state.boundary_history == 'rcr'
                    else closed_rows(endpoint, state.cap_geometry))
            if rows.count != len(state.cap_pressure):
                raise ValueError('RCR boundary force-history partition mismatch')
            force += rows.reaction(state.cap_pressure)
        return force

    @torch.no_grad()
    def step(self, old, **kwargs):
        self.validate_control(old)
        self._iso_context = old
        try:
            accepted, info = AFSEIStepper.step(self, old.ep, **kwargs)
            rcr = info['rcr']
            pressure, geometry = info.pop('_cap_state')
            result = replace(old, ep=accepted, target_volume_ml=rcr['volume_ml'],
                cavity_pressure_mmhg=rcr['cavity_pressure_mmhg'],
                arterial_pc_mmhg=rcr['arterial_pc_mmhg'], cap_pressure=pressure,
                cap_geometry=geometry, outlet_open=rcr['outlet_open'], boundary_history='rcr',
                accepted_port_steps=old.accepted_port_steps+1,
                accepted_closed_steps=old.accepted_closed_steps+int(not rcr['outlet_open']))
            self.validate_control(result)
            # Outlet closure is reported, not yet wired into a four-phase controller.
            rcr['outlet_closed_event'] = old.outlet_open and not result.outlet_open
            return result, info
        except Exception:
            self._reset_scratch(old.ep)
            raise
        finally:
            self._iso_context = None
