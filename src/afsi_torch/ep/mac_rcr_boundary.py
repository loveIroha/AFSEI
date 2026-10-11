"""Frozen-geometry relative MAC outlet/RCR response with paired reactions.

Local non-port/MV cap averages are closed. The existing idealized AV region
has a single pressure/flow relation, optionally one-way. Geometry, solid
forces and EP are NOT advanced; this is a boundary component validation,
not a replacement for the original nonlinear solid/FSI step.
"""
from dataclasses import dataclass,asdict
from math import isfinite
from time import perf_counter
import torch
from .mac_seal import prepare_local_cap
from .windkessel import WindkesselParameters,flow_driven_trial,MMHG_TO_DYN_CM2 as C
from ..mac.grid import divergence
from ..mac.open_boundary import face_weights
from .mac_transport import closed_volume_gradient


@dataclass(frozen=True)
class MACRCRBoundaryOptions:
    patch_width_cells: float = 3.
    valve_resistance_mmhg_s_per_ml: float = 0.
    one_way: bool = True
    flux_tolerance_ml_s: float = 1e-5
    pressure_tolerance_mmhg: float = 1e-7

    def __post_init__(self):
        if (not isfinite(self.patch_width_cells) or self.patch_width_cells<2
                or not isfinite(self.valve_resistance_mmhg_s_per_ml)
                or self.valve_resistance_mmhg_s_per_ml<0 or type(self.one_way) is not bool
                or any(not isfinite(v) or v<=0 for v in
                       (self.flux_tolerance_ml_s,self.pressure_tolerance_mmhg))):
            raise ValueError('valid cap width, nonnegative valve resistance and positive tolerances required')


@dataclass(frozen=True)
class MACRCRBoundaryResult:
    velocity: tuple
    pressure: torch.Tensor
    nodal_velocity: torch.Tensor
    cap_pressure_cgs: torch.Tensor
    outlet_pressure_cgs: torch.Tensor
    material_reaction: torch.Tensor
    pc_mmhg: torch.Tensor
    report: dict


class MACRCRBoundaryResponse:
    """Condense the unchanged linear BE fluid response into a small port solve.

    G u = B_surface u - C_surface J u and its paired force is -G* p.
    Every mobility column uses the true BE viscosity/pressure response;
    no kinetic approximation, EP/solid solve or full-FSI outer iteration.
    The condensed matrix is reusable ONLY at this exact frozen geometry,
    IB stencil, dt and fluid operator. It is not a moving-geometry cache.
    """
    @torch.no_grad()
    def __init__(self,flow,transfer,stencil,prepared,*,geometry,parameters=None,options=None):
        self.flow,self.transfer,self.stencil,self.prepared=flow,transfer,stencil,prepared
        self.parameters=parameters or WindkesselParameters()
        self.options=options or MACRCRBoundaryOptions()
        self.like=prepared.points; self.grid=flow.grid
        if (geometry.shape!=prepared.cap.X.shape or geometry.dtype!=self.like.dtype
                or geometry.device!=self.like.device or not torch.isfinite(geometry).all()):
            raise ValueError('matching frozen nodal geometry required')
        self.geometry=geometry
        if (not prepared.cap.with_ports or prepared.cap.grid!=self.grid or transfer.grid!=self.grid
                or not torch.any(prepared.cap.labels==2)):
            raise ValueError('matching original MAC grid and an existing resolved idealized AV region required')
        points,areas=prepared.cap.surface(geometry)
        if not torch.allclose(points,prepared.points,rtol=0,atol=1e-12) or not torch.allclose(areas,prepared.area_vectors,rtol=0,atol=1e-12):
            raise ValueError('prepared cap must use the supplied frozen geometry')
        self.closed=prepare_local_cap(prepared,patch_width_cells=self.options.patch_width_cells,regions=(0,1))
        self.count=self.closed.count+1
        self.pressure_scale=100*C
        self.binding=self._binding()
        self._volume_gradient=closed_volume_gradient(self._geometry(),prepared.cap.endo,prepared.cap.rim)
        started=perf_counter(); columns=[]; before=flow.calls
        unit=self.like.new_zeros(self.count)
        try:
            for j in range(self.count):
                unit.zero_(); unit[j]=self.pressure_scale
                density=self.boundary_density(unit)
                velocity=flow.response(density)
                nodal=self.interpolate(velocity)
                columns.append(-self.relative_rates(velocity,nodal))
            self.mobility=torch.stack(columns,1)
            if not torch.isfinite(self.mobility).all(): raise RuntimeError('nonfinite boundary mobility')
            normal=self.mobility[:-1,:-1]/self.closed.area[:,None]
            self.lu,self.pivots=torch.linalg.lu_factor(normal)
            self.port_column=torch.linalg.lu_solve(self.lu,self.pivots,
                (self.mobility[:-1,-1]/self.closed.area)[:,None]).flatten()
            self.admittance=self.mobility[-1,-1]-self.mobility[-1,:-1]@self.port_column
            if not torch.isfinite(self.admittance) or self.admittance<=0:
                raise RuntimeError('outlet has nonpositive effective BE mobility; no boundary state published')
            if self.like.is_cuda: torch.cuda.synchronize(self.like.device)
            self.setup_seconds=perf_counter()-started; self.setup_fluid_responses=flow.calls-before
        finally:
            transfer.reset_warm_start()

    def _geometry(self):
        return self.geometry

    def _binding(self):
        return (self.flow.dt,self.flow.rho,self.flow.mu,self.flow.options,id(self.flow.pressure_solver),
                self.options.patch_width_cells,self.prepared.cap.labels._version,
                self.prepared.points._version,self.prepared.area_vectors._version,self._geometry()._version)

    def interpolate(self,velocity):
        self.transfer.reset_warm_start()
        return self.transfer.interpolate(velocity,self.stencil)[0]

    def relative_rates(self,velocity,nodal_velocity):
        local=self.closed.flow(velocity)-self.closed.material_flow(nodal_velocity)
        outlet=self.prepared.relative_flow(velocity,nodal_velocity)[2]
        return torch.cat((local,outlet.reshape(1)))

    def reaction(self,pressure):
        port=self.like.new_zeros(3); port[2]=pressure[-1]
        return self.closed.reaction(pressure[:-1])+self.prepared.material_reaction(port)

    def boundary_density(self,pressure):
        if pressure.shape!=(self.count,): raise ValueError('matching boundary pressure vector required')
        reaction=self.reaction(pressure)
        self.transfer.reset_warm_start()
        spread,_=self.transfer.spread(reaction,self.stencil)
        lift=self.closed.lift(pressure[:-1])
        port=self.like.new_zeros(3); port[2]=pressure[-1]
        outlet=self.prepared.rows.pressure_density(port)
        return tuple(s-l+p for s,l,p in zip(spread,lift,outlet))

    def pressure_for_rates(self,base_rates,old_pc_mmhg):
        local=torch.linalg.lu_solve(self.lu,self.pivots,(base_rates[:-1]/self.closed.area)[:,None]).flatten()
        free_q=base_rates[-1]-self.mobility[-1,:-1]@local
        circuit=flow_driven_trial(old_pc_mmhg,self.like.new_zeros(()),self.flow.dt,self.parameters)
        alpha=circuit.outlet_cgs/self.pressure_scale
        z=(circuit.impedance_mmhg_s_per_ml+self.options.valve_resistance_mmhg_s_per_ml)*C/self.pressure_scale
        q=(free_q-self.admittance*alpha)/(1+self.admittance*z)
        opened=not self.options.one_way or q.item()>0
        port=alpha+z*q if opened else free_q/self.admittance
        pressure=torch.cat((local-self.port_column*port,port.reshape(1)))*self.pressure_scale
        return pressure,opened,q

    @torch.no_grad()
    def advance(self,departure,external_density,old_pc_mmhg):
        """Pure boundary trial. Commit returned Pc only after caller acceptance.

        external_density is explicitly supplied: no source pressure or Land
        load is generated, replaced or added by this component.
        """
        if self._binding()!=self.binding:
            raise RuntimeError('frozen boundary operator changed; rebuild mobility before using it')
        for value in (departure,external_density):
            self.grid.check_velocity(value)
            if any(v.dtype!=self.like.dtype or v.device!=self.like.device or not torch.isfinite(v).all() for v in value):
                raise ValueError('matching finite float64 MAC fields required')
        old_pc=torch.as_tensor(old_pc_mmhg,device=self.like.device,dtype=self.like.dtype)
        flow_driven_trial(old_pc,self.like.new_zeros(()),self.flow.dt,self.parameters)
        before=self.flow.calls
        try:
            base,_,_=self.flow.advance(departure,external_density)
            rates=self.relative_rates(base,self.interpolate(base))
            pressure,opened,predicted_q=self.pressure_for_rates(rates,old_pc)
            boundary=self.boundary_density(pressure)
            density=tuple(a+b for a,b in zip(external_density,boundary))
            # A final ORIGINAL BE solve checks the true physical residual;
            # do not accept an assembled velocity or change it afterwards.
            velocity,fluid_pressure,flow_info=self.flow.advance(departure,density)
            nodal=self.interpolate(velocity); actual=self.relative_rates(velocity,nodal)
            flux_norm=actual[:-1].norm().item(); measured_q=actual[-1]
            circuit_q=measured_q if opened else self.like.new_zeros(())
            circuit=flow_driven_trial(old_pc,circuit_q,self.flow.dt,self.parameters)
            outlet_law=circuit.outlet_mmhg+self.options.valve_resistance_mmhg_s_per_ml*circuit_q
            law_error=abs((pressure[-1]/C-outlet_law).item()) if opened else 0.
            gap=outlet_law-pressure[-1]/C
            if (flux_norm>self.options.flux_tolerance_ml_s
                    or (not opened and abs(measured_q.item())>self.options.flux_tolerance_ml_s)
                    or (self.options.one_way and opened and measured_q.item() < -self.options.flux_tolerance_ml_s)
                    or (self.options.one_way and not opened and gap.item() < -self.options.pressure_tolerance_mmhg)
                    or law_error>self.options.pressure_tolerance_mmhg):
                raise RuntimeError('true MAC seal/one-way/RCR pressure residual failed; no circuit state published')
            reaction=self.reaction(pressure)
            port_pressure=self.like.new_zeros(3); port_pressure[2]=pressure[-1]
            port_density=self.prepared.rows.pressure_density(port_pressure)
            local_lift=self.closed.lift(pressure[:-1])
            # S(C^T p) is already inside the effective response density.
            # Split the actual surface fluid load -B^T p and C^T p work
            # here; never count the material reaction twice.
            fluid_power=sum((u*(p-l)*face_weights(self.grid,c,self.like)).sum()
                for c,(u,p,l) in enumerate(zip(velocity,port_density,local_lift)))
            solid_power=(reaction*nodal).sum(); expected=-(pressure*actual).sum()
            work_error=abs((fluid_power+solid_power-expected).item())
            work_relative=work_error/max(abs(fluid_power.item())+abs(solid_power.item())+abs(expected.item()),1.)
            if work_relative>1e-7: raise RuntimeError('relative outlet pressure work mismatch')
            p=self.parameters; dp=circuit.pc_mmhg-old_pc
            rcr_balance=p.compliance_ml_per_mmhg*dp/self.flow.dt-circuit_q+(circuit.pc_mmhg-p.venous_mmhg)/p.rp_mmhg_s_per_ml
            storage=.5*p.compliance_ml_per_mmhg*((circuit.pc_mmhg-p.venous_mmhg)**2-(old_pc-p.venous_mmhg)**2)*C/self.flow.dt
            capacitor_loss=.5*p.compliance_ml_per_mmhg*dp**2*C/self.flow.dt
            resistance_loss=((circuit.pc_mmhg-p.venous_mmhg)**2/p.rp_mmhg_s_per_ml
                +(p.rc_mmhg_s_per_ml+self.options.valve_resistance_mmhg_s_per_ml)*circuit_q**2)*C
            rcr_power=(outlet_law-p.venous_mmhg)*circuit_q*C
            energy_error=(rcr_power-storage-capacitor_loss-resistance_loss).abs()
            report=dict(outlet_open=opened,measured_relative_q_ml_s=measured_q.item(),
                rcr_input_q_ml_s=circuit_q.item(),predicted_open_q_ml_s=predicted_q.item(),
                pc_old_mmhg=old_pc.item(),pc_mmhg=circuit.pc_mmhg.item(),
                boundary_pressure_mmhg=(pressure[-1]/C).item(),rcr_outlet_mmhg=circuit.outlet_mmhg.item(),
                valve_gap_mmhg=gap.item(),open_pressure_law_error_mmhg=law_error,
                closed_region_flux_norm_ml_s=flux_norm,rcr_balance_ml_s=rcr_balance.item(),
                fluid_boundary_power_erg_s=fluid_power.item(),material_boundary_power_erg_s=solid_power.item(),
                relative_work_error_erg_s=work_error,relative_work_relative_error=work_relative,
                rcr_storage_rate_erg_s=storage.item(),rcr_capacitor_BE_loss_erg_s=capacitor_loss.item(),
                rcr_resistance_dissipation_erg_s=resistance_loss.item(),rcr_external_power_erg_s=rcr_power.item(),
                rcr_energy_error_erg_s=energy_error.item(),
                rcr_energy_relative_error=(energy_error/(storage.abs()+capacitor_loss.abs()+resistance_loss.abs()+rcr_power.abs()).clamp_min(1.)).item(),
                material_current_volume_rate_ml_s=(self._volume_gradient*nodal).sum().item(),
                port_only_mass_defect_ml_s=((self._volume_gradient*nodal).sum()+measured_q).item(),
                mac_max_divergence_s_inv=divergence(velocity,self.grid.spacing).abs().max().item(),
                fluid_solves=self.flow.calls-before,closed_patches=self.closed.count,
                flow=flow_info,options=asdict(self.options),parameters=asdict(self.parameters),
                frozen_geometry=True,material_velocity_kind='original IB projection Ju; nodal geometry is not advanced',
                boundary_reaction_returned=True,actual_relative_port_flow_computed=True,
                flow_driven_rcr_in_subproblem=True,anatomical_valves_present=False,
                ep_advanced=False,myocardium_motion_advanced=False,source_force_changed=False,
                local_wall_sealing_established=False,windkessel_ports_connected_to_afsei_driver=False,
                three_field_integration_implemented=False)
            return MACRCRBoundaryResult(velocity,fluid_pressure,nodal,pressure[:-1].clone(),pressure[-1].clone(),
                reaction,circuit.pc_mmhg.clone(),report)
        finally:
            self.transfer.reset_warm_start()
