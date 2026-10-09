"""Rigid-cavity unsteady Stokes flow coupled implicitly to two idealized ports.

This is a fluid/RCR validation subproblem, not the moving-heart AFSEI driver.
There is no convection, preload, anatomical leaflet or deforming-cell GCL.
For a rigid incompressible cavity Qin=Qout: its two one-way lumped valves
therefore open/close together. This cannot represent isovolumic LV phases.
"""
from dataclasses import dataclass,field
from math import isfinite
import torch
from .operators import CavityOperators
from ..fluid.solvers import SolverOptions
from ..ep.windkessel import WindkesselParameters,flow_driven_trial,MMHG_TO_DYN_CM2 as C


@dataclass(frozen=True)
class CavityFlowConfig:
    dt_s: float=.001
    density: float=1.
    viscosity: float=.04
    atrium_mmhg: float=8.5
    initial_pc_mmhg: float=8.
    mv_resistance: float=.01
    av_resistance: float=.001
    one_way: bool=True
    windkessel: WindkesselParameters=field(default_factory=WindkesselParameters)
    solver: SolverOptions=field(default_factory=lambda:SolverOptions(rtol=1e-11,atol=1e-12,max_iterations=2000,check_every=8))

    def __post_init__(self):
        values=(self.dt_s,self.density,self.viscosity,self.atrium_mmhg,self.initial_pc_mmhg,self.mv_resistance,self.av_resistance)
        if any(not isfinite(v) for v in values) or min(self.dt_s,self.density,self.viscosity,self.mv_resistance,self.av_resistance)<=0:
            raise ValueError('finite pressures and positive dt, density, viscosity and valve resistances required')
        if type(self.one_way)!=bool: raise ValueError('one_way must be a boolean')


@dataclass(frozen=True)
class CavityFluidState:
    step: int
    time_s: float
    velocity: tuple
    pressure: torch.Tensor
    pc_mmhg: torch.Tensor
    q_out_ml_s: torch.Tensor
    port_pressure_cgs: torch.Tensor
    ports_open: bool


class CavityFlowStepper:
    def __init__(self,domain,config=None):
        domain.validate(); self.domain=domain; self.config=config or CavityFlowConfig()
        self.cases={}; self.responses=None

    def operators(self,opened):
        if opened not in self.cases: self.cases[opened]=CavityOperators(self.domain,opened)
        return self.cases[opened]

    def initial_state(self):
        d=self.domain; z=torch.zeros(d.grid.shape,device=d.fluid.device,dtype=torch.float64)
        return CavityFluidState(0,0.,d.grid.zeros(device=d.fluid.device),z,z.new_tensor(self.config.initial_pc_mmhg),z.new_zeros(2),z.new_zeros(2),False)

    @torch.no_grad()
    def _responses(self):
        if self.responses is None:
            cfg=self.config; op=self.operators(True); z=self.domain.grid.zeros(device=self.domain.fluid.device)
            unit=torch.tensor([0.,1.],device=self.domain.fluid.device,dtype=torch.float64)
            av=op.project(z,unit,cfg.dt_s,cfg.density,cfg.solver)
            # A uniform port pressure gives p=constant, u=0 exactly. Use this
            # analytical constant mode instead of subtracting two independently
            # solved responses multiplied by ~10^4 dyn/cm2 reservoir pressure.
            mv=(tuple(-v for v in av[0]),self.domain.fluid.to(torch.float64)-av[1],av[2])
            responses=[mv,av]
            Y=-torch.stack([op.flux(r[0]) for r in responses],1)
            if not torch.allclose(Y,Y.T,atol=1e-10,rtol=1e-9) or torch.linalg.eigvalsh(Y).min()<-1e-10:
                raise RuntimeError('port admittance must be symmetric positive semidefinite')
            self.responses=(responses,Y)
        return self.responses

    @torch.no_grad()
    def step(self,old):
        cfg=self.config; dt=cfg.dt_s; wk=cfg.windkessel; op=self.operators(True)
        star,visc_info=op.viscous(old.velocity,dt,cfg.viscosity/cfg.density,cfg.solver)
        zero=old.pressure.new_zeros(2)
        base,pbase,pinfo=op.project(star,zero,dt,cfg.density,cfg.solver)
        responses,Y=self._responses()
        relaxation=flow_driven_trial(old.pc_mmhg,old.pc_mmhg.new_zeros(()),dt,wk)
        # Values are in mmHg, resistance in mmHg*s/ml, flux in ml/s.
        # Both resistances enter once. Boundary pressure is NOT reapplied to
        # the endocardial solid surface in this fluid subproblem.
        b0=old.pressure.new_tensor([cfg.atrium_mmhg,0.]); b0[1]=relaxation.pc_mmhg
        Z=old.pressure.new_tensor([cfg.mv_resistance,0.]); Z[1]=relaxation.impedance_mmhg_s_per_ml+cfg.av_resistance
        b0*=C; Z*=C
        differential=b0.clone(); differential[1]-=differential[0]; differential[0]=0.
        q=torch.linalg.solve(torch.eye(2,device=Y.device,dtype=Y.dtype)+Y*Z[None],op.flux(base)-Y@differential)
        b=b0+Z*q
        u=tuple(v+(b[1]-b[0])*responses[1][0][c] for c,v in enumerate(base))
        p=pbase+b[0]*self.domain.fluid+(b[1]-b[0])*responses[1][1]
        opened=True
        if cfg.one_way and (q[1]<-1e-10 or q[0]>1e-10):
            opened=False; op=self.operators(False)
            star,visc_info=op.viscous(old.velocity,dt,cfg.viscosity/cfg.density,cfg.solver)
            u,p,pinfo=op.project(star,zero,dt,cfg.density,cfg.solver)
        actual=op.flux(u)
        trial=flow_driven_trial(old.pc_mmhg,actual[1],dt,wk)
        if not opened:
            b=old.pressure.new_tensor([cfg.atrium_mmhg,0.])*C; b[1]=trial.outlet_mmhg*C
        else:
            expected=torch.stack((old.pc_mmhg.new_tensor(cfg.atrium_mmhg)+cfg.mv_resistance*actual[0],
                trial.outlet_mmhg+cfg.av_resistance*actual[1]))*C
            if not torch.allclose(b,expected,rtol=1e-10,atol=1e-9): raise RuntimeError('implicit valve/RCR boundary mismatch')
        qrp=(trial.pc_mmhg-wk.venous_mmhg)/wk.rp_mmhg_s_per_ml
        Eold=.5*cfg.density*op.norm2(old.velocity); Enew=.5*cfg.density*op.norm2(u)
        predictor=sum((v-w).square().mul(m).sum()*op.grid.volume for v,w,m in zip(star,old.velocity,op.potential))*.5*cfg.density/dt
        projection=sum((v-w).square().mul(m).sum()*op.grid.volume for v,w,m in zip(u,star,op.potential))*.5*cfg.density/dt
        viscous=cfg.viscosity*op.grid.volume*sum((v*op.apply(K,v)).sum() for v,K in zip(star,op.viscosity))
        fluid_rate=(Enew-Eold)/dt; fluid_power=-(b*actual).sum()
        delta=trial.pc_mmhg-old.pc_mmhg
        cap_rate=.5*wk.compliance_ml_per_mmhg*(trial.pc_mmhg+old.pc_mmhg)*delta/dt*C
        cap_loss=.5*wk.compliance_ml_per_mmhg*delta.square()/dt*C
        resistive=(cfg.mv_resistance*actual[0].square()+(cfg.av_resistance+wk.rc_mmhg_s_per_ml)*actual[1].square()+wk.rp_mmhg_s_per_ml*qrp.square())*C
        external=(cfg.atrium_mmhg*(-actual[0])-wk.venous_mmhg*qrp)*C
        energy_error=fluid_rate+predictor+projection+viscous+cap_rate+cap_loss+resistive-external
        fluid_error=fluid_rate+predictor+projection+viscous-fluid_power
        divergence=op.divergence(u)
        leak=max((v.masked_select(~a).abs().max().item() if (~a).any() else 0.) for v,a in zip(u,op.active))
        state=CavityFluidState(old.step+1,(old.step+1)*dt,u,p,trial.pc_mmhg,actual,b,opened)
        report=dict(step=state.step,time_s=state.time_s,ports_open=opened,q_in_ml_s=-actual[0].item(),q_out_ml_s=actual[1].item(),
            pc_mmhg=trial.pc_mmhg.item(),mv_pressure_mmhg=b[0].item()/C,av_pressure_mmhg=b[1].item()/C,
            max_divergence_s_inv=divergence.abs().max().item(),mass_error_ml_s=actual.sum().item(),wall_leak_cm_s=leak,
            fluid_energy_error_erg_s=fluid_error.item(),total_energy_error_erg_s=energy_error.item(),external_power_erg_s=external.item(),
            total_energy_relative_error=abs(energy_error.item())/max(abs(external.item()),abs(fluid_power.item()),1.),
            fluid_energy_relative_error=abs(fluid_error.item())/max(abs(fluid_power.item()),abs(fluid_rate.item()),1.),
            rcr_balance_ml_s=(wk.compliance_ml_per_mmhg*delta/dt-actual[1]+qrp).item(),kinetic_energy_erg=Enew.item(),
            viscous_dissipation_erg_s=viscous.item(),predictor_loss_erg_s=predictor.item(),projection_loss_erg_s=projection.item(),
            pressure_iterations=pinfo.iterations,viscosity_iterations=max(i.iterations for i in visc_info))
        if (report['max_divergence_s_inv']>1e-7 or abs(report['mass_error_ml_s'])>1e-8
            or leak!=0. or report['total_energy_relative_error']>1e-7 or report['fluid_energy_relative_error']>1e-7):
            raise RuntimeError(f'fixed-cavity conservation failed: {report}')
        return state,report
