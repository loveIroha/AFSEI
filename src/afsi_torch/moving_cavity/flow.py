"""Conservative moving-cavity flow driven by supplied solid coordinates.

Prescribed/one-way wall motion. The adjoint interface reaction is computed,
but is NOT fed back to the solid solver by this phase's bridge. Incompatible
closed-cavity motion is rejected; neither flux nor geometry is corrected.
"""
from dataclasses import dataclass,field
from math import isfinite
import torch
from .geometry import swept_geometry
from .operators import ALEOperators
from ..fluid.solvers import SolverOptions
from ..ep.windkessel import WindkesselParameters,flow_driven_trial,MMHG_TO_DYN_CM2 as C


@dataclass(frozen=True)
class ALEConfig:
    dt_s: float=.0001
    clock_dt_s: float=.00005
    start_time_s: float=0.
    density: float=1.
    viscosity: float=.04
    atrium_mmhg: float=8.5
    initial_pc_mmhg: float=8.
    mv_resistance: float=.01
    av_resistance: float=.001
    convection: bool=True
    cfl_limit: float=.5
    windkessel: WindkesselParameters=field(default_factory=WindkesselParameters)
    solver: SolverOptions=field(default_factory=lambda:SolverOptions(rtol=1e-10,atol=1e-12,max_iterations=8000,recompute_every=2000,check_every=8))

    def __post_init__(self):
        v=(self.dt_s,self.clock_dt_s,self.start_time_s,self.density,self.viscosity,self.atrium_mmhg,self.initial_pc_mmhg,self.mv_resistance,self.av_resistance,self.cfl_limit)
        if any(not isfinite(a) for a in v) or min(self.dt_s,self.clock_dt_s,self.density,self.viscosity,self.mv_resistance,self.av_resistance,self.cfl_limit)<=0 or self.start_time_s<0 or type(self.convection)!=bool:
            raise ValueError('invalid ALE settings')
        for t in (self.dt_s,self.start_time_s):
            if abs(t/self.clock_dt_s-round(t/self.clock_dt_s))>1e-7: raise ValueError('ALE time must align with its integer clock')


@dataclass(frozen=True)
class ALEState:
    step: int
    tick: int
    solid_x: torch.Tensor
    velocity: torch.Tensor
    pressure: torch.Tensor
    pc_mmhg: torch.Tensor
    q_out_ml_s: torch.Tensor
    port_pressure_cgs: torch.Tensor
    opened: tuple
    has_ejected: bool=False
    previous_solid_x: torch.Tensor | None=None
    last_dt_s: float=.0001


class ALEStepper:
    def __init__(self,mesh,config=None): self.mesh=mesh; self.config=config or ALEConfig()

    def initial_state(self,solid_x=None):
        x=self.mesh.reference_solid if solid_x is None else solid_x
        swept_geometry(self.mesh,x,x,self.config.dt_s)
        return ALEState(0,round(self.config.start_time_s/self.config.clock_dt_s),x.clone(),x.new_zeros((len(self.mesh.faces),3)),
            x.new_zeros(len(self.mesh.cells)),x.new_tensor(self.config.initial_pc_mmhg),x.new_zeros(2),x.new_zeros(2),(False,False),False,x.clone(),self.config.dt_s)

    def closed_pressures(self,op,reaction):
        area=op.geometry.areas; norm=area.norm(dim=1); unit=area/norm[:,None]
        return torch.stack([-((reaction*unit).sum(1)[self.mesh.tags==label]).sum()/norm[self.mesh.tags==label].sum() for label in (1,2)])

    @torch.no_grad()
    def step(self,old,new_solid_x,dt_s=None):
        c=self.config; dt=c.dt_s if dt_s is None else dt_s
        ticks=round(dt/c.clock_dt_s)
        if not isfinite(dt) or dt<=0 or abs(ticks*c.clock_dt_s-dt)>1e-12: raise ValueError('trial dt must align with ALE clock')
        g=swept_geometry(self.mesh,old.solid_x,new_solid_x,dt)
        relative=(old.velocity[self.mesh.cell_faces]-g.grid_velocity[self.mesh.cell_faces]).mean(1).norm(dim=1)
        areas=g.areas.norm(dim=1)[self.mesh.cell_faces]
        height=3*g.volumes/areas.max(1).values
        cfl=(dt*relative/height).max().item()
        if c.convection and cfl>c.cfl_limit: raise ValueError(f'explicit ALE convection CFL {cfl:g} exceeds {c.cfl_limit:g}; reduce dt/remesh')
        wk=c.windkessel; relax=flow_driven_trial(old.pc_mmhg,old.pc_mmhg.new_zeros(()),dt,wk)
        b0=old.velocity.new_tensor([c.atrium_mmhg,0.]); b0[1]=relax.pc_mmhg; b0*=C
        Z=old.velocity.new_tensor([c.mv_resistance,relax.impedance_mmhg_s_per_ml+c.av_resistance])*C
        reasons=[]; selected=None
        # Enumerate the four independent lumped-valve active sets. Open-flow
        # direction and closed-side traction inequalities must BOTH hold.
        for opened in ((True,True),(False,True),(True,False),(False,False)):
            if not any(opened) and abs((g.volumes-g.old_volumes).sum().item()/dt)>1e-8:
                reasons.append('both closed incompatible with imposed volume change'); continue
            op=ALEOperators(self.mesh,g,opened)
            star,rhs,vinfo=op.viscous(old.velocity,dt,c.density,c.viscosity,c.solver,c.convection)
            zero=old.velocity.new_zeros(2)
            base,pbase,pinfo=op.project(star,zero,dt,c.density,c.solver)
            qbase=op.flux(base)
            if all(opened):
                unit=zero.clone(); unit[1]=1.
                du,dp,_=op.project(torch.zeros_like(star),unit,dt,c.density,c.solver)
                qa=op.flux(du,relative=False); Y=torch.stack((-qa,qa),1)*(-1)
                if not torch.allclose(Y,Y.T,rtol=1e-8,atol=1e-9): raise RuntimeError('ALE port admittance lost reciprocity')
                diff=zero.clone(); diff[1]=b0[1]-b0[0]
                q=torch.linalg.solve(torch.eye(2,device=Y.device,dtype=Y.dtype)+Y*Z[None],qbase-Y@diff)
                b=b0+Z*q; u=base+(b[1]-b[0])*du; p=pbase+b[0]+(b[1]-b[0])*dp
            elif any(opened):
                q=qbase; b=b0+Z*q; j=0 if opened[0] else 1
                u=base; p=pbase+b[j]
            else:
                q=qbase; b=b0.clone(); u=base; p=pbase
            actual=op.flux(u)
            if (opened[0] and actual[0]>1e-8) or (opened[1] and actual[1]<-1e-8):
                reasons.append(f'{opened}: reverse open-valve flow'); continue
            reaction=op.reaction(u,p,star,rhs,dt,c.density,c.viscosity,b)
            effective=self.closed_pressures(op,reaction)
            if not any(opened):
                lower=b0[0]-effective[0]; upper=b0[1]-effective[1]
                if lower>upper+1e-6: reasons.append('closed traction bounds have no common pressure'); continue
                gauge=(old.pressure.mean()).clamp(lower,upper)
                p=p+gauge; reaction=op.reaction(u,p,star,rhs,dt,c.density,c.viscosity,b)
                effective=self.closed_pressures(op,reaction)
            if (not opened[0] and effective[0]<b0[0]-1e-6) or (not opened[1] and effective[1]>b0[1]+1e-6):
                reasons.append(f'{opened}: closed valve would reopen'); continue
            selected=(op,star,rhs,u,p,b,actual,reaction,effective,pinfo,vinfo); break
        if selected is None: raise ValueError('no admissible valve state for prescribed wall motion: '+'; '.join(reasons))
        op,star,rhs,u,p,b,q,reaction,effective,pinfo,vinfo=selected
        trial=flow_driven_trial(old.pc_mmhg,q[1],dt,wk)
        expected=torch.stack((old.pc_mmhg.new_tensor(c.atrium_mmhg)+c.mv_resistance*q[0],trial.outlet_mmhg+c.av_resistance*q[1]))*C
        for j,on in enumerate(op.opened):
            if on and not torch.isclose(b[j],expected[j],rtol=1e-9,atol=1e-7): raise RuntimeError('ALE implicit RCR pressure mismatch')
        boundary=self.mesh.tags>=0
        volume_rate=(g.volumes-g.old_volumes).sum()/dt
        residual=op.divergence(u)
        mass_error=volume_rate+q.sum()
        wall_relative=((u-g.grid_velocity)*g.areas).sum(1)[op.fixed]
        free_residual=reaction[~op.fixed].abs().max().item()
        force=self.mesh.transfer_reaction(-reaction.masked_fill(~op.fixed[:,None],0.))
        wall_power=-(reaction[op.fixed]*g.grid_velocity[op.fixed]).sum()
        solid_power=(force*(new_solid_x-old.solid_x)/dt).sum()
        scale=max(q.abs().max().item(),abs(volume_rate.item()),1.)
        if abs(mass_error.item())>1e-7*scale or residual.abs().max()>1e-7*scale or wall_relative.abs().max()>1e-12*scale:
            raise RuntimeError('ALE accepted-step mass/impermeability check failed')
        if not torch.isclose(wall_power,solid_power,rtol=1e-10,atol=1e-8): raise RuntimeError('interface transfer violates virtual work')
        closed_event=old.has_ejected and old.opened[1] and not op.opened[1]
        new=ALEState(old.step+1,old.tick+ticks,new_solid_x.clone(),u,p,trial.pc_mmhg,q,expected,op.opened,
            bool(old.has_ejected or (not op.opened[0] and op.opened[1] and q[1]>1e-8)),old.solid_x.clone(),dt)
        report=dict(step=new.step,time_s=new.tick*c.clock_dt_s,dt_s=dt,mv_open=op.opened[0],av_open=op.opened[1],
            accepted_outlet_closure=bool(closed_event),q_in_ml_s=-q[0].item(),q_out_ml_s=q[1].item(),
            ejection_seen_this_beat=new.has_ejected,
            volume_ml=g.volumes.sum().item(),volume_rate_ml_s=volume_rate.item(),mass_error_ml_s=mass_error.item(),
            max_cell_flux_residual_ml_s=residual.abs().max().item(),max_cell_divergence_s_inv=(residual/g.volumes).abs().max().item(),
            max_gcl_error_ml_s=g.gcl_max_ml_s,wall_relative_flux_max_ml_s=wall_relative.abs().max().item(),
            pc_mmhg=trial.pc_mmhg.item(),mv_port_pressure_mmhg=expected[0].item()/C,av_port_pressure_mmhg=expected[1].item()/C,
            pressure_mean_mmhg=((p*g.volumes).sum()/g.volumes.sum()).item()/C,mv_closed_effective_pressure_mmhg=effective[0].item()/C,av_closed_effective_pressure_mmhg=effective[1].item()/C,
            pressure_iterations=pinfo.iterations,viscosity_iterations=vinfo.iterations,convection_cfl=cfl,
            fluid_reaction_norm_dyn=torch.linalg.vector_norm(force).item(),interface_work_error_erg_s=(solid_power-wall_power).item(),
            free_momentum_residual_dyn=free_residual,absolute_pressure_gauge_fixed_by_open_port=any(op.opened),
            two_way_interface_feedback_implemented=False)
        return new,report,force
