"""Fixed valve active-set trials for the strongly coupled mechanical solve.

A closed trial projects its incompatible pressure RHS onto the mean-zero
space. Only the outer solid volume equation can make that trial acceptable:
the unmodified physical cell divergence and volume rate are checked at commit.
No accepted flux, velocity or geometry is corrected after the solve.
"""
import torch
from ..moving_cavity.flow import ALEStepper,ALEState
from ..moving_cavity.geometry import swept_geometry
from ..moving_cavity.operators import ALEOperators
from ..fluid.solvers import pcg
from ..ep.windkessel import flow_driven_trial,MMHG_TO_DYN_CM2 as C


class CoupledBlood(ALEStepper):
    @torch.no_grad()
    def trial(self,old,x,dt,opened,pressure_offset_mmhg=0.):
        c=self.config; g=swept_geometry(self.mesh,old.solid_x,x,dt)
        op=ALEOperators(self.mesh,g,opened)
        height=3*g.volumes/g.areas.norm(dim=1)[self.mesh.cell_faces].max(1).values
        speed=(old.velocity[self.mesh.cell_faces]-g.grid_velocity[self.mesh.cell_faces]).mean(1).norm(dim=1)
        cfl=(dt*speed/height).max().item()
        if c.convection and cfl>c.cfl_limit: raise ValueError('coupled ALE convection CFL exceeded')
        star,rhs,vinfo=op.viscous(old.velocity,dt,c.density,c.viscosity,c.solver,c.convection)
        zero=x.new_zeros(2)
        relax=flow_driven_trial(old.pc_mmhg,zero[0],dt,c.windkessel)
        b0=torch.stack((old.pc_mmhg.new_tensor(c.atrium_mmhg),relax.pc_mmhg))*C
        Z=x.new_tensor([c.mv_resistance,relax.impedance_mmhg_s_per_ml+c.av_resistance])*C
        if any(opened):
            base,pbase,pinfo=op.project(star,zero,dt,c.density,c.solver)
            qbase=op.flux(base)
            if all(opened):
                unit=zero.clone(); unit[1]=1.
                du,dp,_=op.project(torch.zeros_like(star),unit,dt,c.density,c.solver)
                qa=op.flux(du,relative=False); Y=torch.stack((qa,-qa),1)
                q=torch.linalg.solve(torch.eye(2,device=x.device,dtype=x.dtype)+Y*Z[None],qbase-Y@(b0-b0[0]))
                b=b0+Z*q; u=base+(b[1]-b[0])*du; p=pbase+b[0]+(b[1]-b[0])*dp
            else:
                b=b0+Z*qbase; u=base; p=pbase+b[0 if opened[0] else 1]
        else:
            prhs=-c.density/dt*op.divergence(star)
            # This is a nonlinear TRIAL extension, never a physical acceptance.
            prhs=prhs-prhs.sum()*g.volumes/g.volumes.sum()
            inverse=lambda r:r/op.pdiag+op.coarse*(op.coarse*r).sum()/op.coarse_energy
            p,pinfo=pcg(op.pressure_action,prhs,op.pdiag,fixed=op.pressure_fixed,
                options=c.solver,preconditioner=inverse)
            u=star+dt/c.density*op.inverse_mass[:,None]*op.transpose(p)
            p=p-(p*g.volumes).sum()/g.volumes.sum()+pressure_offset_mmhg*C
            b=b0
        q=op.flux(u)
        reaction=op.reaction(u,p,star,rhs,dt,c.density,c.viscosity,b)
        force=self.mesh.transfer_reaction(-reaction.masked_fill(~op.fixed[:,None],0.))
        return dict(op=op,g=g,u=u,p=p,q=q,b=b,reaction=reaction,force=force,
            effective=self.closed_pressures(op,reaction),pinfo=pinfo,vinfo=vinfo,cfl=cfl,b0=b0)

    def admissible(self,trial):
        o=trial['op'].opened; q=trial['q']; pe=trial['effective']; b=trial['b0']
        return not ((o[0] and q[0]>1e-8) or (o[1] and q[1]<-1e-8)
            or (not o[0] and pe[0]<b[0]-1e-4) or (not o[1] and pe[1]>b[1]+1e-4))

    def commit(self,old,x,dt,t):
        if not self.admissible(t): raise ValueError('coupled valve inequalities failed')
        c=self.config; op=t['op']; g=t['g']; q=t['q']; force=t['force']
        volume_rate=(g.volumes-g.old_volumes).sum()/dt
        residual=op.divergence(t['u']); mass_error=volume_rate+q.sum()
        leak=((t['u']-g.grid_velocity)*g.areas).sum(1)[op.fixed]
        scale=max(q.abs().max().item(),abs(volume_rate.item()),1.)
        if abs(mass_error.item())>1e-7*scale or residual.abs().max()>1e-7*scale or leak.abs().max()>1e-12*scale:
            raise RuntimeError('coupled accepted-step physical mass check failed')
        if not any(op.opened) and abs(volume_rate.item())>1e-8:
            raise RuntimeError('closed coupled solid volume constraint failed')
        wall_power=-(t['reaction'][op.fixed]*g.grid_velocity[op.fixed]).sum()
        solid_power=(force*(x-old.solid_x)/dt).sum()
        if not torch.isclose(wall_power,solid_power,rtol=1e-10,atol=1e-8): raise RuntimeError('coupled interface virtual work failed')
        wk=flow_driven_trial(old.pc_mmhg,q[1],dt,c.windkessel)
        rcr_balance=c.windkessel.compliance_ml_per_mmhg*(wk.pc_mmhg-old.pc_mmhg)/dt-q[1]+(wk.pc_mmhg-c.windkessel.venous_mmhg)/c.windkessel.rp_mmhg_s_per_ml
        if abs(rcr_balance.item())>1e-8*scale: raise RuntimeError('coupled RCR balance failed')
        expected=torch.stack((old.pc_mmhg.new_tensor(c.atrium_mmhg)+c.mv_resistance*q[0],wk.outlet_mmhg+c.av_resistance*q[1]))*C
        for j,on in enumerate(op.opened):
            if on and not torch.isclose(t['b'][j],expected[j],rtol=1e-9,atol=1e-7): raise RuntimeError('coupled RCR mismatch')
        closure=old.has_ejected and old.opened[1] and not op.opened[1]
        new=ALEState(old.step+1,old.tick+round(dt/c.clock_dt_s),x.clone(),t['u'],t['p'],wk.pc_mmhg,q,expected,op.opened,
            bool(old.has_ejected or (op.opened==(False,True) and q[1]>1e-8)),old.solid_x.clone(),dt)
        report=dict(step=new.step,time_s=new.tick*c.clock_dt_s,dt_s=dt,mv_open=op.opened[0],av_open=op.opened[1],
            accepted_outlet_closure=bool(closure),q_in_ml_s=-q[0].item(),q_out_ml_s=q[1].item(),pc_mmhg=wk.pc_mmhg.item(),
            pressure_mean_mmhg=((t['p']*g.volumes).sum()/g.volumes.sum()).item()/C,
            mv_port_pressure_mmhg=expected[0].item()/C,av_port_pressure_mmhg=expected[1].item()/C,
            mv_closed_effective_pressure_mmhg=t['effective'][0].item()/C,av_closed_effective_pressure_mmhg=t['effective'][1].item()/C,
            rcr_balance_ml_s=rcr_balance.item(),
            volume_ml=g.volumes.sum().item(),volume_rate_ml_s=volume_rate.item(),mass_error_ml_s=mass_error.item(),
            max_cell_flux_residual_ml_s=residual.abs().max().item(),max_cell_divergence_s_inv=(residual/g.volumes).abs().max().item(),
            max_gcl_error_ml_s=g.gcl_max_ml_s,wall_relative_flux_max_ml_s=leak.abs().max().item(),
            interface_work_error_erg_s=(solid_power-wall_power).item(),fluid_reaction_norm_dyn=force.norm().item(),
            free_momentum_residual_dyn=t['reaction'][~op.fixed].abs().max().item(),
            pressure_iterations=t['pinfo'].iterations,viscosity_iterations=t['vinfo'].iterations,convection_cfl=t['cfl'],
            two_way_interface_feedback_implemented=True,closed_volume_constraint_active=not any(op.opened))
        return new,report,force
