"""Backward Euler solid inertia and fitted-fluid reaction, solved together.

The inverse of the lagged solid tangent supplies a block preconditioner.
Anderson multisecant updates accelerate the coupled fixed point. Acceptance
always checks the true solid momentum residual, fluid mass, and valve laws.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..nonlinear import gmres,GMRESOptions
from ..moving_cavity.geometry import swept_geometry
from ..ep.windkessel import MMHG_TO_DYN_CM2 as C


@dataclass(frozen=True)
class MechanicalConfig:
    density_g_cm3: float=1.
    force_rtol: float=1e-7
    force_atol_dyn: float=1e-4
    max_iterations: int=40
    history: int=12
    newton_after: int=8
    volume_rate_atol_ml_s: float=1e-8

    def __post_init__(self):
        if any(not isfinite(v) or v<=0 for v in (self.density_g_cm3,self.force_rtol,self.force_atol_dyn,self.volume_rate_atol_ml_s)):
            raise ValueError('positive finite mechanical settings required')
        if any(type(v)!=int or v<1 for v in (self.max_iterations,self.history,self.newton_after)): raise ValueError('invalid iteration limits')


@dataclass(frozen=True)
class SolidState:
    step: int
    time: float
    x: torch.Tensor
    velocity: torch.Tensor
    force: torch.Tensor
    force_time: float
    pressure_time: float
    previous_x: torch.Tensor | None=None
    previous_dt: float | None=None


class TwoWayMechanics:
    def __init__(self,model,blood,config=None):
        self.model=model; self.blood=blood; self.config=config or MechanicalConfig()
        self.mass=model.mesh.X.new_zeros(len(model.mesh.X)).index_add(0,model.mesh.cells.flatten(),
            (model.volumes[:,None].expand(-1,4)*self.config.density_g_cm3/4).flatten())
        if (self.mass<=0).any(): raise ValueError('solid nodes must have positive lumped mass')
        self.tangent=model.tangent_factory()

    @torch.no_grad()
    def step(self,old,blood,dt,time):
        c=self.config; X=old.x; m=self.mass[:,None]/dt**2
        K=self.tangent.assemble(X,time)
        diagonal=K.to_sparse_coo().coalesce(); indices=diagonal.indices(); vv=diagonal.values()
        kd=X.new_zeros(X.numel()).index_add(0,indices[0][indices[0]==indices[1]],vv[indices[0]==indices[1]]).reshape_as(X)
        diag=m-kd
        def H(v): return m*v-torch.sparse.mm(K,v.reshape(-1,1)).reshape_as(X)
        def inverse(r):
            # Active follower terms may make H nonsymmetric: use GMRES.
            v,info=gmres(H,r,precondition=lambda a:a/diag,
                options=GMRESOptions(rtol=1e-6,atol=1e-12,max_iterations=800,restart=60,check_every=20,check_policy='estimated'))
            return v
        if (diag<=0).any(): raise RuntimeError('solid tangent preconditioner has nonpositive diagonal')
        baseline=swept_geometry(self.blood.mesh,X,X,dt)
        boundary=self.blood.mesh.tags>=0
        pg=self.blood.mesh.transfer_reaction(baseline.areas.masked_fill(~boundary[:,None],0.))
        h=inverse(C*pg); schur=(pg*h).sum()
        if schur<=0: raise RuntimeError('closed-volume Schur preconditioner must be positive')
        pscale=max(h.norm().item(),1e-12)
        reasons=[]
        self.histories=[]
        # Try the previously accepted active set first, then alternatives.
        gates=[blood.opened]+[(False,False),(True,False),(False,True),(True,True)]
        gates=list(dict.fromkeys(gates))
        for opened in gates:
            closed=not any(opened)
            u=torch.cat((torch.zeros_like(X).flatten(),X.new_tensor([blood.pressure.mean().item()/C*pscale])))
            if not closed: u[-1]=0.
            us=[]; ds=[]; converged=False
            try:
                solve_inverse=inverse
                if any(opened):
                    # The small basal opening creates a large global added
                    # mass for the volume mode. Include that rank-one fluid
                    # response in the solid inverse (Sherman-Morrison).
                    base=self.blood.trial(blood,X,dt,opened)
                    direction=h/h.norm(); epsilon=1e-7
                    probe=self.blood.trial(blood,X+epsilon*direction,dt,opened)
                    w=-(probe['force']-base['force'])/(epsilon*(pg*direction).sum())
                    z=inverse(w); denom=1+(pg*z).sum()
                    if denom<=0: raise RuntimeError('volume-mode added-mass preconditioner lost positive response')
                    def solve_inverse(r):
                        a=inverse(r)
                        return a-z*(pg*a).sum()/denom
                self.last_history=[]
                self.histories.append(self.last_history)
                def evaluate(value):
                    coords=X+value[:-1].reshape_as(X); self.model.validate(coords)
                    flow=self.blood.trial(blood,coords,dt,opened,value[-1]/pscale)
                    solid_force=self.model.force(coords,time)
                    r=m*(coords-X-dt*old.velocity)-solid_force-flow['force']
                    rate=(flow['g'].volumes-flow['g'].old_volumes).sum().item()/dt
                    merit=r.norm().item()
                    if closed: merit=(merit**2+(rate*dt*C*pg.norm().item()/schur.item())**2)**.5
                    return coords,flow,solid_force,r,rate,merit
                evaluated=None
                for iteration in range(c.max_iterations+1):
                    x,tr,sf,residual,vrate,merit=evaluate(u) if evaluated is None else evaluated
                    fscale=max(sf.norm().item(),tr['force'].norm().item(),1.)
                    tolerance=max(c.force_atol_dyn,c.force_rtol*fscale)
                    fnorm=residual.norm().item()
                    self.last_history.append(dict(gate=opened,iteration=iteration,force=fnorm,tolerance=tolerance,
                        volume_rate=vrate,solid_force=sf.norm().item(),fluid_force=tr['force'].norm().item(),
                        pressure_mmhg=(tr['p']*tr['g'].volumes).sum().item()/tr['g'].volumes.sum().item()/C,
                        displacement_max=(x-X).norm(dim=1).max().item()))
                    if fnorm<=tolerance and (not closed or abs(vrate)<=c.volume_rate_atol_ml_s):
                        if not self.blood.admissible(tr):
                            reasons.append(f'{opened}: valve complementarity'); break
                        converged=True; break
                    if iteration==c.max_iterations: raise RuntimeError(f'coupled iteration limit: force {fnorm:g}/{tolerance:g}; dV/dt {vrate:g}')
                    dx=solve_inverse(-residual)
                    dp=X.new_zeros(())
                    if closed:
                        dp=(-vrate*dt-(pg*dx).sum())/schur
                        dx=dx+h*dp
                        # Do not spend precision-limited iterations converging
                        # an active set that is already well outside its valve
                        # bounds and has a negligible remaining pressure update.
                        if (iteration>=2 and abs(dp.item())<1e-3 and fnorm<.1 and abs(vrate)<1e-5
                            and ((tr['effective'][0]<tr['b0'][0]-.01*C) or (tr['effective'][1]>tr['b0'][1]+.01*C))):
                            reasons.append(f'{opened}: inconsistent closed valve pressure'); break
                    d=torch.cat((dx.flatten(),(dp*pscale).reshape(1)))
                    us.append(u.clone()); ds.append(d.clone())
                    candidate=u+d
                    if iteration>=c.newton_after:
                        # Refine the true coupled residual with a numerical
                        # fluid JVP. The solid tangent remains analytic; no
                        # autograd is applied through iterative fluid solves.
                        volume_scale=C*pg.norm()/schur
                        augmented=torch.cat((residual.flatten(),X.new_tensor([vrate*dt])*volume_scale if closed else u[-1:].clone()))
                        def action(v):
                            vx=v[:-1].reshape_as(X); magnitude=vx.norm()
                            if magnitude>0:
                                epsilon=1e-5
                                perturbed=self.blood.trial(blood,x+epsilon*vx/magnitude,dt,opened,u[-1]/pscale)
                                fluid_jvp=(perturbed['force']-tr['force'])*(magnitude/epsilon)
                            else: fluid_jvp=torch.zeros_like(X)
                            value=H(vx)-fluid_jvp
                            last=v[-1]
                            if closed:
                                value=value-C*pg*(v[-1]/pscale)
                                last=(pg*vx).sum()*volume_scale
                            return torch.cat((value.flatten(),last.reshape(1)))
                        def precondition(r):
                            a=solve_inverse(r[:-1].reshape_as(X)); pressure=r[-1]
                            if closed:
                                pressure=(r[-1]/volume_scale-(pg*a).sum())/schur
                                a=a+h*pressure
                                pressure=pressure*pscale
                            return torch.cat((a.flatten(),pressure.reshape(1)))
                        correction,linear_info=gmres(action,-augmented,precondition=precondition,
                            options=GMRESOptions(rtol=.02,atol=.2*tolerance,restart=40,max_iterations=160,
                                check_every=20,check_policy='estimated',true_check_interval=40))
                        candidate=u+correction
                    elif len(us)>1:
                        width=min(c.history,len(us)-1)
                        DU=torch.stack([us[j+1]-us[j] for j in range(len(us)-width-1,len(us)-1)],1)
                        DD=torch.stack([ds[j+1]-ds[j] for j in range(len(ds)-width-1,len(ds)-1)],1)
                        gram=DD.T@DD
                        reg=max(gram.diagonal().max().item()*1e-12,1e-30)
                        gamma=torch.linalg.solve(gram+reg*torch.eye(width,device=X.device,dtype=X.dtype),DD.T@d)
                        candidate=candidate-(DU+DD)@gamma
                    if not torch.isfinite(candidate).all(): raise RuntimeError('nonfinite coupled multisecant update')
                    # Safeguard multisecant acceleration with true residuals
                    # and moving-mesh/CFL screens, never a geometry projection.
                    accepted=False
                    for direction in (candidate-u,d):
                        for backtrack in range(14):
                            trial_u=u+direction*2.**(-backtrack)
                            try: trial_eval=evaluate(trial_u)
                            except (ValueError,FloatingPointError): continue
                            if trial_eval[-1]<=max(1.5*merit,tolerance):
                                u=trial_u; evaluated=trial_eval; accepted=True; break
                        if accepted: break
                    if not accepted: raise RuntimeError('coupled multisecant backtracking failed')
                if not converged: continue
                bn,diag,force=self.blood.commit(blood,x,dt,tr)
                velocity=(x-X)/dt
                sn=SolidState(old.step+1,time,x,velocity,sf,time,time,X.clone(),dt)
                diag.update(solid_momentum_residual_norm_dyn=fnorm,solid_momentum_tolerance_dyn=tolerance,
                    solid_velocity_max_cm_s=velocity.norm(dim=1).max().item(),coupled_iterations=iteration,
                    solid_density_g_cm3=c.density_g_cm3,
                    original_box_fluid_used=False)
                return sn,bn,diag,force
            except (RuntimeError,ValueError,FloatingPointError) as exc: reasons.append(f'{opened}: {exc}')
        raise RuntimeError('no converged admissible two-way state: '+'; '.join(reasons))
