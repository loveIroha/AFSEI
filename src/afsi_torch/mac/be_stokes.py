"""BE coupled MAC Stokes, ported from AFSI_GPU stokes_fgmres.py (9c39e32).

Retains the upstream scaled saddle operator, flexible right-preconditioned
GMRES, pressure-free projection preconditioner and true physical checks.
Adaptations: alpha=dt*mu/rho (BE, not CN); pressure-Dirichlet/velocity-Neumann
outer box uses open_boundary operators, with NO zero-normal or mean removal.
These boundary conditions are AFSEI case data, not the paper's closed box.
"""
from dataclasses import replace
from math import isfinite
import torch
from .backward_euler import BackwardEulerFlow
from .grid import divergence
from .open_boundary import gradient, velocity_laplacian
from ..nonlinear import gmres, GMRESOptions
from ..krylov_workspace import KrylovWorkspace


class BEStokesFlow(BackwardEulerFlow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.alpha=self.dt*self.mu/self.rho
        self.krylov=KrylovWorkspace()
        self.stokes_preconditioner_rtol=.01

    def lap(self,u):
        return tuple(velocity_laplacian(v,j,self.grid.spacing) for j,v in enumerate(u))

    def helmholtz(self,b):
        u=b
        targets=[max(self.options.helmholtz_atol,self.options.helmholtz_rtol*v.norm().item()) for v in b]
        count=0
        while True:
            residuals=self._measure_residuals(u,b).tolist()
            if not all(isfinite(v) for v in residuals): raise FloatingPointError('BE Stokes Helmholtz residual')
            if all(r<=t for r,t in zip(residuals,targets)): break
            if count>=self.options.max_sweeps: raise RuntimeError('BE Stokes Helmholtz failed')
            u=self._smooth(u,b); count+=self.options.check_every
        self.helmholtz_sweeps+=count
        return u

    @torch.no_grad()
    def solve_rhs(self,b,pressure_initial=None):
        self.grid.check_velocity(b)
        template=b[0].new_zeros(self.grid.shape)
        p=template if pressure_initial is None else pressure_initial.clone()
        if p.shape!=template.shape or not torch.isfinite(p).all(): raise ValueError('invalid BE Stokes pressure')
        h=min(self.grid.spacing); scale=self.dt/self.rho
        sizes=tuple(v.numel() for v in b); shapes=tuple(v.shape for v in b)
        def pack(u,q): return torch.cat(tuple(v.reshape(-1) for v in u)+(q.reshape(-1),))
        def unpack(v):
            parts=v.split((*sizes,template.numel()))
            return tuple(z.reshape(s) for z,s in zip(parts,shapes)),parts[-1].reshape(template.shape)
        def action(v):
            u,q=unpack(v); g=gradient(q,self.grid.spacing)
            return pack(tuple(a-self.alpha*l+h*t for a,l,t in zip(u,self.lap(u),g)),
                        -h*divergence(u,self.grid.spacing))
        cycles=0; poisson_solves=0
        def precondition(v):
            nonlocal cycles,poisson_solves
            r,s=unpack(v); z=self.helmholtz(r)
            rhs=-s/h-divergence(z,self.grid.spacing)
            saved=self.pressure_solver.options
            self.pressure_solver.options=replace(saved,rtol=self.stokes_preconditioner_rtol)
            try: phi,info=self.pressure_solver.solve(rhs)
            finally: self.pressure_solver.options=saved
            cycles+=info['cycles']; poisson_solves+=1
            g=gradient(phi,self.grid.spacing)
            return pack(tuple(a-t for a,t in zip(z,g)),(phi+self.alpha*rhs)/h)
        norm=torch.sqrt(sum(v.square().sum() for v in b)).item()
        if not isfinite(norm): raise ValueError('nonfinite BE Stokes RHS')
        opt=self.pressure_solver.options
        momentum_tol=max(opt.atol,opt.rtol*norm)
        z=self.helmholtz(b)
        div_rhs=-divergence(z,self.grid.spacing)/scale
        div_tol=scale*max(opt.atol,opt.rtol*div_rhs.norm().item())
        div_tol=max(div_tol,100*torch.finfo(p.dtype).eps*max(norm,torch.finfo(p.dtype).tiny)*sum(1/d for d in self.grid.spacing))
        guess_u=self.helmholtz(tuple(v-scale*g for v,g in zip(b,gradient(p,self.grid.spacing))))
        guess=pack(guess_u,scale/h*p)
        rhs=pack(b,template)-action(guess)
        controls=GMRESOptions(rtol=0.,atol=.5*min(momentum_tol,h*div_tol),restart=20,
                              max_iterations=80,check_policy='estimated',true_check_interval=5)
        correction,info=gmres(action,rhs,precondition=precondition,options=controls,workspace=self.krylov)
        u,q=unpack(guess+correction); p=h/scale*q
        residual=tuple(v-self.alpha*l+scale*g-r for v,l,g,r in zip(u,self.lap(u),gradient(p,self.grid.spacing),b))
        momentum=torch.sqrt(sum(v.square().sum() for v in residual)).item()
        div=divergence(u,self.grid.spacing).norm().item()
        if not isfinite(momentum+div) or momentum>momentum_tol or div>div_tol:
            raise RuntimeError(f'BE Stokes true residual failed: momentum {momentum}/{momentum_tol}, div {div}/{div_tol}')
        self.calls+=1; self.pressure_cycles+=cycles
        return u,p,dict(pressure=dict(cycles=cycles,poisson_solves=poisson_solves,backend=self.pressure_solver.backend),
            stokes=dict(solver='fgmres',time='backward-euler',iterations=info['iterations'],
                momentum_residual=momentum,momentum_tolerance=momentum_tol,
                divergence_norm=div,divergence_tolerance=div_tol,
                preconditioner='projection with geometric multigrid',pressure_gauge='physical outer-face Dirichlet'))

    @torch.no_grad()
    def advance(self,departure_velocity,density,pressure_initial=None):
        return self.solve_rhs(tuple(v+self.dt/self.rho*f for v,f in zip(departure_velocity,density)),pressure_initial)

    @torch.no_grad()
    def response(self,density):
        rhs=tuple(self.dt/self.rho*f for f in density)
        norm=torch.sqrt(sum(v.square().sum() for v in rhs)).item()
        if not isfinite(norm): raise ValueError('nonfinite Stokes response')
        if norm==0: return tuple(torch.zeros_like(v) for v in rhs)
        u,_,_=self.solve_rhs(tuple(v/norm for v in rhs))
        return tuple(norm*v for v in u)
