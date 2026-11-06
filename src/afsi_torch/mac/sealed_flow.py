"""Adapters for fixed partition boundary data on original MAC time schemes.

3D retains BE diffusion/projection and the original BE FE/IB Newton driver.
2D retains its original explicit momentum/lagged FE update. The new pressure
operator incorporates the impermeable faces, using Galerkin-MG-PCG.
"""
from math import isfinite
import torch
from .sealed_pressure import SealedPressure,gradient,divergence,tightened_pressure_options
from .open_boundary import velocity_laplacian
from .be_workspace import BEHelmholtzWorkspace
from .execution import tensor_kernel


class SealedBEWorkspace(BEHelmholtzWorkspace):
    def load(self,rhs):
        return super().load(self.flow.domain.mask(rhs))

    def _load_density(self,density):
        f=self.flow
        for b,u,value,m in zip(self.rhs,self.u,density,f.domain.masks):
            b.copy_(m*f.dt/f.rho*value);u.copy_(b)

    def _load_departure(self,departure,density):
        f=self.flow
        for b,u,a,value,m in zip(self.rhs,self.u,departure,density,f.domain.masks):
            b.copy_(m*(a+f.dt/f.rho*value));u.copy_(b)

    def _advance(self):
        f=self.flow;source,target=self.u,self.next
        for _ in range(f.options.check_every):
            for c,(out,u,b,d,m) in enumerate(zip(target,source,self.rhs,f.diagonals,f.domain.masks)):
                out.copy_(m*(u+(b-u+f.dt*f.mu/f.rho*velocity_laplacian(u,c,f.grid.spacing))/d))
            source,target=target,source
        if f.options.check_every%2:
            for out,value in zip(self.u,source): out.copy_(value)
        return self.u


class SealedBEFlow:
    def __getattr__(self,name): return getattr(self.base,name)

    def __init__(self,base,domain,*,fused=True,graphs=True,helmholtz_solver='jacobi'):
        self.base,self.domain=base,domain
        self.grid,self.dt,self.rho,self.mu=base.grid,base.dt,base.rho,base.mu
        self.options,self.diagonals=base.options,base.diagonals
        self.pressure_solver=SealedPressure(base.grid,domain.masks,domain.planes,
            options=tightened_pressure_options(base.pressure_solver.options,
                domain.config.get('pressure_solver_tolerances')),fused=fused,graphs=graphs,
            max_iterations=domain.config['pressure_max_iterations'],
            compact_geometry=domain.config.get('pressure_compact_geometry',False))
        self.workspace=SealedBEWorkspace(self,fused=fused,graph=self.options.helmholtz_backend == 'graph')
        self.calls=self.pressure_cycles=self.helmholtz_sweeps=0
        self._metrics=tensor_kernel(self._measure,domain.masks[0].device) if fused else self._measure
        self._residuals=tensor_kernel(self._measure_residuals,domain.masks[0].device) if fused else self._measure_residuals
        if helmholtz_solver not in ('jacobi','pcg'):
            raise ValueError('sealed BE Helmholtz solver must be jacobi or pcg')
        self.helmholtz_solver=helmholtz_solver
        self.viscous_pcg=None
        if helmholtz_solver=='pcg':
            from .helmholtz_pcg import BEViscousPCG
            self.viscous_pcg=BEViscousPCG(self.grid,self.dt,self.rho,self.mu,domain.masks,self.diagonals,
                fused=fused,max_iterations=min(2000,self.options.max_sweeps),check_every=self.options.check_every)

    @property
    def helmholtz_pcg_iterations(self):
        return self.viscous_pcg.iterations if self.viscous_pcg is not None else 0

    def advect(self,u): return self.domain.mask(self.base.advect(self.domain.mask(u)))

    def _measure_residuals(self,u,b):
        return torch.stack([torch.linalg.vector_norm(m*(v-r-self.dt*self.mu/self.rho*
            velocity_laplacian(v,c,self.grid.spacing)))
            for c,(v,r,m) in enumerate(zip(u,b,self.domain.masks))])

    def _measure(self,u,b):
        r=self._measure_residuals(u,b)
        return torch.stack((r,torch.stack([v.norm() for v in b])),dim=1).reshape(-1)

    def _solve(self,u,b,pressure_initial=None):
        numbers=self._metrics(u,b).tolist()
        if not all(isfinite(v) for v in numbers): raise FloatingPointError('nonfinite sealed BE momentum solve')
        targets=[max(self.options.helmholtz_atol,self.options.helmholtz_rtol*v) for v in numbers[1::2]]
        residuals=numbers[::2];sweeps=0
        cg_iterations=0
        if self.viscous_pcg is not None and not all(r<=t for r,t in zip(residuals,targets)):
            u,solve_info=self.viscous_pcg.solve(b,targets)
            cg_iterations=solve_info.iterations
            residuals=self._residuals(u,b).tolist()
            if not all(isfinite(r) and r<=t for r,t in zip(residuals,targets)):
                raise RuntimeError('PCG predictor fails the original physical Helmholtz residual targets')
        while not all(r<=t for r,t in zip(residuals,targets)):
            if sweeps>=self.options.max_sweeps: raise RuntimeError('sealed BE Helmholtz failed to converge')
            u=self.workspace.advance();sweeps+=self.options.check_every
            residuals=self._residuals(u,b).tolist()
            if not all(isfinite(v) for v in residuals):raise FloatingPointError('nonfinite sealed BE residual')
        rhs=-self.rho/self.dt*divergence(u,self.grid.spacing)
        p,info=self.pressure_solver.solve(rhs,pressure_initial)
        g=gradient(p,self.grid.spacing,self.domain.masks)
        velocity=tuple(v-self.dt/self.rho*t for v,t in zip(u,g))
        self.calls+=1;self.pressure_cycles+=info['cycles'];self.helmholtz_sweeps+=sweeps
        return velocity,p,dict(pressure=info,helmholtz_sweeps=sweeps,
            helmholtz_solver=self.helmholtz_solver,helmholtz_pcg_iterations=cg_iterations,
            helmholtz_residuals=residuals,helmholtz_tolerances=targets,
            boundary='sealed reservoirs and embedded normal-flux collars')

    @torch.no_grad()
    def advance(self,departure,density,pressure_initial=None):
        u,b=self.workspace.load_forcing(density,self.domain.mask(departure))
        return self._solve(u,b,pressure_initial)

    @torch.no_grad()
    def response(self,density):
        u,b=self.workspace.load_forcing(density)
        return self._solve(u,b)[0]


class SealedChannelFlow:
    def __getattr__(self,name): return getattr(self.base,name)

    def __init__(self,base,domain,*,fused=True,graphs=True):
        self.base,self.domain=base,domain
        self.pressure_solver=SealedPressure(base.grid,domain.masks,domain.planes,
            options=base.pressure_solver.options,fused=fused,graphs=graphs,
            max_iterations=domain.config['pressure_max_iterations'])
        n=base.grid.shape[0];lo,hi=domain.planes
        cells=torch.arange(n,device=domain.masks[0].device)
        faces=torch.arange(n+1,device=cells.device)
        self.vent_cells=((cells>=lo)&(cells<hi)).to(domain.masks[0].dtype)
        self.vent_faces=((faces>=lo)&(faces<=hi)).to(domain.masks[0].dtype)
        self._predict=tensor_kernel(self._predict_mixed,domain.masks[0].device) if fused else self._predict_mixed

    def _predict_mixed(self,velocity,density):
        from ..mac2d.pressure_channel import open_laplacian
        u,v=velocity;hx,hy=self.grid.spacing
        lu,lv=open_laplacian(velocity,self.grid.spacing)
        # Central exterior: Neumann velocity / reference pressure. End
        # reservoirs retain impermeability and no-slip tangential side data.
        lu=lu.clone();lv=lv.clone()
        lu[:,0]+=2*self.vent_faces*u[:,0]/hy**2
        lu[:,-1]+=2*self.vent_faces*u[:,-1]/hy**2
        lv[:,0]+=self.vent_cells*(v[:,1]+v[:,0])/hy**2
        lv[:,-1]+=self.vent_cells*(v[:,-2]+v[:,-1])/hy**2
        ux=.5*(u[1:]+u[:-1])
        du=torch.cat((2*(ux[:1].square()-u[:1].square())/hx,
            torch.diff(ux.square(),dim=0)/hx,2*(u[-1:].square()-ux[-1:].square())/hx),0)
        vp=torch.cat((v[:1],v,v[-1:]),0);ve=.5*(vp[1:]+vp[:-1])
        ghost=(2*self.vent_faces-1)[:,None]
        up=torch.cat((ghost*u[:,:1],u,ghost*u[:,-1:]),1);ue=.5*(up[:,1:]+up[:,:-1])
        du+=torch.diff(ve*ue,dim=1)/hy
        vy=.5*(v[:,1:]+v[:,:-1])
        low=2*(vy[:,:1].square()-v[:,:1].square())/hy*self.vent_cells[:,None]
        high=2*(v[:,-1:].square()-vy[:,-1:].square())/hy*self.vent_cells[:,None]
        dv=torch.cat((low,torch.diff(vy.square(),dim=1)/hy,high),1)+torch.diff(ve*ue,dim=0)/hx
        return self.domain.mask(tuple(q+self.dt*(-adv+self.mu/self.rho*lap+f/self.rho)
            for q,adv,lap,f in zip(velocity,(du,dv),(lu,lv),density)))

    @torch.no_grad()
    def step(self,velocity,density,*,time,pressure_initial=None):
        from ..mac2d.flow import ChannelResult
        if not isfinite(time) or not all(torch.isfinite(v).all() for v in (*velocity,*density)):
            raise ValueError('finite sealed channel input required')
        velocity=self.domain.mask(velocity)
        speed=torch.stack([v.abs().max() for v in velocity])
        cfl=(self.dt*speed/speed.new_tensor(self.grid.spacing)).sum().item()
        centered=(self.dt*speed.square().sum()/(self.mu/self.rho)).item()
        if cfl>.25 or centered>1: raise ValueError('original explicit channel stability guard exceeded')
        tentative=self._predict(velocity,density)
        rhs=-self.rho/self.dt*divergence(tentative,self.grid.spacing)
        p,info=self.pressure_solver.solve(rhs,pressure_initial)
        g=gradient(p,self.grid.spacing,self.domain.masks)
        u=tuple(v-self.dt/self.rho*q for v,q in zip(tentative,g))
        return ChannelResult(u,p,dict(pressure=info,courant=cfl,centered_number=centered,
            viscous_number=self.viscous_number,inlet_time_s=time,
            boundary='sealed reservoirs and embedded normal-flux collars'))
