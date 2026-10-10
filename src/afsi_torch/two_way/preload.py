"""Passive static preload consistent with the CLOSED fitted blood interface.

Numerical continuation does not advance voltage, cell gates or circulation.
Optional nominal loading time separately updates strain/distortion history.
The reference mesh/material fields are retained. Pressure acts on the entire
closed interface, including the idealized basal cap, exactly as in the FSI
reaction. No additional endocardial pressure is applied after initialization.

Explicit pressure_load and geometry_validator adapters also allow the source
endocardial-traction problem to reuse static continuation on the original MAC
framework; the default closed fitted-fluid path is unchanged.
"""
from dataclasses import dataclass,field,replace
from math import isfinite
import torch
from ..moving_cavity.geometry import area_vectors,tetra_volumes,swept_geometry
from ..ep.windkessel import MMHG_TO_DYN_CM2 as C
from ..ep.buffered_load import ActiveLoadState
from ..nonlinear import NewtonOptions,GMRESOptions,newton,NonlinearFailure


class ClosedPressure:
    """p*dV/dx and its exact directional derivative, using the same H/H.T."""
    def __init__(self,mesh):
        self.mesh=mesh; self.faces=mesh.faces[mesh.tags>=0]

    def transfer(self,force):
        m=self.mesh; width=force.shape[1:]
        local=force[:,None].expand(-1,3,*width)/3
        nodal=force.new_zeros((m.mapping.shape[0],*width)).index_add(0,self.faces.flatten(),local.reshape(-1,*width))
        result=torch.sparse.mm(m.mapping.transpose(0,1),nodal.reshape(len(nodal),-1))
        return result.reshape(m.mapping.shape[1],*width)

    def volume(self,x): return tetra_volumes(self.mesh.coordinates(x),self.mesh.cells).sum()
    def gradient(self,x): return self.transfer(area_vectors(self.mesh.coordinates(x),self.faces))
    def force(self,x,pressure_mmhg): return pressure_mmhg*C*self.gradient(x)

    def action(self,x,pressure_mmhg,v):
        shape=v.shape; many=v.ndim==3
        dx=torch.sparse.mm(self.mesh.mapping,v.reshape(len(x),-1)).reshape(self.mesh.mapping.shape[0],*shape[1:])
        nodes=self.mesh.coordinates(x)[self.faces]; delta=dx[self.faces]
        e1=nodes[:,1]-nodes[:,0]; e2=nodes[:,2]-nodes[:,0]
        if many: e1=e1[:,:,None]; e2=e2[:,:,None]
        da=.5*(torch.linalg.cross(delta[:,1]-delta[:,0],e2,dim=1)+torch.linalg.cross(e1,delta[:,2]-delta[:,0],dim=1))
        return pressure_mmhg*C*self.transfer(da)


class EndocardialPressure:
    """Pinned source ds(2) follower pressure; comparison only, no cap load."""
    def __init__(self,solid_mesh,volume_mesh):
        self.faces=solid_mesh.surface(1); self.volume_mesh=volume_mesh
    def volume(self,x): return tetra_volumes(self.volume_mesh.coordinates(x),self.volume_mesh.cells).sum()
    def transfer(self,force,x):
        width=force.shape[1:]
        local=force[:,None].expand(-1,3,*width)/3
        return x.new_zeros((len(x),*width)).index_add(0,self.faces.flatten(),local.reshape(-1,*width))
    def force(self,x,pressure_mmhg):
        # Solid-outward normal points into the blood: reverse for inflation.
        return -pressure_mmhg*C*self.transfer(area_vectors(x,self.faces),x)
    def action(self,x,pressure_mmhg,v):
        n=x[self.faces]; d=v[self.faces]; e1=n[:,1]-n[:,0]; e2=n[:,2]-n[:,0]
        if v.ndim==3: e1=e1[:,:,None]; e2=e2[:,:,None]
        da=.5*(torch.linalg.cross(d[:,1]-d[:,0],e2,dim=1)+torch.linalg.cross(e1,d[:,2]-d[:,0],dim=1))
        return -pressure_mmhg*C*self.transfer(da,x)


@dataclass(frozen=True)
class PreloadConfig:
    target_mmhg: float=8.
    initial_step_mmhg: float=.5
    maximum_step_mmhg: float=1.
    minimum_step_mmhg: float=.03125
    force_rtol: float=1e-8
    force_atol_dyn: float=1e-3
    coarse_shape: tuple=(4,4,5)
    nominal_steps: int=0  # 0: adaptive static loading; >0: land-history sample grid
    pressure_predictor: bool=True
    nonlinear: NewtonOptions=field(default_factory=lambda:NewtonOptions(rtol=1e-8,atol=1e-3,max_iterations=40,
        linear=GMRESOptions(rtol=.01,atol=1e-10,restart=100,max_iterations=4000,check_policy='estimated',true_check_interval=100),
        linear_tolerance_fraction=.2,linear_forcing='fixed'))

    def __post_init__(self):
        object.__setattr__(self,'coarse_shape',tuple(self.coarse_shape))
        if any(not isfinite(v) or v<=0 for v in (self.target_mmhg,self.initial_step_mmhg,self.maximum_step_mmhg,
            self.minimum_step_mmhg,self.force_rtol,self.force_atol_dyn)): raise ValueError('positive preload settings required')
        if not self.minimum_step_mmhg<=self.initial_step_mmhg<=self.maximum_step_mmhg: raise ValueError('invalid continuation increments')
        if len(self.coarse_shape)!=3 or any(type(n)!=int or n<2 for n in self.coarse_shape): raise ValueError('coarse grid needs 3 sizes >=2')
        if type(self.nominal_steps)!=int or self.nominal_steps<0: raise ValueError('nonnegative integer nominal steps required')
        if type(self.pressure_predictor)!=bool: raise ValueError('pressure predictor must be boolean')


def coarse_basis(X,shape):
    """GPU trilinear displacement space; contains all affine/rigid fields."""
    lower=X.min(0).values; lengths=X.max(0).values-lower
    if (lengths<=0).any(): raise ValueError('3D solid needed for preload coarse basis')
    shape_tensor=torch.tensor(shape,device=X.device); coords=(X-lower)/lengths*(shape_tensor-1)
    base=coords.floor().to(torch.int64); base=torch.minimum(base,shape_tensor-2); fraction=coords-base
    scalar=X.new_zeros((len(X),shape[0]*shape[1]*shape[2]))
    for i in (0,1):
        for j in (0,1):
            for k in (0,1):
                corner=base+base.new_tensor([i,j,k]); index=(corner[:,0]*shape[1]+corner[:,1])*shape[2]+corner[:,2]
                weight=(fraction[:,0] if i else 1-fraction[:,0])*(fraction[:,1] if j else 1-fraction[:,1])*(fraction[:,2] if k else 1-fraction[:,2])
                scalar.scatter_add_(1,index[:,None],weight[:,None])
    scalar=scalar[:,scalar.abs().sum(0)>1e-14]
    # Remove genuine linear dependencies before coarse LU factorization.
    values,vectors=torch.linalg.eigh(scalar.T@scalar)
    keep=values>values.max()*1e-12
    scalar=scalar@(vectors[:,keep]/values[keep].sqrt()[None])
    basis=X.new_zeros((len(X),3,scalar.shape[1],3))
    for i in range(3): basis[:,i,:,i]=scalar
    return basis.reshape(X.numel(),-1)


class CoarseInverse:
    """GPU two-level smoothing/coarse correction for Newton-GMRES.

The full source tangent can be nonsymmetric. Coarse LU and outer GMRES
retain that tangent; the coarse space changes convergence, never physics.
"""
    def __init__(self,action,diagonal,basis):
        self.action=action; self.diagonal=diagonal; self.shape=diagonal.shape
        # Estimate the high-frequency scale on this GPU. Fixed damping can
        # amplify volumetric modes for the nearly incompressible source solid.
        v=torch.cos(torch.arange(diagonal.numel(),device=diagonal.device,dtype=diagonal.dtype)).reshape(self.shape)
        for _ in range(20):
            w=action(v)/diagonal; scale=w.norm()
            if not torch.isfinite(scale) or scale<=0: raise RuntimeError('invalid preload smoother scale')
            v=w/scale
        self.omega=1.2/scale.item()
        self.P=basis.clone()
        # Smoothed interpolation follows the assembled material tangent,
        # rather than assuming a regular Cartesian elasticity operator.
        for _ in range(12):
            AP=action(self.P.reshape(*self.shape,-1)).reshape(self.P.shape)
            self.P-=self.omega*AP/diagonal.flatten()[:,None]
        AP=action(self.P.reshape(*self.shape,-1)).reshape(self.P.shape)
        coarse=self.P.T@AP
        self.LU,self.pivots=torch.linalg.lu_factor(coarse)
        if not torch.isfinite(self.LU).all(): raise RuntimeError('invalid preload coarse factorization')
    def __call__(self,r):
        z=self.omega*r/self.diagonal
        reduced=r-self.action(z)
        rhs=self.P.T@reduced.flatten()
        coeff=torch.linalg.lu_solve(self.LU,self.pivots,rhs[:,None])[:,0]
        z=z+(self.P@coeff).reshape(self.shape)
        return z+self.omega*(r-self.action(z))/self.diagonal


class PreloadFailure(RuntimeError):
    def __init__(self,message,accepted_x,history,rejected):
        super().__init__(message); self.accepted_x=accepted_x; self.history=history; self.rejected=rejected


@dataclass(frozen=True)
class PreloadResult:
    x: torch.Tensor
    pressure_mmhg: float
    force_residual_norm_dyn: float
    force_tolerance_dyn: float
    volume_ml: float
    reference_volume_ml: float
    history: list
    rejected: list


class PassivePreloader:
    def __init__(self,driver,config=None,*,pressure_load=None,geometry_validator=None):
        self.driver=driver; self.config=config or PreloadConfig()
        self.pressure=ClosedPressure(driver.blood_driver.mesh) if pressure_load is None else pressure_load
        self.geometry_validator=geometry_validator
        self.basis=coarse_basis(driver.mesh.X,self.config.coarse_shape)
        from ..paper_lv import PaperHOTangent
        # Static preload explicitly has zero active stress: its exact active
        # Jacobian is zero. Avoid assembling the full Land element Jacobian.
        self.tangent=PaperHOTangent(driver.model)

    @torch.no_grad()
    def solve(self,progress=None,*,observe_state=None):
        d=self.driver; c=self.config; X=d.mesh.X
        zero=X.new_zeros(len(X)); load=ActiveLoadState(X,zero,zero,zero,zero,d.config.fsi.time.dt,False)
        d.buffer.stage(load); x=X.clone(); pressure=0.; step=c.initial_step_mmhg; history=[]; rejected=[]
        nominal=0; previous_x=None; previous_pressure=None
        while pressure<c.target_mmhg-1e-12:
            next_target=c.target_mmhg if not c.nominal_steps else c.target_mmhg*(nominal+1)/c.nominal_steps
            candidate=min(pressure+step,next_target,c.target_mmhg); cache={}
            def validate(y):
                d.model.validate(y)
                if self.geometry_validator is None: swept_geometry(d.blood_driver.mesh,x,y,.001)
                else: self.geometry_validator(x,y)
            def residual(y): return -d.model.force(y,d.config.start_time_s)-self.pressure.force(y,candidate)
            def linearization(y):
                K=self.tangent.assemble(y,d.config.start_time_s)
                def action(v):
                    elastic=-torch.sparse.mm(K,v.reshape(y.numel(),-1)).reshape(v.shape)
                    return elastic-self.pressure.action(y,candidate,v)
                diagonal=-self.tangent.diagonal(K).reshape_as(y)
                if (diagonal<=0).any(): raise RuntimeError('nonpositive passive tangent diagonal')
                cache.update(action=action,diagonal=diagonal); return action
            inverse=lambda y:CoarseInverse(cache['action'],cache['diagonal'],self.basis)
            force_scale=max(self.pressure.force(x,candidate).norm().item(),1.)
            target=max(c.force_atol_dyn,c.force_rtol*force_scale)
            options=replace(c.nonlinear,rtol=0.,atol=target)
            guess=x
            if c.pressure_predictor and previous_x is not None:
                predicted=x+(candidate-pressure)/(pressure-previous_pressure)*(x-previous_x)
                try:
                    validate(predicted)
                    # Extrapolation may remove slow elastic modes while
                    # temporarily increasing a stiff volumetric residual.
                    # Bound the hint; its final true force tolerance is unchanged.
                    if residual(predicted).norm()<50*residual(x).norm().clamp_min(1.): guess=predicted
                except (ValueError,FloatingPointError,RuntimeError):
                    pass  # A hint can fail without changing the accepted load state.
            try:
                predictor_failed=False
                try:
                    result=newton(residual,guess,validate=validate,linearization_factory=linearization,preconditioner_factory=inverse,options=options)
                except NonlinearFailure:
                    if guess is x: raise
                    predictor_failed=True
                    result=newton(residual,x,validate=validate,linearization_factory=linearization,preconditioner_factory=inverse,options=options)
                final=residual(result.x).norm().item()
                if final>target: raise RuntimeError('preload true equilibrium residual failed')
                previous_x,previous_pressure=x,pressure
                x=result.x; pressure=candidate
                if c.nominal_steps and abs(pressure-next_target)<=1e-12: nominal+=1
                entry=dict(pressure_mmhg=pressure,residual_norm_dyn=final,tolerance_dyn=target,volume_ml=self.pressure.volume(x).item(),
                    newton_iterations=result.iterations,linear_iterations=sum(h.get('linear',{}).get('iterations',0) for h in result.history),
                    maximum_displacement_cm=(x-X).norm(dim=1).max().item(),pressure_predictor_used=guess is not previous_x,
                    pressure_predictor_fallback=predictor_failed,history=result.history)
                history.append(entry)
                step=min(c.maximum_step_mmhg,step*(1.5 if result.iterations<=6 else 1.))
            except (RuntimeError,ValueError,FloatingPointError) as exc:
                rejected.append(dict(from_mmhg=pressure,to_mmhg=candidate,error=str(exc)))
                step=(candidate-pressure)/2
                if step<c.minimum_step_mmhg:
                    raise PreloadFailure('preload continuation failed; only converged loading states retained',x.clone(),history,rejected) from exc
                continue
            if observe_state: observe_state(entry,x)
            if progress: progress(entry)
        sf=d.model.force(x,d.config.start_time_s); fp=self.pressure.force(x,pressure)
        tolerance=max(c.force_atol_dyn,c.force_rtol*max(sf.norm().item(),fp.norm().item(),1.))
        error=(sf+fp).norm().item()
        if error>tolerance: raise PreloadFailure('final passive preload equilibrium failed',x.clone(),history,rejected)
        return PreloadResult(x,pressure,error,tolerance,self.pressure.volume(x).item(),self.pressure.volume(X).item(),history,rejected)


@dataclass(frozen=True)
class PreloadStrainHistory:
    x: torch.Tensor
    stretch: torch.Tensor
    zs: torch.Tensor
    zw: torch.Tensor
    pressure_mmhg: float
    duration_s: float
    steps: int


class SourceLoadingHistory:
    """Source 0--500 ms passive pressure ramp: EP/GPB/Land frozen, Z evolves.

Only converged nominal physical loading endpoints publish strain history.
Intermediate numerical continuation states never advance this clock.
"""
    def __init__(self,driver,target_mmhg=8.,duration_s=.5,steps=100):
        if type(steps)!=int or steps<1 or not isfinite(target_mmhg) or target_mmhg<=0 or not isfinite(duration_s) or duration_s<=0:
            raise ValueError('positive loading history pressure/duration and integer steps required')
        self.driver=driver; self.target=target_mmhg; self.duration=duration_s; self.steps=steps; self.accepted=0
        self.stretch=torch.ones_like(driver.mesh.X[:,0]); self.zs=torch.zeros_like(self.stretch); self.zw=self.zs.clone(); self.x=None
    @torch.no_grad()
    def observe(self,entry,x):
        from ..ep.land import distortion_step
        p=entry['pressure_mmhg']; expected=self.target*(self.accepted+1)/self.steps
        entry['physical_loading_endpoint']=False
        if p>expected+1e-10: raise ValueError('loading skipped a source strain-history endpoint')
        if abs(p-expected)>1e-10: return
        F=self.driver.model.element_gradient(x)
        cell=torch.linalg.vector_norm(torch.einsum('eij,ej->ei',F,self.driver.mesh.fiber),dim=-1)
        stretch=self.driver.ep.project_cell_field(cell)
        if not torch.isfinite(stretch).all() or (stretch<=0).any(): raise FloatingPointError('invalid loading stretch projection')
        delta=stretch-self.stretch; dt=self.duration/self.steps
        zs=distortion_step(self.zs,delta,dt,40.14); zw=distortion_step(self.zw,delta,dt,405.86)
        if not torch.isfinite(zs).all() or not torch.isfinite(zw).all(): raise FloatingPointError('invalid loading distortion')
        self.stretch,self.zs,self.zw,self.x=stretch,zs,zw,x.clone(); self.accepted+=1
        entry.update(physical_loading_endpoint=True,physical_loading_time_s=self.duration*self.accepted/self.steps,
            zs_max_abs=zs.abs().max().item(),zw_max_abs=zw.abs().max().item())
    def final(self):
        if self.accepted!=self.steps or self.x is None: raise ValueError('incomplete source loading history')
        return PreloadStrainHistory(self.x,self.stretch,self.zs,self.zw,self.target,self.duration,self.steps)


def initialize_preloaded(driver,result,*,strain_history=None):
    """Static fluid-solid initialization at pacing time; GPB/Land unadvanced."""
    from .driver import ThreeFieldState
    c=driver.blood_driver.config; p=result.pressure_mmhg
    if abs(c.atrium_mmhg-p)>1e-10 or c.initial_pc_mmhg<p:
        raise ValueError('static preloaded blood needs atrium == cavity pressure and arterial Pc >= cavity pressure')
    state=driver.initialize_three_fields(x=result.x)
    e=state.ep; b=state.blood
    if strain_history is not None:
        from ..ep.driver import StrainHistory
        h=strain_history
        if abs(h.pressure_mmhg-p)>1e-12 or not torch.equal(h.x,result.x): raise ValueError('loading history belongs to a different preload state')
        driver.ode._validate(e.tissue,h.stretch,h.zs,h.zw,e.fsi.time)
        if not torch.allclose(h.stretch,e.strain.stretch,rtol=1e-11,atol=1e-11): raise ValueError('loading stretch differs from preloaded geometry')
        e=replace(e,strain=StrainHistory(h.stretch.clone(),h.zs.clone(),h.zw.clone()))
    # The static initial force history is passive, even though the next
    # interval is the first paced Land interval.
    load=replace(e.force_load,use_land=False,anchor=None,ramp_factor=0.,zs_old=e.strain.zs.clone(),zw_old=e.strain.zw.clone())
    driver.buffer.stage(load); sf=driver.model.force(e.fsi.x,e.fsi.time)
    e=replace(e,force_load=load,fsi=replace(e.fsi,force=sf))
    b=replace(b,pressure=torch.full_like(b.pressure,p*C),port_pressure_cgs=b.port_pressure_cgs.new_tensor([c.atrium_mmhg,c.initial_pc_mmhg])*C)
    trial=driver.blood_driver.trial(b,e.fsi.x,c.dt_s,(False,False),p)
    if not driver.blood_driver.admissible(trial): raise ValueError('initial preloaded valves are not admissible')
    force=trial['force']; error=(sf+force).norm().item()
    if error>result.force_tolerance_dyn: raise RuntimeError('preload pressure force differs from the actual closed blood reaction')
    if trial['u'].abs().max()>1e-12 or trial['op'].divergence(trial['u']).abs().max()>1e-12:
        raise RuntimeError('preload creates spurious initial fluid motion')
    initialized=ThreeFieldState(e,b)
    report=dict(preload_performed=True,preload_pressure_mmhg=p,preload_volume_ml=result.volume_ml,
        preload_reference_volume_ml=result.reference_volume_ml,preload_force_residual_norm_dyn=error,
        preload_force_tolerance_dyn=result.force_tolerance_dyn,initial_solid_velocity_max_cm_s=0.,initial_blood_velocity_max_cm_s=0.,
        atrium_mmhg=c.atrium_mmhg,initial_pc_mmhg=c.initial_pc_mmhg,
        initial_pc_runoff_ml_s=(c.initial_pc_mmhg-c.windkessel.venous_mmhg)/c.windkessel.rp_mmhg_s_per_ml,
        preload_loading_steps=len(result.history),preload_rejected_attempts=len(result.rejected),
        initialization='passive static closed-interface preload; matched atrial pressure; circulation starts at first pacing')
    report.update(mechanical_equilibrium_established=True,circulation_stationary=abs(c.initial_pc_mmhg-c.windkessel.venous_mmhg)<=1e-12,
        ep_advanced_during_preload=False,circulation_advanced_during_preload=False,
        initial_cavity_pressure_mmhg=p,initial_q_in_ml_s=0.,initial_q_out_ml_s=0.,
        preload_force_relative_residual=error/max(sf.norm().item(),force.norm().item(),1.),
        minimum_solid_jacobian=torch.linalg.det(driver.model.element_gradient(e.fsi.x)).min().item(),
        maximum_displacement_cm=(e.fsi.x-driver.mesh.X).norm(dim=1).max().item(),
        reference_state_assumption='input geometry retained as unloaded reference; inverse prestress not performed')
    report.update(loading_history_mode='source-pressure-ramp' if strain_history is not None else 'static-zero-z',
        loading_history_duration_s=strain_history.duration_s if strain_history is not None else None,
        loading_history_steps=strain_history.steps if strain_history is not None else 0,
        initial_zs_max_abs=e.strain.zs.abs().max().item(),initial_zw_max_abs=e.strain.zw.abs().max().item())
    return initialized,report
