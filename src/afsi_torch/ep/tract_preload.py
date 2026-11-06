"""Validation-only equilibrium correction for LV plus fixed tract pressure work."""
from dataclasses import replace
import torch
from .mac_balanced_preload import EndocardialAndCapPressure
from .mac_transport import closed_volume_gradient
from .windkessel import MMHG_TO_DYN_CM2 as C
from ..paper_lv import PaperHOTangent
from ..two_way.preload import PreloadConfig,coarse_basis,CoarseInverse
from ..nonlinear import newton,NonlinearFailure


class TractPressureLoad:
    def __init__(self,model,storage):
        self.closed=EndocardialAndCapPressure(model)
        self.storage=storage
        self._action_key=None

    def volume(self,x):
        return self.closed.volume(x)+self.storage.volume(x)

    def force(self,x,pressure_mmhg,weight=1.):
        return self.closed.force(x,pressure_mmhg)+C*pressure_mmhg*weight*self.storage.discrete(x,x)

    def linearization(self,x,pressure_mmhg,weight=1.):
        # Both endpoints of discrete(x,x) move in a static gradient.
        # derivative(old,x) fixes old, hence its zero-increment limit is H/2.
        tract=self.storage.derivative(x,x)
        return lambda v:self.closed.action(x,pressure_mmhg,v)+2*C*pressure_mmhg*weight*tract(v)

    def action(self,x,pressure_mmhg,v):
        key=(id(x),float(pressure_mmhg))
        if self._action_key!=key:
            self._static_action=self.linearization(x,pressure_mmhg)
            self._action_x=x  # retain the object; Python cannot reuse its id
            self._action_key=key
        return self._static_action(v)


@torch.no_grad()
def correct_equilibrium(driver,x0,domain,pressure_mmhg=8.,*,progress=None):
    """Freeze EP/active load externally; statically continue added tract work.

    This does not publish a cycle state, a loading history, or a restart.
    The caller must have staged a zero active stress before this solve.
    """
    settings=PreloadConfig()
    tangent=PaperHOTangent(driver.model)
    basis=coarse_basis(driver.mesh.X,settings.coarse_shape)
    pressure=TractPressureLoad(driver.model,domain.storage)
    x=x0.clone();weight=0.;step=1.;history=[];rejected=[]
    while weight<1.-1e-12:
        target=min(1.,weight+step);cache={}
        def validate(y):
            driver.model.validate(y)
            domain.validate_wall(domain.storage.cap,y)
            domain.storage.volume(y)
        def residual(y):
            return -driver.model.force(y,driver.config.start_time_s)-pressure.force(y,pressure_mmhg,target)
        def linearization(y):
            K=tangent.assemble(y,driver.config.start_time_s)
            dp=pressure.linearization(y,pressure_mmhg,target)
            def action(v):
                return -torch.sparse.mm(K,v.reshape(y.numel(),-1)).reshape_as(v)-dp(v)
            diagonal=-tangent.diagonal(K).reshape_as(y)
            if (diagonal<=0).any():raise ValueError('nonpositive static material diagonal')
            cache.update(action=action,diagonal=diagonal)
            return action
        tolerance=max(settings.force_atol_dyn,settings.force_rtol*float(pressure.force(x,pressure_mmhg,target).norm()))
        options=replace(settings.nonlinear,rtol=0.,atol=tolerance)
        try:
            result=newton(residual,x,validate=validate,linearization_factory=linearization,
                preconditioner_factory=lambda y:CoarseInverse(cache['action'],cache['diagonal'],basis),options=options)
        except NonlinearFailure as exc:
            rejected.append(dict(weight=target,error=str(exc),residual=exc.result.residual_norm,
                                 history=exc.result.history))
            step=(target-weight)/2
            if step<1/64:raise
            continue
        error=float(residual(result.x).norm())
        if error>tolerance:raise ValueError('corrected preload true force residual failed')
        x=result.x;weight=target
        entry=dict(tract_load_fraction=weight,residual_norm_dyn=error,tolerance_dyn=tolerance,
            newton_iterations=result.iterations,history=result.history,
            max_correction_cm=float((x-x0).norm(dim=-1).max()))
        history.append(entry)
        if progress:progress(entry)
    return x,dict(status='passed',pressure_mmhg=pressure_mmhg,history=history,rejected=rejected,
        force_residual_norm_dyn=error,force_tolerance_dyn=tolerance,
        ep_ode_clock_advanced=False,active_stress_frozen_zero=True,production_restart=False)
