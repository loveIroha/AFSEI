"""Occasional effective active-stress/work metrics on the existing FE space."""
import torch


@torch.no_grad()
def active_metrics(model,load,x):
    """One scalar transfer to CPU; never called from a nonlinear residual."""
    F=model.element_gradient(x);fiber=model.mesh.fiber
    v=torch.einsum('eij,ej->ei',F,fiber)
    old=torch.einsum('eij,ej->ei',load.previous_F,fiber)
    lam=v.square().sum(-1).sqrt()[:,None]
    ta=load.tension(x);weights=load.geometry.weights
    J=torch.linalg.det(F)[:,None]
    multiplier=1+4.9*(lam.clamp_min(1)-1)
    stress=ta*multiplier*lam.square()/J
    reference=weights.sum();current=(weights*J).sum()
    rate=(v*(v-old)/load.dt).sum(-1)[:,None]
    data=torch.stack(((weights*lam).sum()/reference,(weights*ta).sum()/reference/1e4,
        ta.max()/1e4,(weights*J*stress).sum()/current/1e4,
        -(weights*ta*multiplier*rate).sum()/1e7,
        (weights*(lam<.87)).sum()/reference,J.min(),J.max()))
    return dict(zip(('reference_mean_stretch','reference_mean_ta_kpa','max_ta_kpa',
        'current_mean_active_fiber_cauchy_kpa','active_force_power_w',
        'reference_fraction_stretch_below_087','minimum_J','maximum_J'),data.cpu().tolist()))
