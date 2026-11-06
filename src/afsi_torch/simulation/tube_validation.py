"""Output-only metrics for the single-layer propagation/contraction test.

The activation regression excludes stimulated nodes and both end regions.
Unactivated sections remain missing, never zero activation time.
"""
import numpy as np


def propagation_metrics(reference, activation, axial_start, length, stimulus_width):
    z=reference[:,2]-axial_start
    sections=[]
    for position in np.unique(z):
        values=activation[z == position]
        finite=np.isfinite(values)
        sections.append(dict(z_cm=float(position), activated_fraction=float(finite.mean()),
            activation_mean_ms=float(values.mean()*1000) if finite.all() else None,
            activation_spread_ms=float(np.ptp(values)*1000) if finite.all() else None))
    fit=[r for r in sections if r['activation_mean_ms'] is not None
         and max(stimulus_width,.2*length) <= r['z_cm'] <= .8*length]
    speed=r2=None
    if len(fit)>=3:
        x=np.array([r['z_cm'] for r in fit]);y=np.array([r['activation_mean_ms'] for r in fit])/1000
        slope,intercept=np.polyfit(x,y,1)
        if slope>0:
            speed=float(1/slope)
            denom=float(((y-y.mean())**2).sum())
            r2=float(1-((y-(slope*x+intercept))**2).sum()/denom) if denom>0 else None
    finite=np.isfinite(activation)
    return dict(activation_threshold_mv=-20.,activated_fraction=float(finite.mean()),
        first_activation_ms=float(activation[finite].min()*1000) if finite.any() else None,
        last_activation_ms=float(activation[finite].max()*1000) if finite.any() else None,
        axial_conduction_speed_cm_s=speed,activation_fit_r_squared=r2,
        fitted_sections=len(fit),sections=sections)


def contraction_metrics(reference,x,inner_rings,center):
    def areas(points):
        p=points[inner_rings,:2];p=p-p.mean(1,keepdims=True)
        q=np.roll(p,-1,axis=1)
        return .5*(p[...,0]*q[...,1]-p[...,1]*q[...,0]).sum(1)
    a0=areas(reference);a=areas(x)
    radial=reference[:,:2]-np.asarray(center)
    radial=radial/np.linalg.norm(radial,axis=1,keepdims=True)
    inward=-((x-reference)[:,:2]*radial).sum(1)
    return dict(minimum_area_ratio=float((a/a0).min()),
        maximum_section_area_reduction_percent=float(100*(1-(a/a0).min())),
        mean_section_area_reduction_percent=float(100*(1-(a/a0).mean())),
        maximum_inward_displacement_cm=float(inward.max()),
        maximum_displacement_cm=float(np.linalg.norm(x-reference,axis=1).max()),
        section_area_ratio=(a/a0).tolist())
