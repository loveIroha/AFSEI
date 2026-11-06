"""Read-only momentum identities for the existing BE/projection cut flow.

Viscosity is evaluated on the reconstructed predictor, as in the actual
scheme. No linear solve, transfer, constitutive call or state update occurs.
"""
from math import sqrt
from time import perf_counter
import torch
from ..mac.open_boundary import velocity_laplacian
from .windkessel import MMHG_TO_DYN_CM2 as C


@torch.no_grad()
def fixed_tract_momentum(flow,old_velocity,departure,velocity,pressure,density,
                         *,axis,bottom,aperture):
    grid=flow.grid;h=grid.spacing[axis];dt=flow.dt
    if not 1<=bottom<grid.shape[axis]:raise ValueError('interior fixed neck required')
    low=pressure.narrow(axis,0,1);high=pressure.narrow(axis,grid.shape[axis]-1,1)
    gp=flow.domain.masks[axis]*torch.cat((2*low/h,torch.diff(pressure,dim=axis)/h,-2*high/h),axis)
    cut_pressure=None
    for plane,value in zip(flow.planes,flow.last_boundary['pressures_mmhg']):
        if plane.axis!=axis:continue
        sign=-1. if plane.side==0 else 1.
        gp.select(axis,plane.side).add_(sign*2*C/h*value*plane.aperture)
        if plane.side==-1:cut_pressure=value
    if cut_pressure is None:raise ValueError('positive physical pressure cut required')
    u=velocity[axis];old=old_velocity[axis];dep=departure[axis]
    predictor=u+dt/flow.rho*gp
    lap=velocity_laplacian(predictor,axis,grid.spacing)
    terms=(flow.rho/dt*(u-old),flow.rho/dt*(old-dep),-flow.mu*lap,-density[axis],gp)
    count=aperture.sum()
    def average(value):return (value.movedim(axis,0)[bottom:]*aperture).sum((1,2))/count
    profiles=torch.stack([average(v) for v in (*terms,u,old)])
    weights=pressure.new_full((grid.shape[axis]-bottom+1,),h);weights[0]*=.5;weights[-1]*=.5
    integrals=(profiles[:5]*weights).sum(1)/C
    neck=((pressure.select(axis,bottom-1)+pressure.select(axis,bottom))*.5*aperture).sum()/count/C
    helm=flow.domain.masks[axis]*(predictor-dep-dt/flow.rho*(density[axis]+flow.mu*lap))
    area=count*grid.volume/h
    numbers=torch.cat((integrals,torch.stack((neck,cut_pressure,profiles[5,-1]*area,
        profiles[6,-1]*area,helm.norm(),count))))
    # Exactly one batched transfer, only on configured diagnostic samples.
    values=numbers.detach().cpu().tolist()
    names=('unsteady','transport','viscous_predictor','ib_and_applied_force')
    result=dict(terms_mmhg=dict(zip(names,values[:4])),neck_mmhg=values[5],cut_mmhg=values[6],
        measured_drop_mmhg=values[5]-values[6],predicted_drop_mmhg=sum(values[:4]),
        closure_error_mmhg=sum(values[:4])-(values[5]-values[6]),
        pressure_integral_error_mmhg=-values[4]-(values[5]-values[6]),
        q_new_ml_s=values[7],q_old_ml_s=values[8],predictor_residual_norm_cm_s=values[9],
        interpretation='Fixed collar neck to physical cut; BE predictor viscosity and semi-Lagrangian transport; not an anatomical valve pressure.')
    return result


@torch.no_grad()
def observe_cut_momentum(driver,state,info):
    started=perf_counter()
    if info['cardiac']['solved_mode']!='rcr':return dict(status='not-rcr')
    interval=round(info['accepted_dt_s']/driver.protocol.dt_ep_s)
    endpoint=driver._fsi_stepper(interval);problem=endpoint._last_cut_problem
    flow=endpoint.flow;domain=flow.domain;axis=domain.axis
    aperture=domain.apertures[1]
    x,_,_,_,u,p,flow_info,_,density,_,_=problem.last_data
    bottom=(domain.shared_top_index+1 if domain.metadata.get('neck_corner_trim_cells',0)
            else domain.bottom_index)
    budget=fixed_tract_momentum(flow,problem.state.velocity,problem.departure,u,p,density,
        axis=axis,bottom=bottom,aperture=aperture)
    budget['profile_start_coordinate_cm']=endpoint.grid.origin[axis]+bottom*endpoint.grid.spacing[axis]
    budget['profile_scope']=('fixed upper rectangular duct above trimmed collar' if bottom!=domain.bottom_index
                             else 'fixed rectangle from basal collar neck')
    # Cauchy-Schwarz bound from the ORIGINAL predictor residual tolerance.
    n=endpoint.grid.shape[axis]-bottom+1
    area_cells=domain.metadata['cut_areas_cm2'][1]/(endpoint.grid.volume/endpoint.grid.spacing[axis])
    weight_norm=endpoint.grid.spacing[axis]*sqrt((n-1.5)/area_cells)
    tolerance=flow_info['helmholtz_tolerances'][axis]
    budget['predictor_tolerance_cm_s']=tolerance
    budget['closure_allowance_mmhg']=max(1e-8,10*flow.rho/flow.dt*weight_norm*tolerance/C)
    budget['consistent_with_scheme']=(abs(budget['closure_error_mmhg'])<=budget['closure_allowance_mmhg']
        and abs(budget['pressure_integral_error_mmhg'])<=1e-8
        and budget['predictor_residual_norm_cm_s']<=max(1e-11,1.01*tolerance))
    budget['time_ms']=state.fsi.time*1000
    budget['core_minus_neck_mmhg']=info['balanced']['pressure_core']['mean_mmhg']-budget['neck_mmhg']
    budget['diagnostic_seconds']=perf_counter()-started
    return budget
