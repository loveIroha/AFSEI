"""Frozen-field transport audit; optional ONE unchanged coupled validation step.

Centered/donor derivatives and momentum fluxes are diagnostic comparisons,
not replacement solvers or reference solutions. dt factors apply only to a
frozen advection input and do not constitute trajectory time convergence.
"""
import argparse
from pathlib import Path
from time import perf_counter
import hashlib
import json
import traceback
import numpy as np
import torch
from afsi_torch.mac.grid import MACGrid,slab,divergence
from afsi_torch.mac.open_boundary import sample_face
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.ep.gao_tracts import LVTractDomain
from afsi_torch.circulation.geometry import PortGeometry
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


def derivatives(q,speed,spacing):
    centered=torch.zeros_like(q);donor=torch.zeros_like(q)
    for axis,h in enumerate(spacing):
        before=torch.cat((q[slab(axis,0,1)],q[slab(axis,None,-1)]),axis)
        after=torch.cat((q[slab(axis,1,None)],q[slab(axis,-1,None)]),axis)
        centered+=speed[...,axis]*(after-before)/(2*h)
        donor+=(speed[...,axis].clamp_min(0)*(q-before)
                +speed[...,axis].clamp_max(0)*(after-q))/h
    return centered,donor


@torch.no_grad()
def manufactured_checks(device):
    grid=MACGrid((12,12,12),(1.2,1.2,1.2))
    points=grid.coordinates(2,device=device,dtype=torch.float64)
    speed=torch.zeros_like(points);speed[...,0]=.4
    keep=(points[...,0]>.2)&(points[...,0]<1.)
    errors={}
    for name,q,expected in [('constant',torch.ones_like(points[...,0]),0.),
                            ('linear',2*points[...,0]+3,.8)]:
        value=(q-sample_face(q,points-.025*speed,grid,2))/.025
        errors[name]=float((value[keep]-expected).abs().max())
    q=.5*points[...,0].square();_,donor=derivatives(q,speed,grid.spacing)
    value=(q-sample_face(q,points-.025*speed,grid,2))/.025
    errors['quadratic_sl_matches_exact_donor']=float((value[keep]-donor[keep]).abs().max())
    if max(errors.values())>2e-12:raise ValueError('analytic diagnostic checks failed')
    return errors


def integrate(field,mask,k0,k1,h,rho):
    weights=field.new_full((k1-k0+1,),h);weights[0]*=.5;weights[-1]*=.5
    return float(rho*(field[:,:,k0:k1+1]*mask[:,:,None]*weights).sum()/mask.sum()/C)


def flux_budget(u,mask,k0,k1,grid,rho):
    """Finite-volume boundary flux of axial momentum on a fixed prism.

    Side carried velocity uses centered cell/edge interpolation. This is
    an independently evaluated conservative diagnostic, not exact equality
    to the characteristic/trapezoidal operator on arbitrary rough fields.
    """
    ux,uy,w=u;hx,hy,hz=grid.spacing;area=mask.sum()*hx*hy
    m=mask.to(w.dtype)
    bx=torch.cat((m.new_zeros((1,m.shape[1])),m),0)-torch.cat((m,m.new_zeros((1,m.shape[1]))),0)
    by=torch.cat((m.new_zeros((m.shape[0],1)),m),1)-torch.cat((m,m.new_zeros((m.shape[0],1))),1)
    wc=.5*(w[:,:,:-1]+w[:,:,1:])
    wx=.5*(torch.cat((wc[:1],wc),0)+torch.cat((wc,wc[-1:]),0))
    wy=.5*(torch.cat((wc[:,:1],wc),1)+torch.cat((wc,wc[:,-1:]),1))
    side_x=(bx[:,:,None]*ux[:,:,k0:k1]*wx[:,:,k0:k1]).sum()*hy*hz
    side_y=(by[:,:,None]*uy[:,:,k0:k1]*wy[:,:,k0:k1]).sum()*hx*hz
    axial=((w[:,:,k1].square()-w[:,:,k0].square())*m).sum()*hx*hy
    side_q=(bx[:,:,None]*ux[:,:,k0:k1]).sum()*hy*hz+(by[:,:,None]*uy[:,:,k0:k1]).sum()*hx*hz
    qlow=(w[:,:,k0]*m).sum()*hx*hy;qhigh=(w[:,:,k1]*m).sum()*hx*hy
    div=divergence(u,grid.spacing)
    qdefect=((div[:,:,k0:k1]*m[:,:,None]).sum()*grid.volume)
    return dict(axial_flux_mmhg=float(rho*axial/area/C),
        lateral_flux_mmhg=float(rho*(side_x+side_y)/area/C),
        total_momentum_flux_mmhg=float(rho*(axial+side_x+side_y)/area/C),
        q_low_ml_s=float(qlow),q_high_ml_s=float(qhigh),side_mass_flux_ml_s=float(side_q),
        finite_volume_mass_error_ml_s=float(qhigh-qlow+side_q),
        divergence_integral_ml_s=float(qdefect),
        mass_identity_error_ml_s=float(qhigh-qlow+side_q-qdefect))


@torch.no_grad()
def audit(cfg):
    path=Path(cfg['checkpoint']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    started=perf_counter();checksum=hashlib.sha256(path.read_bytes()).hexdigest()
    device=cfg.get('device','cuda');torch.set_num_threads(4)
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['metadata']))
        take=lambda key:torch.as_tensor(a[key],device=device)
        X=take('X');x=take('x');faces=take('faces');tags=take('facet_tags')
        anchor=take('gao_tract_anchor');u=tuple(take(f'velocity_{c}') for c in range(3))
    cut=meta.get('cut_boundary')
    if not cut or meta['balanced']['mode']!='rcr':raise ValueError('physical-cut ejection checkpoint required')
    data=meta['config']['fsi']['fluid'];grid=MACGrid(tuple(data['shape']),tuple(data['lengths']),tuple(data['origin']))
    cap=MACBasalPorts(X,faces[tags==1],grid,ports=PortGeometry(**meta['balanced_execution']['port_geometry']),
        integration_rule=meta['balanced_execution'].get('cap_integration_rule','centroid'))
    domain=LVTractDomain(grid,cap,anchor,mode='rcr',**cut.get('geometry',{}))
    masked=domain.mask(u);mask_error=max(float((a-b).abs().max()) for a,b in zip(masked,u))
    if mask_error>1e-9:raise ValueError('checkpoint velocity no longer matches its fixed geometry')
    u=masked;w=u[2];dt=meta['previous_dt'];rho=data['rho'];h=grid.spacing
    points=grid.coordinates(2,device=device,dtype=w.dtype)
    speed=torch.stack((sample_face(u[0],points,grid,0),sample_face(u[1],points,grid,1),w),-1)
    center,donor=derivatives(w,speed,h)
    inputs={'centered_advective':center*domain.masks[2],'infinitesimal_donor':donor*domain.masks[2]}
    for factor in cfg.get('dt_factors',[1.,.5,.25,.125]):
        inputs[f'semi_lagrangian_dt_factor_{factor:g}']=(w-sample_face(w,points-factor*dt*speed,grid,2)*domain.masks[2])/(factor*dt)
    support=1+int(torch.ceil((x[:,2].max()+2*h[2]-grid.origin[2])/h[2]))
    if domain.shared_top_index is not None:support=max(support,domain.shared_top_index+1)
    if support>=grid.shape[2]-2:raise ValueError('insufficient free upper duct')
    footprint=domain.neck_apertures[1]
    regions=[('basal_av_neck_prism',footprint,domain.bottom_index,support),
             ('upper_av_duct',domain.apertures[1],support,grid.shape[2])]
    if not domain.metadata['neck_corner_trim_cells']:
        regions.append(('whole_av_prism',domain.apertures[1],domain.bottom_index,grid.shape[2]))
    report=dict(status='passed',scope=__doc__,checkpoint=str(path),checkpoint_sha256=checksum,
        input_time_ms=meta['time']*1000,geometry=domain.metadata,physical_dt_s=dt,
        numerical_models_changed=False,manufactured_checks=manufactured_checks(device),
        masked_velocity_difference_cm_s=mask_error,regions={})
    for name,mask,k0,k1 in regions:
        values={key:integrate(v,mask,k0,k1,h[2],rho) for key,v in inputs.items()}
        report['regions'][name]=dict(z_low_cm=grid.origin[2]+k0*h[2],z_high_cm=grid.origin[2]+k1*h[2],
            aperture_cm2=float(mask.sum()*h[0]*h[1]),transport_integral_mmhg=values,
            flux_diagnostic=flux_budget(u,mask,k0,k1,grid,rho),
            sl_minus_centered_mmhg=values['semi_lagrangian_dt_factor_1']-values['centered_advective'])
    q=float((w[:,:,-1]*domain.apertures[1]).sum()*h[0]*h[1]);area=domain.metadata['cut_areas_cm2'][1]
    report.update(q_out_ml_s=q,mean_outlet_speed_cm_s=q/area,kinematic_viscosity_cm2_s=data['mu']/rho)
    if cfg.get('advance_one_step',False):
        from afsi_torch.ep import mac_balanced_checkpoint as restart
        from afsi_torch.ep.cut_diagnostics import fixed_tract_momentum
        d,s,_=restart.load(path,cardiac=True)
        new,info=d.step(s,target_tick=s.tick+d.maximum_ticks,diagnostics=True)
        e=d._fsi_stepper(round(info['accepted_dt_s']/d.protocol.dt_ep_s));problem=e._last_cut_problem
        flow=e.flow;xx,_,_,_,uu,p,_,_,density,_,_=problem.last_data
        budget_bottom=(domain.shared_top_index+1 if domain.metadata['neck_corner_trim_cells']
                       else domain.bottom_index)
        budget=fixed_tract_momentum(flow,problem.state.velocity,problem.departure,uu,p,density,
            axis=2,bottom=budget_bottom,aperture=domain.apertures[1])
        # The identical current input produced the real next-step departure.
        expected_departure=w-dt*inputs['semi_lagrangian_dt_factor_1']
        report['one_unchanged_coupled_step']=dict(time_ms=new.fsi.time*1000,
            phase=new.phase,accepted_dt_s=info['accepted_dt_s'],
            departure_difference_cm_s=float((expected_departure-problem.departure[2]).abs().max()),
            mass_error_ml_s=info['balanced']['mass_error_ml_s'],
            core_pressure_mmhg=info['balanced']['pressure_core']['mean_mmhg'],momentum=budget)
        original=integrate(inputs['semi_lagrangian_dt_factor_1'],domain.apertures[1],
                           budget_bottom,grid.shape[2],h[2],rho)
        one=report['one_unchanged_coupled_step']
        one['budget_start_coordinate_cm']=grid.origin[2]+budget_bottom*h[2]
        one['transport_independent_difference_mmhg']=original-budget['terms_mmhg']['transport']
        if (one['departure_difference_cm_s']>1e-8 or abs(one['transport_independent_difference_mmhg'])>1e-8
                or abs(budget['closure_error_mmhg'])>1e-6):
            raise ValueError('independent next-step transport/momentum verification failed')
        restart.save(out/'diagnostic_checkpoint.npz',d,new,{'scope':'one unchanged diagnostic step; not production continuation'})
    if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('source checkpoint was modified')
    report['elapsed_seconds']=perf_counter()-started
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('configuration',type=Path)
    cfg=json.loads(p.parse_args().configuration.read_text())
    try:audit(cfg)
    except Exception as exc:
        out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
        (out/'failure.json').write_text(json.dumps(dict(status='failed',failure=str(exc),traceback=traceback.format_exc()),indent=2))
        raise
