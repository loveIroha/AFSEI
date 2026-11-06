"""Frozen-checkpoint force localization and homogeneous pressure responses.

No EP/ODE, Newton, physical time step, restart publication or parameter edit.
Linear responses are diagnostic probes of the existing fixed MAC operator.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.ep.mac_transport import closed_volume_gradient
from afsi_torch.ep.mac_compact_cap import cap_reaction
from afsi_torch.ep.mac_pressure_audit import cavity_masks
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.sealed_pressure import gradient
from afsi_torch.mac.open_boundary import velocity_laplacian


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);started=perf_counter();initial_hash=checksum(args.checkpoint)
    rows=json.loads((args.checkpoint.parent/'report.json').read_text())['records']
    published=rows[-1];expected=published['cut_momentum_budget']
    d,s,_=restart.load(args.checkpoint,cardiac=True)
    if abs(s.fsi.time*1000-published['time_ms'])>1e-7:raise ValueError('report and checkpoint differ')
    if s.cap_mode!='rcr' or s.wall_pressure_mmhg or s.cap_reference_mmhg:
        raise ValueError('this audit expects the validated zero-extra-wall-load ejection case')
    dt=s.force_load.dt_s;e=d._fsi_stepper(round(dt/d.protocol.dt_ep_s))
    old=s.cap_geometry;x=s.fsi.x
    prepare_cut_flow(e,replace(s.fsi,x=old),s)
    frozen=e.transfer.prepare(old);cap=frozen.rows;load=e.transfer.pressure_load
    average=(closed_volume_gradient(old,load.faces,load.rim)
        +4*closed_volume_gradient((old+x)/2,load.faces,load.rim)
        +closed_volume_gradient(x,load.faces,load.rim))/6
    average+=e.flow.domain.storage.discrete(old,x)
    mode=-C*(average-cap.reaction(frozen.cap_unit_pressure)/C)
    stencil=replace(frozen,mode=mode,dual=mode/mode.square().sum())
    if len(s.cap_pressure)!=cap.count:raise ValueError('frozen cap layout does not match saved reaction')
    f=s.fsi.force;bp=s.cap_pressure
    e.transfer.reset_warm_start()
    density,amplitude_zero,_=e.transfer.relative_parts(f,bp,stencil,rows=cap,check_finite=False)
    density=tuple(v.clone() for v in density)
    alpha=(stencil.dual*f).sum()
    nodal_cap=cap_reaction(cap,bp,check_finite=False)
    def spread(force):
        e.transfer.base.reset_warm_start()
        return e.transfer.base.spread(force,stencil.original)[0]
    sf=spread(f-nodal_cap);sc=spread(nodal_cap);sm=spread(mode)
    lift=cap.lift(bp)
    parts=dict(standard_fe_and_support=sf,
        paired_cap_reaction=tuple(a-b for a,b in zip(sc,lift)),
        pressure_mode_correction=tuple(alpha*(a-b) for a,b in zip(stencil.target,sm)))
    total=tuple(sum(v[c] for v in parts.values()) for c in range(3))
    difference=tuple(a-b for a,b in zip(density,total))
    relative=(sum(v.square().sum() for v in difference)/sum(v.square().sum() for v in density)).sqrt()

    flow=e.flow;domain=flow.domain;axis=domain.axis;grid=e.grid;h=grid.spacing[axis]
    bottom=domain.bottom_index;n=grid.shape[axis];aperture=domain.apertures[1]
    area_cells=aperture.sum();area=domain.metadata['cut_areas_cm2'][1]
    def mean_faces(v):return (v.movedim(axis,0)[bottom:]*aperture).sum((1,2))/area_cells
    def integral(v,start,stop):
        profile=mean_faces(v)[start-bottom:stop-bottom+1]
        weights=torch.full_like(profile,h);weights[0]*=.5;weights[-1]*=.5
        return (profile*weights).sum()/C
    positions=grid.origin[axis]+torch.arange(bottom,n+1,device=x.device,dtype=x.dtype)*h
    support=float(torch.maximum(old[:,axis].max(),x[:,axis].max()))+2.5*h
    split=min(n-1,max(bottom+1,ceil((support-grid.origin[axis])/h)))
    basals=cap.prepared.points[:,axis]
    body_total=float(integral(-density[axis],bottom,n))
    body_error=body_total-expected['terms_mmhg']['ib_and_applied_force']
    if abs(body_error)>1e-6:
        raise ValueError(f'saved forcing reconstruction failed: pressure contribution error {body_error}')
    if float(relative)>1e-7:raise ValueError('standard/cap/mode decomposition does not reconstruct the saved forcing')
    decomposition={name:dict(total_mmhg=float(integral(-v[axis],bottom,n)),
        basal_support_region_mmhg=float(integral(-v[axis],bottom,split)),
        remote_tract_mmhg=float(integral(-v[axis],split,n))) for name,v in parts.items()}
    def face_pressure(k):
        return ((s.fsi.pressure.select(axis,k-1)+s.fsi.pressure.select(axis,k))*.5*aperture).sum()/area_cells/C
    pcut=published['balanced']['prescribed_cut_pressure_mmhg'];pneck=float(face_pressure(bottom));psplit=float(face_pressure(split))
    gp=list(gradient(s.fsi.pressure,grid.spacing,domain.masks))
    for plane in flow.planes:
        sign=-1. if plane.side==0 else 1.
        gp[plane.axis].select(plane.axis,plane.side).add_(sign*2*C/grid.spacing[plane.axis]*pcut*plane.aperture)
    predictor=s.fsi.velocity[axis]+dt/flow.rho*gp[axis]
    viscous=-flow.mu*velocity_laplacian(predictor,axis,grid.spacing)
    qdot=(published['balanced']['physical_outlet_flow_ml_s']-rows[-2]['balanced']['physical_outlet_flow_ml_s'])/dt
    segments={}
    for label,start,stop,pdrop in (('basal_support',bottom,split,pneck-psplit),('remote_tract',split,n,psplit-pcut)):
        acceleration=flow.rho*(stop-start)*h/area*qdot/C
        visc=float(integral(viscous,start,stop));body=float(integral(-density[axis],start,stop))
        segments[label]=dict(length_cm=(stop-start)*h,pressure_drop_mmhg=pdrop,unsteady_mmhg=acceleration,
            viscous_predictor_mmhg=visc,ib_and_applied_force_mmhg=body,
            inferred_transport_plus_predictor_residual_mmhg=pdrop-acceleration-visc-body)
    remote_profile=mean_faces(density[axis])[split-bottom:]
    report=dict(physical_time_advanced=False,checkpoint_sha256=initial_hash,time_ms=s.fsi.time*1000,
        density_relative_reconstruction_error=float(relative),
        saved_ib_pressure_mmhg=expected['terms_mmhg']['ib_and_applied_force'],reconstructed_ib_pressure_mmhg=body_total,
        ib_pressure_reconstruction_error_mmhg=body_error,force_decomposition=decomposition,
        mass_solve_difference_pressure_mmhg=float(integral(-difference[axis],bottom,n)),
        geometry=dict(fixed_neck_cm=domain.metadata['bottom_coordinate_cm'],basal_cap_min_cm=float(basals.min()),
            basal_cap_max_cm=float(basals.max()),conservative_support_bound_cm=support,
            diagnostic_split_cm=grid.origin[axis]+split*h,cut_cm=domain.metadata['cut_coordinate_cm']),
        remote_mean_axial_force_max_dyn_cm3=float(remote_profile.abs().max()),segments=segments,
        pressure_mode_amplitude_mmhg=float(alpha),additional_homogeneous_fluid_responses=0)
    def publish():
        (args.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    publish();print(json.dumps(dict(stage='forcing_reconstructed',**report),indent=2),flush=True)

    # Use one explicitly recomputed endpoint mask, never fit it to published P.
    _,core=cavity_masks(x,e.model.endo_faces,grid,guard_cells=d.pressure_guard_cells)
    current_core=float(s.fsi.pressure[core].mean()/C)
    def pressure_pair(p):
        neck=((p.select(axis,bottom-1)+p.select(axis,bottom))*.5*aperture).sum()/area_cells
        return dict(core_mmhg=float(p[core].mean()/C),neck_mmhg=float(neck/C),
            core_minus_neck_mmhg=float((p[core].mean()-neck)/C))
    pressure_sum=torch.zeros_like(s.fsi.pressure);pressure_components={}
    zero=grid.zeros(device=x.device)
    for name,force in parts.items():
        _,p,_=flow.base.advance(zero,force,None)
        pressure_sum+=p;pressure_components[name]=pressure_pair(p)
    _,combined_pressure,_=flow.base.advance(zero,density,None)
    pressure_relative=float((pressure_sum-combined_pressure).norm()/combined_pressure.norm().clamp_min(1.))
    if pressure_relative>1e-7:raise ValueError('homogeneous pressure superposition failed')
    boundary_pressure=pcut*flow.units[0][1]
    pressure_components['prescribed_cut_value']=pressure_pair(boundary_pressure)
    pressure_components['saved_history_transport_remainder']=pressure_pair(s.fsi.pressure-combined_pressure-boundary_pressure)
    report.update(core_diagnostic=dict(published_core_mmhg=published['balanced']['pressure_core']['mean_mmhg'],
        recomputed_endpoint_core_mmhg=current_core,endpoint_core_cell_count=int(core.sum()),
        published_core_cell_count=published['balanced']['pressure_core']['cell_count'],
        note='Current endpoint mask is recomputed; published driver may reuse an earlier mask. No pressure field or reference is changed.'),
        pressure_components=pressure_components,pressure_superposition_relative_error=pressure_relative,
        additional_homogeneous_fluid_responses=4,
        old_velocity_or_departure_reconstructed=False,
        pressure_remainder_interpretation='Saved pressure minus combined zero-departure force response and prescribed-cut unit response; not an independent recovery of the saved transport history.')
    report['checkpoint_unchanged']=checksum(args.checkpoint)==initial_hash
    report['elapsed_seconds']=perf_counter()-started;report['completed']=True;publish()

    z=positions.cpu().numpy();fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name,v in parts.items():
        profile=mean_faces(-v[axis]).cpu().numpy()/C
        ax[0,0].plot(z,profile,label=name)
    ax[0,0].plot(z,mean_faces(-density[axis]).cpu().numpy()/C,'k--',label='Total saved-equivalent force')
    ax[0,0].set(xlabel='Axial coordinate [cm]',ylabel='Force pressure gradient [mmHg/cm]',title='AV-column force localization')
    pressure_profile=(s.fsi.pressure.movedim(axis,0)[bottom:]*aperture).sum((1,2))/area_cells/C
    ax[0,1].plot(z[:-1]+.5*h,pressure_profile.cpu().numpy(),label='AV-column mean pressure')
    ax[0,1].set(xlabel='Axial coordinate [cm]',ylabel='Pressure [mmHg]',title='Fixed neck to actual pressure cut')
    for a in ax[0]:a.axvline(grid.origin[axis]+split*h,color='gray',ls=':',label='Conservative support split');a.legend(fontsize=7)
    names=list(decomposition)
    ax[1,0].barh(names,[decomposition[k]['total_mmhg'] for k in names])
    ax[1,0].set(xlabel='Axial integrated force contribution [mmHg]',title='Pieces sum to saved IB/body-force contribution')
    names=list(pressure_components)
    ax[1,1].barh(names,[pressure_components[k]['core_minus_neck_mmhg'] for k in names])
    ax[1,1].set(xlabel='Core minus neck [mmHg]',title='Fixed-operator pressure bookkeeping (signed pieces)')
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle(f'Frozen AFSEI checkpoint audit at {s.fsi.time*1000:.1f} ms; no physical advance')
    fig.savefig(args.output/'force_space.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
