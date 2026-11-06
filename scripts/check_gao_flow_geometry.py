"""Frozen-state, matched-outflow sensitivity to idealized AV tract padding.

Diagnostic only. No physical step, EP/ODE or constitutive update. Original
MAC BE/projection and paired cap operators are called unchanged. A scalar
pressure-mode input and closed-cap reactions are adjusted to match the saved
outflow and the original weak relative-flux constraints. This is not a new
coupled physical solution or a change to the production geometry/config.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.ep.mac_transport import closed_volume_gradient
from afsi_torch.ep.mac_pressure_audit import cavity_masks
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.cut_plane import CutPlane, CutPlaneFlow
from afsi_torch.mac.sealed_flow import SealedBEFlow
from afsi_torch.mac.sealed_pressure import gradient, divergence
from afsi_torch.mac.open_boundary import velocity_laplacian


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def widened_domain(original,padding,material_cap):
    """Add whole MAC cells away from MV; retain a rectangular AV tract."""
    d=copy.copy(original);g=d.grid;axis=d.axis
    other=[a for a in range(3) if a!=axis]
    d.bounds=copy.deepcopy(original.bounds)
    centers=[[.5*(lo+hi) for lo,hi in bounds] for bounds in d.bounds]
    separation=[centers[1][j]-centers[0][j] for j in range(2)]
    primary=max(range(2),key=lambda j:abs(separation[j]))
    for j,a in enumerate(other):
        lo,hi=d.bounds[1][j];step=padding*g.spacing[a]
        if j==primary:
            if separation[j]>0:hi+=step
            else:lo-=step
        else:lo-=step;hi+=step
        if lo<g.origin[a]+2*g.spacing[a] or hi>g.origin[a]+g.lengths[a]-2*g.spacing[a]:
            raise ValueError('diagnostic AV padding reaches box support margin')
        d.bounds[1][j]=[lo,hi]
    like=original.masks[0]
    coordinates=[g.origin[a]+g.spacing[a]*(torch.arange(g.shape[a],device=like.device,dtype=like.dtype)+.5)
                 for a in range(3)]
    d.apertures=[];d.label=torch.zeros_like(original.label)
    for label,bounds in enumerate(d.bounds,1):
        aperture=torch.ones(tuple(g.shape[a] for a in other),device=like.device,dtype=torch.bool)
        for j,(a,(lo,hi)) in enumerate(zip(other,bounds)):
            shape=[1,1];shape[j]=len(coordinates[a])
            aperture &= ((coordinates[a]>=lo)&(coordinates[a]<hi)).reshape(shape)
        if any((aperture&q).any() for q in d.apertures):raise ValueError('AV padding overlaps fixed MV tract')
        d.apertures.append(aperture)
        shape=[1]*3
        for a,n in zip(other,aperture.shape):shape[a]=n
        z=torch.arange(g.shape[axis],device=like.device);profile=[1]*3;profile[axis]=len(z)
        region=aperture.reshape(shape).expand(g.shape)&(z>=d.bottom_index).reshape(profile)
        d.label[region]=label
    masks=[torch.ones_like(q) for q in original.masks]
    for c in range(3):
        lo=d.label.narrow(c,0,g.shape[c]-1);hi=d.label.narrow(c,1,g.shape[c]-1)
        if c!=axis:masks[c].narrow(c,1,g.shape[c]-1).masked_fill_((lo!=hi)&((lo>0)|(hi>0)),0.)
    for label,aperture in enumerate(d.apertures,1):
        if label!=2:masks[axis].select(axis,-1).masked_fill_(aperture,0.)
    d.masks=tuple(masks)
    d.cut_planes=(CutPlane(axis,-1,d.apertures[1],'av_cut'),)
    d.metadata=dict(original.metadata,tract_bounds_cm=d.bounds,
        cut_areas_cm2=[float(a.sum())*g.volume/g.spacing[axis] for a in d.apertures],
        diagnostic_padding_cells=padding,diagnostic_padding='AV expands away from MV and on both orthogonal sides')
    d.storage=type(original.storage)(material_cap,d)
    return d


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);start=perf_counter();sha=digest(a.checkpoint)
    last=json.loads((a.checkpoint.parent/'report.json').read_text())['records'][-1]
    driver,state,_=restart.load(a.checkpoint,cardiac=True)
    dt=state.force_load.dt_s;old=state.cap_geometry;x=state.fsi.x
    endpoint=driver._fsi_stepper(round(dt/driver.protocol.dt_ep_s))
    prepare_cut_flow(endpoint,replace(state.fsi,x=old),state)
    original_flow=endpoint.flow;original_domain=original_flow.domain
    cut_pressure=last['balanced']['prescribed_cut_pressure_mmhg']
    target_q=last['balanced']['physical_outlet_flow_ml_s']
    _,core=cavity_masks(x,endpoint.model.endo_faces,endpoint.grid,guard_cells=driver.pressure_guard_cells)
    st=endpoint.transfer.prepare(old);load=endpoint.transfer.pressure_load
    average=(closed_volume_gradient(old,load.faces,load.rim)
        +4*closed_volume_gradient((old+x)/2,load.faces,load.rim)
        +closed_volume_gradient(x,load.faces,load.rim))/6

    def stencil_for(flow):
        endpoint.flow=flow
        frozen=endpoint.transfer.prepare(old)
        mode=-C*(average+flow.domain.storage.discrete(old,x)-frozen.rows.reaction(frozen.cap_unit_pressure)/C)
        return replace(frozen,mode=mode,dual=mode/mode.square().sum())

    # Construct the EXACT finite-displacement mode used by the accepted step.
    mode=-C*(average+original_domain.storage.discrete(old,x)-st.rows.reaction(st.cap_unit_pressure)/C)
    st=replace(st,mode=mode,dual=mode/mode.square().sum())
    bp=state.cap_pressure.clone()
    material_force=state.fsi.force-st.rows.reaction(bp)
    endpoint.transfer.reset_warm_start()
    density,_,_=endpoint.transfer.relative_parts(state.fsi.force,bp,st,rows=st.rows,check_finite=False)
    density=tuple(q.clone() for q in density)
    gp=list(gradient(state.fsi.pressure,endpoint.grid.spacing,original_domain.masks))
    for plane in original_flow.planes:
        gp[plane.axis].select(plane.axis,plane.side).add_(2*C/endpoint.grid.spacing[plane.axis]*cut_pressure*plane.aperture.to(x.dtype))
    predictor=tuple(u+dt/original_flow.rho*g for u,g in zip(state.fsi.velocity,gp))
    # Algebraically reconstructed momentum input, not independently recovered
    # old velocity or a rerun of semi-Lagrangian advection.
    departure=tuple(m*(u-dt/original_flow.rho*(original_flow.mu*velocity_laplacian(u,c,endpoint.grid.spacing)+f))
        for c,(u,f,m) in enumerate(zip(predictor,density,original_domain.masks)))
    original_flow.prescribe([cut_pressure])
    replay_u,replay_p,_=original_flow.advance(departure,density,None)
    replay=dict(velocity_relative_error=float((sum((u-v).square().sum() for u,v in zip(replay_u,state.fsi.velocity))
        /sum(v.square().sum() for v in state.fsi.velocity)).sqrt()),
        pressure_relative_error=float((replay_p-state.fsi.pressure).norm()/state.fsi.pressure.norm()),
        outflow_error_ml_s=float(original_flow.flux(replay_u).sum())-target_q)
    if replay['velocity_relative_error']>1e-7 or replay['pressure_relative_error']>1e-7 or abs(replay['outflow_error_ml_s'])>1e-5:
        raise ValueError(f'accepted-state replay failed: {replay}')
    report=dict(time_ms=state.fsi.time*1000,physical_time_advanced=False,checkpoint_sha256=sha,
        original_operators_unchanged=True,matched_outflow_ml_s=target_q,cut_pressure_mmhg=cut_pressure,
        momentum_input='Reconstructed from the accepted BE momentum identity; not independent old-velocity/advection history. Same input masked onto each diagnostic domain.',
        scope='Frozen-force, pressure-mode-controlled local flow sensitivity, with original weak cap relative-flux constraints. Not a full material-equilibrium solve or new PV trajectory.',
        replay=replay,cases=[])
    def publish():
        tmp=a.output/'report.json.working';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(a.output/'report.json')
    publish();print(json.dumps(dict(stage='accepted_state_replayed',**report)),flush=True)

    for padding in (0,1,2):
        tick=perf_counter()
        if padding==0:flow=original_flow;case_st=st
        else:
            domain=widened_domain(original_domain,padding,endpoint._gao_material_cap)
            domain.validate_wall(endpoint._gao_material_cap,x)
            flow=CutPlaneFlow(SealedBEFlow(endpoint._gao_original_flow,domain,fused=True,graphs=True),domain.cut_planes)
            case_st=stencil_for(flow)
        endpoint.flow=flow;flow.prescribe([cut_pressure]);rows=case_st.rows
        if rows.count!=len(bp):raise ValueError('physical cap partition changed during tract-only test')
        n=rows.count-2;area=flow.domain.metadata['cut_areas_cm2'][1]
        def observe(u):
            endpoint.transfer.reset_warm_start();nodal,_=endpoint.transfer.interpolate(u,case_st,check_finite=False)
            rates=rows.flow(u)-rows.material_flow(nodal)
            return torch.cat((rates[:n]/rows.area[:n],(flow.flux(u).sum()/area).reshape(1))),nodal,rates
        def forcing(delta):
            pressures=bp.clone();pressures[:n]+=C*delta[:n]
            force=material_force+rows.reaction(pressures)+delta[-1]*case_st.mode
            endpoint.transfer.reset_warm_start()
            out,_,_=endpoint.transfer.relative_parts(force,pressures,case_st,rows=rows,check_finite=False)
            return out,pressures
        zeros=bp.new_zeros(n+1);f0,_=forcing(zeros)
        u0,p0,_=flow.advance(departure,f0,None)
        r0,_,_=observe(u0);target=r0.new_zeros(n+1);target[-1]=target_q/area
        rhs=target-r0
        linear_info=dict(columns=0,rank=None,condition_number=None)
        solution=zeros.clone()
        if float(rhs.norm())>1e-8:
            columns=[];delta=zeros.clone()
            print(json.dumps(dict(stage='building_original_fluid_cap_responses',padding_cells=padding,columns=n+1)),flush=True)
            for j in range(n+1):
                delta.zero_();delta[j]=1.
                pressure=bp.new_zeros(rows.count);pressure[:n]=C*delta[:n]
                force=rows.reaction(pressure)+delta[-1]*case_st.mode
                endpoint.transfer.reset_warm_start()
                df,_,_=endpoint.transfer.relative_parts(force,pressure,case_st,rows=rows,check_finite=False)
                du,_p,_i=flow.base.advance(endpoint.grid.zeros(device=x.device),df,None)
                columns.append(observe(du)[0])
                if (j+1)%25==0:print(json.dumps(dict(stage='response_columns',padding_cells=padding,done=j+1,total=n+1)),flush=True)
            matrix=torch.stack(columns,1)
            # Preserve the small response basis so follow-up RHS comparisons
            # can reuse completed fluid solves instead of rebuilding columns.
            torch.save(dict(matrix=matrix.cpu(),rhs=rhs.cpu(),padding_cells=padding,
                cap_pressure_cgs=bp.cpu(),target_q_ml_s=target_q,cut_pressure_mmhg=cut_pressure),
                a.output/f'cap_response_basis_padding_{padding}.pt')
            scale=matrix.norm(dim=0).clamp_min(1e-20);normalized=matrix/scale
            U,sigma,Vh=torch.linalg.svd(normalized,full_matrices=False)
            keep=sigma>sigma[0]*1e-12
            solution=(Vh.T@torch.where(keep,(U.T@rhs)/sigma.clamp_min(1e-300),0.))/scale
            linear_info=dict(columns=n+1,rank=int(keep.sum()),dimension=n+1,
                condition_number=float(sigma[0]/sigma[keep][-1]),
                algebraic_residual_norm=float((matrix@solution-rhs).norm()))
        final_force,final_bp=forcing(solution)
        u,pfield,_=flow.advance(departure,final_force,None)
        residual,nodal,rates=observe(u)
        q=float(flow.flux(u).sum());cap_error=float(rates[:n].abs().max())
        flux_error=abs(q-target_q)
        if flux_error>1e-5 or cap_error>1e-6:
            raise ValueError(f'matched-flow/cap check failed padding={padding}: Q error {flux_error}, cap {cap_error}')
        axis=flow.domain.axis;bottom=flow.domain.bottom_index;ap=flow.domain.apertures[1]
        neck=((pfield.select(axis,bottom-1)+pfield.select(axis,bottom))*.5*ap).sum()/ap.sum()/C
        core_pressure=float(pfield[core].mean()/C)
        physical_bp=final_bp[:-2]/C
        profile=[]
        for k in range(bottom,endpoint.grid.shape[axis]+1):
            face=u[axis].select(axis,k)[ap]
            if k==endpoint.grid.shape[axis]:pk=pfield.new_tensor(cut_pressure)
            else:pk=((pfield.select(axis,k-1)+pfield.select(axis,k))*.5*ap).sum()/ap.sum()/C
            profile.append(dict(z_cm=endpoint.grid.origin[axis]+k*endpoint.grid.spacing[axis],
                pressure_mean_mmhg=float(pk),mean_cm_s=float(face.mean()),max_cm_s=float(face.max()),min_cm_s=float(face.min()),
                momentum_flux_pressure_mmhg=float(flow.rho*face.square().mean()/C)))
        nodal0,_=endpoint.transfer.interpolate(state.fsi.velocity,case_st,check_finite=False)
        case=dict(padding_cells=padding,tract_area_cm2=area,tract_bounds_cm=flow.domain.bounds[1],
            matched_outflow_ml_s=q,outflow_error_ml_s=flux_error,max_closed_cap_relative_flux_ml_s=cap_error,
            pressure_mode_compensation_coefficient=float(solution[-1]),
            closed_cap_pressure_mmhg=physical_bp.cpu().tolist(),
            core_mmhg=core_pressure,neck_mmhg=float(neck),cut_mmhg=cut_pressure,
            core_minus_cut_mmhg=core_pressure-cut_pressure,neck_minus_cut_mmhg=float(neck)-cut_pressure,
            closed_cap_pressure_min_mmhg=float(physical_bp.min()),closed_cap_pressure_max_mmhg=float(physical_bp.max()),
            nodal_velocity_change_relative_norm=float((nodal-nodal0).norm()/nodal0.norm().clamp_min(1e-30)),
            max_divergence_s_inv=float(divergence(u,endpoint.grid.spacing).abs().max()),
            matched_flow_constraint_solve=linear_info,elapsed_seconds=perf_counter()-tick,profile=profile)
        report['cases'].append(case);publish()
        print(json.dumps({k:v for k,v in case.items() if k!='profile'}),flush=True)
    report.update(completed=True,checkpoint_unchanged=digest(a.checkpoint)==sha,elapsed_seconds=perf_counter()-start)
    publish()
    fig,ax=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for case in report['cases']:
        z=[q['z_cm'] for q in case['profile']];name=f"AV padding +{case['padding_cells']} cells; A={case['tract_area_cm2']:.3f} cm2"
        ax[0].plot(z,[q['pressure_mean_mmhg'] for q in case['profile']],label=name)
        ax[1].plot(z,[q['max_cm_s'] for q in case['profile']],label=name)
    ax[0].set(ylabel='Mean static pressure [mmHg]',title='Pressure at matched outflow')
    ax[1].set(ylabel='Max axial velocity [cm/s]',title='Jet profile at matched outflow')
    for a0 in ax:a0.set_xlabel('Axial coordinate [cm]');a0.grid(alpha=.2);a0.legend(fontsize=8)
    fig.suptitle('Frozen-state diagnostic: same AV opening, Qout and cut pressure; no physical advance')
    fig.savefig(a.output/'flow_geometry.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(completed=True,elapsed_seconds=report['elapsed_seconds'],checkpoint_unchanged=report['checkpoint_unchanged'])),flush=True)


if __name__=='__main__':main()
