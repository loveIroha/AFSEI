"""Additional response basis for completed matched-flow geometry probes.

No cap Schur columns are repeated. Two new fluid responses per widened
domain distinguish keeping the original Eulerian forcing from changing its
existing pressure-mode transfer. Uncontrolled responses are explicitly not
admissible physical coupled solutions.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from check_gao_flow_geometry import digest,widened_domain
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.ep.mac_transport import closed_volume_gradient
from afsi_torch.ep.mac_pressure_audit import cavity_masks
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.cut_plane import CutPlaneFlow
from afsi_torch.mac.sealed_flow import SealedBEFlow
from afsi_torch.mac.sealed_pressure import gradient
from afsi_torch.mac.open_boundary import velocity_laplacian


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('results',type=Path)
    a=p.parse_args();target=a.results/'force_basis.json'
    if target.exists():raise FileExistsError(target)
    matched=json.loads((a.results/'report.json').read_text());assert matched['completed']
    torch.set_num_threads(4);tick=perf_counter();sha=digest(a.checkpoint)
    assert sha==matched['checkpoint_sha256']
    driver,s,_=restart.load(a.checkpoint,cardiac=True)
    old=s.cap_geometry;x=s.fsi.x;dt=s.force_load.dt_s
    e=driver._fsi_stepper(round(dt/driver.protocol.dt_ep_s))
    prepare_cut_flow(e,replace(s.fsi,x=old),s)
    original_flow=e.flow;original_domain=e.flow.domain
    bp=s.cap_pressure;pcut=matched['cut_pressure_mmhg'];grid=e.grid
    _,core=cavity_masks(x,e.model.endo_faces,grid,guard_cells=driver.pressure_guard_cells)
    st=e.transfer.prepare(old);load=e.transfer.pressure_load
    average=(closed_volume_gradient(old,load.faces,load.rim)+4*closed_volume_gradient((old+x)/2,load.faces,load.rim)
        +closed_volume_gradient(x,load.faces,load.rim))/6
    def complete(frozen,flow):
        mode=-C*(average+flow.domain.storage.discrete(old,x)-frozen.rows.reaction(frozen.cap_unit_pressure)/C)
        return replace(frozen,mode=mode,dual=mode/mode.square().sum())
    st=complete(st,original_flow)
    e.transfer.reset_warm_start();density,_,_=e.transfer.relative_parts(s.fsi.force,bp,st,rows=st.rows,check_finite=False)
    density=tuple(v.clone() for v in density)
    grad=list(gradient(s.fsi.pressure,grid.spacing,original_domain.masks))
    for plane in original_flow.planes:
        grad[plane.axis].select(plane.axis,plane.side).add_(2*C/grid.spacing[plane.axis]*pcut*plane.aperture.to(x.dtype))
    star=tuple(u+dt/original_flow.rho*g for u,g in zip(s.fsi.velocity,grad))
    departure=tuple(m*(v-dt/original_flow.rho*(original_flow.mu*velocity_laplacian(v,c,grid.spacing)+f))
        for c,(v,f,m) in enumerate(zip(star,density,original_domain.masks)))
    original_reaction=st.rows.reaction(bp)
    report=dict(physical_time_advanced=False,checkpoint_sha256=sha,
        additional_fluid_responses=0,cap_schur_columns_repeated=0,
        interpretation='Signed frozen-operator response bookkeeping, not a prediction of a material-equilibrated geometry change. Raw responses do not enforce matched flow or weak cap constraints.',
        cases=[])
    def summary(u,p,flow,frozen):
        e.transfer.reset_warm_start();nodal,_=e.transfer.interpolate(u,frozen,check_finite=False)
        rate=frozen.rows.flow(u)-frozen.rows.material_flow(nodal)
        return dict(core_mmhg=float(p[core].mean()/C),outflow_ml_s=float(flow.flux(u).sum()),
            max_closed_cap_relative_flux_ml_s=float(rate[:-2].abs().max()))
    for padding in (1,2):
        domain=widened_domain(original_domain,padding,e._gao_material_cap)
        flow=CutPlaneFlow(SealedBEFlow(e._gao_original_flow,domain,fused=True,graphs=True),domain.cut_planes)
        e.flow=flow;flow.prescribe([pcut]);frozen=complete(e.transfer.prepare(old),flow)
        reaction=frozen.rows.reaction(bp)
        reaction_error=float((reaction-original_reaction).norm()/original_reaction.norm())
        if reaction_error>1e-12:raise ValueError('tract-only change altered material cap reaction')
        e.transfer.reset_warm_start();new_density,_,_=e.transfer.relative_parts(s.fsi.force,bp,frozen,rows=frozen.rows,check_finite=False)
        change=tuple(v-w for v,w in zip(new_density,density))
        # Two new response components; no new complete matched-flow solve.
        u_geometry,p_geometry,_=flow.advance(departure,density,None)
        du,dp,_=flow.base.advance(grid.zeros(device=x.device),change,None)
        before_control=summary(tuple(v+w for v,w in zip(u_geometry,du)),p_geometry+dp,flow,frozen)
        geometry_only=summary(u_geometry,p_geometry,flow,frozen)
        final=next(q for q in matched['cases'] if q['padding_cells']==padding)
        baseline=matched['cases'][0]
        axis=domain.axis;bottom=domain.bottom_index;ap=domain.apertures[1];h=grid.spacing[axis]
        def body(v):
            line=(v[axis].movedim(axis,0)[bottom:]*ap).sum((1,2))/ap.sum()
            weights=torch.full_like(line,h);weights[0]*=.5;weights[-1]*=.5
            return float(-(line*weights).sum()/C)
        item=dict(padding_cells=padding,cap_reaction_relative_change=reaction_error,
            pressure_mode_relative_vector_change=float((frozen.mode-st.mode).norm()/st.mode.norm()),
            pressure_mode_direction_cosine=float((frozen.mode*st.mode).sum()/(frozen.mode.norm()*st.mode.norm())),
            saved_force_projection_original=float((st.dual*s.fsi.force).sum()),
            saved_force_projection_widened=float((frozen.dual*s.fsi.force).sum()),
            original_density_in_widened_column_mmhg=body(density),
            updated_density_in_widened_column_mmhg=body(new_density),
            force_change_in_widened_column_mmhg=body(change),
            original_eulerian_forcing_on_widened_domain=geometry_only,
            updated_transfer_before_cap_and_flow_control=before_control,
            matched_case_core_mmhg=final['core_mmhg'],
            signed_core_pressure_change_mmhg=dict(
                geometry_operator_only=geometry_only['core_mmhg']-baseline['core_mmhg'],
                existing_pressure_mode_transfer=before_control['core_mmhg']-geometry_only['core_mmhg'],
                cap_and_matched_flow_control=final['core_mmhg']-before_control['core_mmhg']),
            causal_caveat='The density change is from the existing pressure-mode transfer: the material force, physical cap reaction and base FE/IB geometry are unchanged. The compensation coefficient is not itself a prescribed wall or cavity pressure.')
        report['cases'].append(item);report['additional_fluid_responses']+=2
        target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(item),flush=True)
    report.update(completed=True,checkpoint_unchanged=digest(a.checkpoint)==sha,elapsed_seconds=perf_counter()-tick)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    labels=['Original','Padding +1','Padding +2'];cases=matched['cases']
    for key,label in (('core_minus_cut_mmhg','Core - cut'),('neck_minus_cut_mmhg','Neck - cut')):
        ax[0].plot(labels,[q[key] for q in cases],'o-',label=label)
    ax[0].set(ylabel='Pressure difference [mmHg]',title='Matched-flow frozen-state response');ax[0].legend()
    bottom=[0.,0.]
    for key,label in (('geometry_operator_only','Geometry / original Eulerian input'),
        ('existing_pressure_mode_transfer','Existing pressure-mode transfer'),('cap_and_matched_flow_control','Cap + matched-flow adjustment')):
        vals=[q['signed_core_pressure_change_mmhg'][key] for q in report['cases']]
        ax[1].plot(['Padding +1','Padding +2'],vals,'o-',label=label)
    ax[1].axhline(0,color='gray',lw=.8);ax[1].set(ylabel='Signed core-pressure change [mmHg]',title='Additional response basis (not physical steps)');ax[1].legend(fontsize=8)
    for axis in ax:axis.grid(alpha=.2)
    fig.suptitle('Same Qout and cut pressure; diagnostic motion is not held fixed')
    fig.savefig(a.results/'flow_geometry_summary.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(completed=True,additional_fluid_responses=4,elapsed_seconds=report['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
