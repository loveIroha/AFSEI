"""Audit whole LV surface mass and quadrature; never advance or edit a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_mass_budget import ClosedCavityMassAudit,MassAuditTargets


@torch.no_grad()
def diagnose(path,levels,targets,output,write_vtk):
    sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    driver,state,_=checkpoint.load(path)
    ticks=round(state.force_load.dt_s/driver.protocol.dt_ep_s)
    endpoint=driver._fsi_stepper(ticks); grid=endpoint.grid
    audit=ClosedCavityMassAudit(driver.mesh.X,driver.model.endo_faces,grid)
    torch.cuda.synchronize(); started=perf_counter()
    report,fields=audit.evaluate(state.force_load.previous_x,state.fsi.x,state.fsi.velocity,
        dt=state.force_load.dt_s,levels=levels,targets=targets,return_fields=True)
    # The original IB method enforces a volume L2 projection, not equality
    # between raw surface Peskin samples and P1 boundary traces. Audit both.
    endpoint.transfer.reset_warm_start()
    stencil=endpoint.transfer.prepare(state.force_load.previous_x)
    projected,_=endpoint.transfer.interpolate(state.fsi.velocity,stencil)
    w=(state.fsi.x-state.force_load.previous_x)/state.force_load.dt_s
    mismatch=projected-w
    report.update(original_ib_motion_residual_l2_cm=(state.force_load.dt_s*mismatch.norm()).item(),
        original_ib_velocity_difference_max_cm_s=mismatch.abs().max().item(),
        original_ib_projection_checked=True)
    torch.cuda.synchronize(); elapsed=perf_counter()-started
    if write_vtk:
        import meshio
        x=state.fsi.x
        material=(x-state.force_load.previous_x)/state.force_load.dt_s
        for name,vertices,faces in (
            ('wall',x,driver.model.endo_faces[:,[0,2,1]]),
            ('cap',torch.cat((x,x[audit.rim[:,0]].mean(0)[None])),
             torch.cat((audit.rim,torch.full((len(audit.rim),1),len(x),device=x.device,dtype=torch.int64)),1))):
            f=fields[name]
            meshio.write(output/(name+'_mass_budget.vtu'),meshio.Mesh(vertices.cpu().numpy(),
                [('triangle',faces.cpu().numpy())],
                cell_data={key:[value.cpu().numpy()] for key,value in f.items()
                           if key in ('relative_flux','absolute_relative_flux','area')}
                    | {'mean_normal_slip_cm_s':[(f['relative_flux']/f['area']).cpu().numpy()]},
                point_data={'material_velocity_cm_s':material.cpu().numpy()} if name=='wall' else {}))
    assert sha==hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if report['swept_geometry_identity_error_ml_s']>1e-7:
        raise AssertionError('swept material flux and closed volume difference disagree')
    if max(r['flux_decomposition_error_ml_s'] for r in report['levels'])>1e-8:
        raise AssertionError('fluid/material/relative flux decomposition failed')
    report.update(checkpoint=str(path),source_sha256=sha,source_checkpoint_unchanged=True,
        time_s=state.fsi.time,step=state.fsi.step,grid_shape=grid.shape,
        gpu=torch.cuda.get_device_name(),audit_seconds=elapsed,
        closed_cap_mode=getattr(driver,'closed_cap',False),
        cap_patch_width_cells=getattr(driver,'cap_patch_width_cells',None),
        circulation_phase_not_advanced=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',nargs='+',required=True)
    p.add_argument('--levels',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--signed-mass-tolerance',type=float,default=.01)
    p.add_argument('--absolute-leak-tolerance',type=float,default=.1)
    p.add_argument('--quadrature-tolerance',type=float,default=.001)
    p.add_argument('--write-vtk',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): p.error('actual CUDA required')
    if len(a.levels)<2 or sorted(set(a.levels))!=a.levels or any(k<0 or k>4 for k in a.levels):
        p.error('at least two increasing distinct levels in 0..4 required')
    targets=MassAuditTargets(a.signed_mass_tolerance,a.absolute_leak_tolerance,a.quadrature_tolerance)
    a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for i,path in enumerate(a.checkpoint):
        output=a.output/f'snapshot_{i}'; output.mkdir()
        report=diagnose(path,tuple(a.levels),targets,output,a.write_vtk)
        last=report['levels'][-1]
        print(json.dumps(dict(checkpoint=path,wall_signed_ml_s=last['wall']['relative_flux_ml_s'],
            wall_absolute_ml_s=last['wall']['absolute_relative_flux_ml_s'],
            cap_signed_ml_s=last['cap']['relative_flux_ml_s'],cap_absolute_ml_s=last['cap']['absolute_relative_flux_ml_s'],
            closed_signed_ml_s=last['closed_relative_flux_ml_s'],
            rt0_closed_fluid_flux_ml_s=last['rt0_closed_fluid_flux_ml_s'],
            quadrature_passed=report['peskin_quadrature_converged_to_targets'],
            audit_passed=report['surface_mass_audit_passed'],seconds=report['audit_seconds']),indent=2),flush=True)
        reports.append(report)
        torch.cuda.empty_cache()
    # A valid audit can establish that the supplied state FAILS a mass target.
    # "passed" below describes execution/identities, not sealing or physiology.
    result=dict(passed=True,scope=__doc__,snapshots=reports,
        all_surface_mass_audits_passed=all(r['surface_mass_audit_passed'] for r in reports),
        windkessel_actual_port_flow_connected=False,three_field_integration_implemented=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
