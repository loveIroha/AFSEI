"""Frozen original-MAC outlet/RCR boundary validation; not an EP/FSI run.

The supplied constant pressure drive is an explicit algebraic test load,
not Land tension or a replacement myocardial activation waveform.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_ports import MACBasalPorts
from afsi_torch.ep.mac_rcr_boundary import MACRCRBoundaryResponse
from afsi_torch.ep.mac_mass_budget import peskin_trace
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C
from afsi_torch.mac.spectral_dirichlet import SpectralDirichletPressure


def synchronize(): torch.cuda.synchronize()


def pressure_drive(solver,mmhg):
    pressure=solver.like.new_zeros(solver.count); pressure[-1]=-mmhg*C
    return solver.boundary_density(pressure)


def export_cap(path,solver,result):
    import meshio
    cap=solver.prepared.cap; x=solver.geometry
    edges=x[cap.rim]; center=x[cap.rim[:,0]].mean(0)
    fan=torch.cat((edges,center[None,None].expand(len(edges),1,3)),1)
    triangles=torch.einsum('qka,eai->eqki',cap.subtriangles,fan).reshape(-1,3,3)
    vertices=triangles.reshape(-1,3).cpu().numpy()
    cells=torch.arange(len(vertices)).reshape(-1,3).numpy()
    p=solver.like.new_full((len(cap.labels),),result.outlet_pressure_cgs.item()/C)
    p[solver.closed.selected]=result.cap_pressure_cgs[solver.closed.patches]/C
    relative=peskin_trace(solver.grid,result.velocity,solver.prepared.points)-cap.material_points(result.nodal_velocity)
    flux=(relative*solver.prepared.area_vectors).sum(-1)
    area=solver.prepared.area_vectors.norm(dim=-1)
    meshio.write(path,meshio.Mesh(vertices,[('triangle',cells)],cell_data={
        'region':[cap.labels.cpu().numpy()], 'pressure_mmhg':[p.cpu().numpy()],
        'relative_flux_ml_s':[flux.cpu().numpy()],
        'normal_relative_velocity_cm_s':[(flux/area).cpu().numpy()]}))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--drive-mmhg',type=float,default=90.)
    p.add_argument('--initial-pc-mmhg',type=float,default=85.)
    p.add_argument('--write-vtk',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): p.error('actual CUDA required')
    if not 1<=a.steps<=10000 or not np.isfinite([a.drive_mmhg,a.initial_pc_mmhg]).all():
        p.error('1..10000 steps and finite test pressures required')
    a.output.mkdir(parents=True,exist_ok=False)
    sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    driver,state,_=checkpoint.load(a.checkpoint)
    dt=state.force_load.dt_s
    endpoint=driver._fsi_stepper(round(dt/driver.protocol.dt_ep_s))
    # Same original Dirichlet pressure matrix, existing accelerated backend.
    if not isinstance(endpoint.flow.pressure_solver,SpectralDirichletPressure):
        endpoint.flow.pressure_solver=SpectralDirichletPressure(endpoint.grid,device=state.fsi.x.device,
            options=endpoint.config.pressure_solver)
    geometry=state.fsi.x
    synchronize(); started=perf_counter()
    stencil=endpoint.transfer.prepare(geometry)
    cap=MACBasalPorts(driver.mesh.X,driver.model.endo_faces,endpoint.grid)
    prepared=cap.prepare(geometry)
    solver=MACRCRBoundaryResponse(endpoint.flow,endpoint.transfer,stencil,prepared,geometry=geometry)
    synchronize(); setup=perf_counter()-started
    zero=endpoint.grid.zeros(device=geometry.device)
    # Independent zero-departure probes cover both complementarity branches.
    probes={}
    for name,drive in (('open_probe',a.initial_pc_mmhg+5.),('closed_probe',a.initial_pc_mmhg-5.)):
        r=solver.advance(zero,pressure_drive(solver,drive),a.initial_pc_mmhg)
        probes[name]=dict(r.report,test_drive_mmhg=drive)
    if not probes['open_probe']['outlet_open'] or probes['closed_probe']['outlet_open']:
        raise AssertionError('independent pressure-drive probes did not cover open/closed branches')
    source=pressure_drive(solver,a.drive_mmhg)
    velocity=zero; pc=a.initial_pc_mmhg; rows=[]; result=None
    started=perf_counter()
    try:
        for index in range(1,a.steps+1):
            synchronize(); tick_start=perf_counter()
            departure=endpoint.flow.advect(velocity)
            trial=solver.advance(departure,source,pc)
            # Publish only a fully checked BOUNDARY trial, never EP/solid state.
            velocity,pc,result=trial.velocity,trial.pc_mmhg,trial
            synchronize(); elapsed=perf_counter()-tick_start
            rows.append(dict(trial.report,step=index,validation_time_s=index*dt,wall_seconds=elapsed))
            if index%5==0 or index==a.steps:
                print(f'Boundary step={index}, t={index*dt*1000:.3f}ms, '
                    f'Q={trial.report["measured_relative_q_ml_s"]:.6g}ml/s, '
                    f'Pc={trial.report["pc_mmhg"]:.6g}mmHg, '
                    f'open={trial.report["outlet_open"]}, wall={elapsed:.3f}s',flush=True)
    finally:
        elapsed=perf_counter()-started
        if sha!=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest():
            raise AssertionError('source EP/FSI checkpoint changed')
        all_rows=list(probes.values())+rows
        report=dict(status='completed' if len(rows)==a.steps else 'incomplete',scope=__doc__,
            accepted_steps=len(rows),dt_s=dt,validation_duration_s=len(rows)*dt,
            source_checkpoint=a.checkpoint,source_sha256=sha,source_checkpoint_unchanged=True,
            source_fsi_time_s_unchanged=state.fsi.time,gpu=torch.cuda.get_device_name(),grid_shape=endpoint.grid.shape,
            test_drive_mmhg=a.drive_mmhg,initial_pc_mmhg=a.initial_pc_mmhg,
            geometry=cap.metadata,setup_seconds=setup,schur_setup_seconds=solver.setup_seconds,
            schur_setup_fluid_responses=solver.setup_fluid_responses,
            dense_boundary_matrix_shape=list(solver.mobility.shape),
            elapsed_boundary_loop_seconds=elapsed,
            mean_boundary_seconds_per_step=sum(r['wall_seconds'] for r in rows)/len(rows) if rows else None,
            first_boundary_seconds=rows[0]['wall_seconds'] if rows else None,
            mean_boundary_seconds_after_first=sum(r['wall_seconds'] for r in rows[1:])/len(rows[1:]) if len(rows)>1 else None,
            timing_scope='CUDA synchronized boundary response; includes convection and two original BE solves; setup and output excluded; no EP/solid solve',
            max_closed_flux_norm_ml_s=max(r['closed_region_flux_norm_ml_s'] for r in all_rows),
            max_open_pressure_law_error_mmhg=max(r['open_pressure_law_error_mmhg'] for r in all_rows),
            max_rcr_balance_ml_s=max(abs(r['rcr_balance_ml_s']) for r in all_rows),
            max_relative_work_relative_error=max(r['relative_work_relative_error'] for r in all_rows),
            max_rcr_energy_relative_error=max(r['rcr_energy_relative_error'] for r in all_rows),
            probes=probes,rows=rows,frozen_geometry=True,
            operator_material_velocity='Ju; no physical nodal geometry or history advanced',
            fluid_response_advanced=bool(rows),flow_driven_rcr_in_subproblem=bool(rows),
            ep_advanced=False,myocardium_motion_advanced=False,active_tension_waveform_replaced=False,
            pointwise_impermeability_established=False,windkessel_ports_connected_to_afsei_driver=False,
            full_cardiac_cycle_implemented=False,three_field_integration_implemented=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if result is not None:
            # Separate validation artifact: not an AFSEI restart checkpoint.
            metadata=dict(producer='afsei-mac-rcr-frozen-validation',source_sha256=sha,
                dt_s=dt,steps=len(rows),source_fsi_time_s=state.fsi.time)
            arrays={f'velocity_{c}':u.cpu().numpy() for c,u in enumerate(result.velocity)}
            arrays.update(pressure=result.pressure.cpu().numpy(),nodal_velocity=result.nodal_velocity.cpu().numpy(),
                pc_mmhg=result.pc_mmhg.cpu().numpy(),cap_pressure_cgs=result.cap_pressure_cgs.cpu().numpy(),
                outlet_pressure_cgs=result.outlet_pressure_cgs.cpu().numpy())
            np.savez_compressed(a.output/'boundary_state.npz',metadata=json.dumps(metadata),**arrays)
            if a.write_vtk: export_cap(a.output/'basal_boundary.vtu',solver,result)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','probes','geometry','scope')},indent=2),flush=True)


if __name__=='__main__': main()
