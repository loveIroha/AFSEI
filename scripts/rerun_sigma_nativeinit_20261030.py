"""Rerun only the requested 3D standard-CG/nodal pair.

Reuses the existing controlled benchmark and its shared TorchCor CG kernel.
Changes only effective conductivity and the initial conditioning policy.
"""
from pathlib import Path
import hashlib,json,sys,time,os
import numpy as np,torch,meshio
root=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))
from scripts import benchmark_ep_controlled as core
from afsi_torch.config import load_config
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.ep.cells import GPBCell
from validation.torchcor_pde import load_core,COMMIT

@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    out=root/'results/ep_opencarp_sigma_nativeinit_20261030';out.mkdir(exist_ok=False)
    demo=root/'demo/ep_opencarp_sigma_nativeinit';demo.mkdir(exist_ok=True)
    source=root/'results/ep_controlled_h0p125_20261029/original_config.json'
    old=json.loads(source.read_text());cfg=json.loads(source.read_text())
    # The old AFSEI benchmark has a normalized Sv. Preserve it as requested.
    # Match sigma/(Sv*Cm), not the raw conductivity number in another unit system.
    native_sigma_s_m=np.array([.17*.62/(.17+.62),.019*.24/(.019+.24),.019*.24/(.019+.24)])
    native_sigma_us_cm=native_sigma_s_m*10000;native_sv_cm=1400.
    sigma=native_sigma_us_cm*cfg['parameters']['surface_to_volume_per_cm']/native_sv_cm
    cfg['parameters']['conductivity_us_per_cm']=sigma.tolist()
    cfg['initial']['conditioning_s']=0.
    cfg['case_name']='3d_opencarp_diffusion_native_gpb_initial'
    # Correct the old metadata: the controlled pair executes shared TorchCor CG, not AmgX.
    cfg['ep_numerics'].update(backend='torch_pcg_jacobi',time_scheme='crank_nicolson')
    cfg['reference'].update(comparison_scope='Only conductivity and initial conditioning updated; other original 3D settings retained',
        parameter_source='openCARP effective monodomain sigma/(Sv*Cm), preserving AFSEI normalized Sv',
        stimulus_source=old['reference']['stimulus_source'])
    path=demo/'config.json';path.write_text(json.dumps(cfg,indent=2));(out/'original_config.json').write_text(json.dumps(cfg,indent=2))
    (out/'baseline_original_config.json').write_bytes(source.read_bytes())
    typed=load_config(path,EPTissueConfig)
    c=dict(lengths_cm=list(typed.domain.lengths_cm),spacing_cm=typed.domain.spacing_cm,
        dt_ep_s=typed.time.dt_ep_s,dt_ode_ms=typed.time.dt_ode_ms,duration_s=typed.time.end_time_s,
        parameters=typed.parameters.__dict__,stimulus_extent_cm=typed.stimulus.maximum_cm[0],
        stimulus_ua_cm2=typed.stimulus.amplitude_ua_per_cm2,stimulus_duration_s=typed.stimulus.duration_s,conditioning_s=0.)
    (out/'config.json').write_text(json.dumps(c,indent=2))
    classes,provenance=load_core(root/'.deps'/('torchcor-'+COMMIT))
    cell=GPBCell(typed.time.dt_ep_s,typed.time.dt_ode_ms);scalar=cell.initial(torch.zeros(1,device='cuda'))
    initialization=dict(policy='Raw AFSEI GPB native state assigned identically to every node; no conditioning',
        conditioning_s=0.,native_voltage_mv=float(scalar.voltage_v[0])*1000,
        state_values=scalar.gpb[0].cpu().tolist(),same_initial_state_for_both_methods=True)
    (out/'initialization.json').write_text(json.dumps(initialization,indent=2))
    np.savez(out/'raw_initial_state.npz',voltage_v=scalar.voltage_v.cpu().numpy(),gpb=scalar.gpb.cpu().numpy())
    alignment=dict(opencarp_result='results/opencarp_gpb_h0p0625_20261030',opencarp_rerun=False,
        native_sigma_us_per_cm=native_sigma_us_cm.tolist(),native_sv_per_cm=native_sv_cm,
        native_effective_diffusion_cm2_s=(native_sigma_us_cm/native_sv_cm).tolist(),
        afsei_normalized_sv_per_cm=typed.parameters.surface_to_volume_per_cm,
        afsei_sigma_us_per_cm=sigma.tolist(),
        afsei_effective_diffusion_cm2_s=(sigma/typed.parameters.surface_to_volume_per_cm).tolist(),
        equation_note='Both equations are identical after division by Sv*Cm; no other AFSEI physical parameter changed.',
        preserved=dict(mesh_spacing_mm=.125,nodes=229425,cells=1290240,dt_ep_ms=.05,dt_ode_ms=.002,
            duration_ms=120,cell='production AFSEI GPB',ode='SDIRK',splitting='original frozen-V source into CN',
            stimulus_uA_cm2=typed.stimulus.amplitude_ua_per_cm2),
        execution=dict(actual_solver_pair='shared pinned TorchCor Jacobi-CG and PDEMethod wrapper',
            amgx_used=False,physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name()),
        remaining_differences_vs_opencarp=['mesh size/connectivity','EP time step','GPB implementation/initial state',
            'ODE integrator and reaction/diffusion organization','duration','CPU versus GPU','timing scopes'])
    (out/'alignment.json').write_text(json.dumps(alignment,indent=2))
    started=time.perf_counter();print(json.dumps(dict(status='starting',output=str(out),initial_voltage_mv=initialization['native_voltage_mv'],alignment=alignment)),flush=True)
    try:
        row=core.run_case(3,c,typed,scalar,classes,out,primary_only=True)
        audits={}
        for method in ('consistent_tc','nodal_tc'):
            folder=out/'3d'/method
            m=meshio.read(folder/'ep_00000.vtu');v=m.point_data['voltage_mV'];activation=m.point_data['activation_ms']
            audits[method]=dict(nodes=len(v),initial_voltage_min_mv=float(v.min()),initial_voltage_max_mv=float(v.max()),
                all_nodes_have_native_initial_voltage=bool(np.all(v==initialization['native_voltage_mv'])),
                all_voltages_finite=bool(np.isfinite(v).all()),initial_activation_unset=bool(np.isnan(activation).all()))
            if not audits[method]['all_nodes_have_native_initial_voltage']:raise RuntimeError('Initial voltage assignment failed')
        (out/'initial_frame_audit.json').write_text(json.dumps(audits,indent=2))
        report=dict(status='completed',cases=[row],torchcor=provenance,elapsed_s=time.perf_counter()-started,
            scope='3D only; original controlled standard CG/nodal pair; same shared TorchCor CG kernel',
            timing='PDE RHS+checks+CG; calibration/setup/ODE/VTK excluded; separate full trajectory wall times are reported',
            alignment=alignment,initialization=initialization)
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(exc))));raise

if __name__=='__main__':main()
