"""Continue only the unfinished method; never rerun the completed trajectory."""
from pathlib import Path
import importlib.util,json,sys,time,hashlib
import numpy as np,torch,meshio
root=Path('/mnt/large2/gjh/AFSEI');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'src'))
from afsi_torch.config import load_config
from afsi_torch.ep.experiment import EPTissueConfig
from afsi_torch.ep.cells import GPBCell
from validation.torchcor_pde import load_core,COMMIT

@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(4)
    out=root/'results/ep_opencarp_sigma_nativeinit_20261030';demo=root/'demo/ep_opencarp_sigma_nativeinit'
    completed=out/'3d/consistent_tc/fields.npz'
    if not completed.is_file():raise RuntimeError('Completed standard-CG trajectory required')
    if (out/'3d/nodal_tc/fields.npz').exists():raise RuntimeError('Nodal trajectory is already complete; do not rerun it')
    # Retain the incomplete folder for provenance, never reuse partial Vm as a restart.
    partial=out/'3d/nodal_tc';archive=out/'3d/nodal_tc_interrupted_before_completion'
    if partial.exists():
        if archive.exists():raise RuntimeError('Partial archive already exists')
        if partial.resolve().parent!=(out/'3d').resolve():raise RuntimeError('Unexpected partial path')
        partial.rename(archive)
    source=root/'scripts/benchmark_ep_controlled.py';text=source.read_text()
    substitutions={
        ";dest.mkdir();X,cells=mesh(dim,c)":";dest.mkdir(exist_ok=True);X,cells=mesh(dim,c)",
        'output={};data={};inputs=[];kind=':'output=resume_output.copy();data=resume_data.copy();inputs=[];kind=',
        '    for name,m in zip(NAMES,methods):\n        folder=':'    for name,m in zip(NAMES,methods):\n        if name in resume_data:continue\n        folder=',
        '    replay=frozen(methods,inputs,dt,options)':'    replay={"status":"not repeated when resuming only the unfinished method"}',
    }
    for old,new in substitutions.items():
        if text.count(old)!=1:raise RuntimeError('Unexpected benchmark source for resume: '+old)
        text=text.replace(old,new)
    copied=demo/'controlled_resume.py';copied.write_text(text)
    spec=importlib.util.spec_from_file_location('sigma_controlled_resume',copied);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    with np.load(completed) as z:data={k:z[k] for k in z.files}
    pde=data['pde_step_s'];ode=data['ode_step_s'];act=data['activation_s']
    if len(pde)!=2400 or len(ode)!=2400 or len(data['traces_v'])!=2401:raise RuntimeError('Incomplete standard-CG trajectory')
    restored=dict(pde_mean_ms=float(pde.mean()*1000),pde_total_s=float(pde.sum()),ode_total_s=float(ode.sum()),
        wall_with_output_and_calibration_s=None,mean_iterations=None,max_residual_ratio=None,
        activated_fraction=float(np.isfinite(act).mean()),far_corner_activation_ms=float(act[-1]*1000),
        provenance='Completed trajectory reused verbatim. Per-step timing arrays persisted; whole-wall and iteration aggregates were not persisted before interruption.')
    core.resume_data={'consistent_tc':data};core.resume_output={'consistent_tc':restored}
    config=load_config(demo/'config.json',EPTissueConfig);c=json.loads((out/'config.json').read_text())
    classes,provenance=load_core(root/'.deps'/('torchcor-'+COMMIT))
    scalar=GPBCell(config.time.dt_ep_s,config.time.dt_ode_ms).initial(torch.zeros(1,device='cuda'))
    assert float(scalar.voltage_v[0])*1000==-81.5455936324844
    resume=dict(reason='Initial-voltage clarification; final native voltage confirmed, no physics change',
        reused_standard_sha256=hashlib.sha256(completed.read_bytes()).hexdigest(),
        original_benchmark_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        numerical_method_changes=False,standard_repeated=False,nodal_resumed_from_initial_state=True,
        saved_standard_timing_scope='Exact PDE and ODE sums retained; unavailable full-wall timing is not reconstructed')
    (out/'resume.json').write_text(json.dumps(resume,indent=2));print(json.dumps(resume),flush=True)
    start=time.perf_counter()
    row=core.run_case(3,c,config,scalar,classes,out,primary_only=True)
    audits={}
    for method in ('consistent_tc','nodal_tc'):
        m=meshio.read(out/'3d'/method/'ep_00000.vtu');v=m.point_data['voltage_mV']
        audits[method]=dict(nodes=len(v),initial_voltage_min_mv=float(v.min()),initial_voltage_max_mv=float(v.max()),
            all_nodes_have_native_initial_voltage=bool(np.all(v==-81.5455936324844)),all_voltages_finite=bool(np.isfinite(v).all()),
            initial_activation_unset=bool(np.isnan(m.point_data['activation_ms']).all()))
        if not audits[method]['all_nodes_have_native_initial_voltage']:raise RuntimeError('Initial frame voltage mismatch')
    (out/'initial_frame_audit.json').write_text(json.dumps(audits,indent=2))
    report=dict(status='completed',cases=[row],torchcor=provenance,resume=resume,resume_elapsed_s=time.perf_counter()-start,
        alignment=json.loads((out/'alignment.json').read_text()),initialization=json.loads((out/'initialization.json').read_text()),
        scope='3D original controlled standard CG/nodal pair; shared pinned TorchCor CG, no AmgX',
        timing='Exact per-step PDE and ODE arrays for both full trajectories; standard-CG whole-wall aggregate unavailable after interruption')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
