"""Read-only original IB volume transport and pressure-work validation."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_transport import (ClosedVolumeTransport,closed_volume_gradient,
    audit_projected_transport)
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2


def gradient_benchmark(x,load,count):
    original=torch.func.grad(load.volume)
    analytic=lambda y:closed_volume_gradient(y,load.faces,load.rim)
    a,b=original(x),analytic(x)
    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    timing={}
    for name,operation in (('autograd',original),('analytic',analytic)):
        for _ in range(3): operation(x)
        torch.cuda.synchronize(); started=perf_counter()
        for _ in range(count): operation(x)
        torch.cuda.synchronize(); timing[name+'_seconds_per_call']=(perf_counter()-started)/count
    return dict(**timing,repetitions=count,maximum_gradient_difference_cm2=(a-b).abs().max().item(),
        scope='same actual geometry; CUDA synchronized batches after warmup; not full-step speedup')


@torch.no_grad()
def diagnose(path,*,benchmark_count=0):
    sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    driver,state,_=checkpoint.load(path)
    old=state.force_load.previous_x; dt=state.force_load.dt_s
    endpoint=driver._fsi_stepper(round(dt/driver.protocol.dt_ep_s))
    torch.cuda.synchronize(); started=perf_counter()
    stencil=endpoint.transfer.prepare(old)
    torch.cuda.synchronize(); prepare_seconds=perf_counter()-started
    started=perf_counter()
    transport=ClosedVolumeTransport.prepare(old,state.fsi.x,driver.model.endo_faces,dt,rim=driver.model.rim)
    report=audit_projected_transport(transport,endpoint.transfer,stencil,state.fsi.velocity)
    load=MACEndocardialPressure(driver.model)
    source_gradient=load.force(state.fsi.x,1.)/MMHG_TO_DYN_CM2
    source_error=(source_gradient-transport.current_covectors[0]).abs().max().item()
    if source_error>1e-10: raise AssertionError('source wall pressure and current wall covector disagree')
    torch.cuda.synchronize(); audit_seconds=perf_counter()-started
    report.update(source_wall_pressure_covector_max_error_cm2=source_error,
        checkpoint=str(path),source_sha256=sha,source_checkpoint_unchanged=True,
        time_s=state.fsi.time,step=state.fsi.step,grid_shape=endpoint.grid.shape,
        gpu=torch.cuda.get_device_name(),cap_patch_width_cells=getattr(driver,'cap_patch_width_cells',None),
        transfer_prepare_seconds=prepare_seconds,audit_seconds=audit_seconds,
        timing_scope='offline geometry and original IB/adjoint audit; stencil preparation separately timed')
    if benchmark_count:
        report['gradient_benchmark']=gradient_benchmark(state.fsi.x,load,benchmark_count)
    if sha!=hashlib.sha256(Path(path).read_bytes()).hexdigest():
        raise AssertionError('input checkpoint changed during read-only audit')
    if report['swept_geometry_identity_error_ml_s']>1e-7 or report['transport_defect_identity_error_ml_s']>1e-7:
        raise AssertionError('geometric/projection transport identity failed')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',nargs='+',required=True)
    p.add_argument('--benchmark-gradients',type=int,default=0,metavar='REPEATS')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available(): p.error('actual CUDA required')
    if a.benchmark_gradients<0 or a.benchmark_gradients>1000: p.error('benchmark repetitions in 0..1000 required')
    a.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for path in a.checkpoint:
        report=diagnose(path,benchmark_count=a.benchmark_gradients)
        print(json.dumps(report,indent=2),flush=True)
        reports.append(report); torch.cuda.empty_cache()
    result=dict(passed=True,scope=__doc__,snapshots=reports,
        all_volume_transport_audits_passed=all(r['volume_transport_audit_passed'] for r in reports),
        pointwise_impermeability_established=False,windkessel_actual_port_flow_connected=False,
        three_field_integration_implemented=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
