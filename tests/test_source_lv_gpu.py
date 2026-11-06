"""Real CUDA/IB integration, generalized tangent, and new checkpoint schema."""
from dataclasses import replace
import torch
import pytest
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_isovolumic import precise_driver
from test_afsei_driver import assert_same
from afsi_torch.ep.mac_source_cycle import SourceLVStepper,SourceCycleProblem,SourceLVState
from afsi_torch.ep.protocol import EPProtocol,CycleState
from afsi_torch.ep.checkpoint import save,load


def test_source_lv_gpu_pressure_volume_jacobian_restart(mac_case,tmp_path):
    base=precise_driver(mac_case)
    cfg=replace(base.config,start_time_s=0.,protocol=EPProtocol(pulse_starts_s=(.5,)))
    driver=SourceLVStepper(base.mesh,cfg)
    old=driver.initialize()
    filled,info=driver.step(old)
    assert filled.cavity_pressure_mmhg==pytest.approx(8*filled.fsi.time/.2,abs=1e-7)
    assert filled.phase=='PRELOAD' and info['ep']['steps']==0
    path=tmp_path/'source_lv.npz';save(path,driver,filled)
    restored,state,_=load(path)
    assert isinstance(state,SourceLVState) and isinstance(restored,SourceLVStepper)
    assert restored.execution_settings()==driver.execution_settings()
    assert_same(filled,state,exact=True)
    a,_=driver.step(filled);b,_=restored.step(state);assert_same(a,b)
    # Manufactured time shift tests the IVC equation with actual EP/ODE and IB.
    # It is not a physical 500-ms preload history or a production result.
    t=driver.protocol.pulse_ticks[0];time=t*driver.protocol.dt_ep_s
    ep=replace(a.ep,tick=t,fsi=replace(a.fsi,time=time,force_time=time,pressure_time=time),
               cycle=CycleState(mode='land',resting_land=a.tissue.land.clone()))
    active=replace(a,ep=ep,phase='IVC',beat=1)
    contracted,ci=driver.step(active)
    assert ci['ep']['steps']>0 and ci['ep']['land_advanced']
    assert contracted.phase=='IVC' and contracted.outlet_flow_ml_s==0
    assert abs(ci['circulation']['volume_balance_error_ml'])<=1e-3
    save(path,driver,contracted);_,same,_=load(path);assert_same(contracted,same,exact=True)
    endpoint=driver._fsi_stepper(driver.maximum_ticks)
    stencil=endpoint.transfer.prepare(contracted.fsi.x)
    # Synthetic arterial state opens the valve in this small test fixture.
    control=replace(contracted,arterial_pc_mmhg=.05)
    problem=SourceCycleProblem(endpoint,contracted.fsi,stencil,
                               endpoint.flow.advect(contracted.fsi.velocity),control)
    z=contracted.fsi.x.new_zeros(contracted.fsi.x.numel()+1)
    # Central differences must stay on one branch of the existing tension-only
    # fiber/active-stretch laws; the just-stimulated state is arbitrarily near 1.
    z[:-1]=(.002*(contracted.fsi.x-contracted.fsi.x.mean(0))).flatten()
    z[-1]=(2.-contracted.cavity_pressure_mmhg)/problem.pressure_scale
    assert problem.trial(torch.tensor(2.)).flow_ml_s>0
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype));v/=v.norm();v[-1]=.1
    action=problem.linearization(z);eps=2e-6
    finite=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),finite,rtol=2e-4,atol=2e-6)
