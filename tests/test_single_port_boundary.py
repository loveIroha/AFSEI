import pytest
import torch
from afsi_torch.mac.grid import MACGrid, divergence
from afsi_torch.mac.backward_euler import BackwardEulerFlow, BEFlowOptions
from afsi_torch.ep.single_port import SinglePortBoundary
from afsi_torch.ep.windkessel import WindkesselParameters


@pytest.mark.parametrize('dt',[1e-4,5e-5])
def test_phase_boundary_and_rcr_derivative(dt):
    grid=MACGrid((8,8,8),(1.,1.,1.))
    base=BackwardEulerFlow(grid,dt=dt,device='cpu',
        options=BEFlowOptions(convection=False,helmholtz_backend='workspace'))
    aperture=torch.zeros((8,8),dtype=torch.bool);aperture[2:6,2:6]=True
    f=SinglePortBoundary(base,aperture)
    pars=WindkesselParameters();zero=grid.zeros(device='cpu')
    with pytest.raises(RuntimeError):f.advance(zero,zero)
    f.configure('filling',old_pc_mmhg=85.,parameters=pars,inlet_mmhg=8.)
    u,p,info=f.advance(zero,zero)
    assert info['single_port']['outward_flux_ml_s']<0
    assert info['single_port']['phase_direction_valid']
    assert f.last_circuit.pc_mmhg<85.
    assert float(f.commit(accepted_step=1))<85.
    with pytest.raises(RuntimeError):f.commit(accepted_step=1)
    assert f.old_pc==85.
    assert divergence(u,grid.spacing).abs().max()<1e-7
    f.configure('closed',old_pc_mmhg=85.,parameters=pars)
    u,p,info=f.advance(zero,zero)
    assert torch.equal(u[2][:,:,-1],torch.zeros_like(u[2][:,:,-1]))
    assert info['single_port']['outward_flux_ml_s']==0
    assert f.last_circuit.pc_mmhg==pytest.approx((pars.compliance_ml_per_mmhg*85.+dt*8./pars.rp_mmhg_s_per_ml)/(pars.compliance_ml_per_mmhg+dt/pars.rp_mmhg_s_per_ml))
    f.configure('ejection',old_pc_mmhg=7.,parameters=pars)
    u0,p0,_=f.advance(zero,zero)
    forcing=tuple(torch.sin(torch.arange(v.numel(),dtype=v.dtype)).reshape_as(v) for v in zero)
    du=f.response(forcing);u1,p1,info=f.advance(zero,forcing)
    error=sum((a-b-c).square().sum() for a,b,c in zip(u1,u0,du)).sqrt()
    scale=sum(c.square().sum() for c in du).sqrt()
    assert error/scale<2e-7
    assert f.active.last_boundary['rcr_gap_mmhg'].abs()<1e-9
    assert f.old_pc==7.
    # No pump: positive afterload drives reverse flow, which must be reported
    # for the future phase controller to reject/close; never silently clamped.
    assert not f.advance(zero,zero)[2]['single_port']['phase_direction_valid']
    with pytest.raises(RuntimeError,match='reverse-flow'):f.commit(accepted_step=2)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_cuda_port_commit_and_rejected_reverse_flow():
    grid=MACGrid((8,8,8),(1.,1.,1.))
    base=BackwardEulerFlow(grid,dt=1e-4,device='cuda:0',
        options=BEFlowOptions(convection=False,helmholtz_backend='workspace'))
    aperture=torch.zeros((8,8),dtype=torch.bool,device='cuda:0');aperture[2:6,2:6]=True
    f=SinglePortBoundary(base,aperture);zero=grid.zeros(device='cuda:0')
    pars=WindkesselParameters()
    f.configure('closed',old_pc_mmhg=85.,parameters=pars)
    u,p,info=f.advance(zero,zero)
    assert info['single_port']['outward_flux_ml_s']==0
    assert divergence(u,grid.spacing).abs().max()<1e-7
    assert float(f.commit(accepted_step=1))<85.
    f.configure('ejection',old_pc_mmhg=85.,parameters=pars)
    f.advance(zero,zero)
    with pytest.raises(RuntimeError,match='reverse-flow'):f.commit(accepted_step=2)
