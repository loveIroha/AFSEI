import pytest
import torch
from afsi_torch.mac.grid import MACGrid, divergence
from afsi_torch.mac.backward_euler import BackwardEulerFlow, BEFlowOptions
from afsi_torch.mac.cut_plane import CutPlane
from afsi_torch.ep.single_port import PhysicalPortDomain
from afsi_torch.ep.dual_port import DualPortBoundary
from afsi_torch.ep.windkessel import WindkesselParameters, flow_driven_trial


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_two_ports_phase_closure_rcr_response_and_commit(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    grid=MACGrid((8,8,8),(1.,1.,1.));a=torch.zeros((8,8),dtype=torch.bool,device=device);a[1:3,2:6]=True
    b=torch.zeros_like(a);b[5:7,2:6]=True
    base=BackwardEulerFlow(grid,dt=1e-4,device=device,options=BEFlowOptions(convection=False,helmholtz_backend='workspace'))
    # Manufactured box boundary test, not an LV or connected anatomical tracts.
    domain=PhysicalPortDomain(grid,a|b)
    model=DualPortBoundary(base,domain,CutPlane(2,-1,a,'MV'),CutPlane(2,-1,b,'AV'))
    pars=WindkesselParameters();zero=grid.zeros(device=device)
    model.configure('filling',old_pc_mmhg=85.,parameters=pars,inlet_mmhg=8.)
    u,p,info=model.advance(zero,zero)
    assert torch.equal(u[2][:,:,-1][b],torch.zeros_like(u[2][:,:,-1][b]))
    assert info['dual_port']['inlet_outward_ml_s']<0
    assert model.old_pc==85.
    assert divergence(u,grid.spacing).abs().max()<1e-7
    expected=flow_driven_trial(85.,0.,1e-4,pars).pc_mmhg
    assert float(model.commit(accepted_step=1))==pytest.approx(expected)
    with pytest.raises(RuntimeError):model.commit(accepted_step=1)
    for step,phase in [(2,'isovolumic_contraction'),(3,'isovolumic_relaxation')]:
        model.configure(phase,old_pc_mmhg=85.,parameters=pars)
        u,p,info=model.advance(zero,zero)
        assert torch.equal(u[2][:,:,-1],torch.zeros_like(u[2][:,:,-1]))
        assert float(model.commit(accepted_step=step))==pytest.approx(expected)
    model.configure('ejection',old_pc_mmhg=7.,parameters=pars)
    u0,p,_=model.advance(zero,zero)
    forcing=tuple(torch.sin(torch.arange(v.numel(),device=device,dtype=v.dtype)).reshape_as(v) for v in zero)
    du=model.response(forcing);u1,p,info=model.advance(zero,forcing)
    error=sum((x-y-z).square().sum() for x,y,z in zip(u1,u0,du)).sqrt()
    scale=sum(z.square().sum() for z in du).sqrt()
    assert error/scale<3e-7
    assert torch.equal(u1[2][:,:,-1][a],torch.zeros_like(u1[2][:,:,-1][a]))
    assert abs(model.active.last_boundary['rcr_gap_mmhg'])<1e-9
    model.advance(zero,zero)
    with pytest.raises(RuntimeError,match='reverse-flow'):model.commit(accepted_step=4)
    # Overlap cannot masquerade as two independent ports.
    with pytest.raises(ValueError,match='overlap'):DualPortBoundary(base,domain,CutPlane(2,-1,a,'MV'),CutPlane(2,-1,a,'AV'))
