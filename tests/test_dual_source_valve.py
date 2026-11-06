import pytest
import torch
from afsi_torch.mac.grid import MACGrid,divergence
from afsi_torch.mac.backward_euler import BackwardEulerFlow,BEFlowOptions
from afsi_torch.mac.cut_plane import CutPlane
from afsi_torch.ep.single_port import PhysicalPortDomain
from afsi_torch.ep.dual_port import DualPortBoundary
from afsi_torch.ep.aortic_valve import AorticValveParameters,ValveClosureRequired
from afsi_torch.ep.windkessel import WindkesselParameters


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_dual_source_valve_fluid_flux_and_closed_faces(device):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA required')
    grid=MACGrid((8,8,8),(1.,1.,1.))
    a=torch.zeros((8,8),dtype=torch.bool,device=device);a[1:3,2:6]=True
    b=torch.zeros_like(a);b[5:7,2:6]=True
    base=BackwardEulerFlow(grid,dt=1e-4,device=device,
        options=BEFlowOptions(convection=False,helmholtz_backend='workspace'))
    model=DualPortBoundary(base,PhysicalPortDomain(grid,a|b),CutPlane(2,-1,a,'in'),
        CutPlane(2,-1,b,'out'),valve=AorticValveParameters())
    r=WindkesselParameters();zero=grid.zeros(device=device)
    # Manufactured boundary test, not a connected LV. Negative downstream
    # pressure drives forward flow from the other exterior reference faces.
    model.configure('ejection',old_pc_mmhg=-10.,parameters=r)
    u,p,info=model.advance(zero,zero)
    assert torch.equal(u[2][:,:,-1][a],torch.zeros_like(u[2][:,:,-1][a]))
    assert info['dual_port']['outlet_outward_ml_s']>0
    assert info['dual_port']['upstream_port_mmhg']>info['dual_port']['arterial_pressure_mmhg']
    assert abs(float(info['cut_boundary']['valve_flow_residual_ml_s']))<1e-8
    assert float(divergence(u,grid.spacing).abs().max())<1e-7
    model.commit(accepted_step=1)
    # A closure signal leaves no pending circuit state that could be committed.
    model.configure('ejection',old_pc_mmhg=85.,parameters=r)
    with pytest.raises(ValveClosureRequired):model.advance(zero,zero)
    with pytest.raises(RuntimeError):model.commit(accepted_step=2)
    for step,phase in [(2,'isovolumic_contraction'),(3,'isovolumic_relaxation')]:
        model.configure(phase,old_pc_mmhg=85.,parameters=r)
        u,p,info=model.advance(zero,zero)
        assert torch.equal(u[2][:,:,-1][a|b],torch.zeros_like(u[2][:,:,-1][a|b]))
        assert float(model.commit(accepted_step=step))<85.
    model.configure('filling',old_pc_mmhg=85.,parameters=r,inlet_mmhg=8.)
    u,p,info=model.advance(zero,zero)
    assert torch.equal(u[2][:,:,-1][b],torch.zeros_like(u[2][:,:,-1][b]))
    assert info['dual_port']['inlet_outward_ml_s']<0
