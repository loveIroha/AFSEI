"""Closed pressure reference: static equilibrium and exact nonlinear tangent."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import balanced_path,dynamic_driver
from test_afsei_mac_cycle import driver
from afsi_torch.ep.mac_balanced_dynamic import BalancedDynamicProblem,BalancedState
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


def test_prescribed_closed_reference_retains_static_preload(mac_case,balanced_path):
    path,_=balanced_path;d=driver(mac_case);s=d.initialize_preload(path)
    control=replace(s,mode='filling',cap_mode=None,wall_pressure_mmhg=8.,
        cap_reference_mmhg=8.,filling_pressure_mmhg=0.)
    d._boundary_context=control
    try:
        e=d._fsi_stepper(d.maximum_ticks)
        trial,info=e.physical_step(s.fsi,s.fsi.time+e.flow.dt,True)
    finally:d._boundary_context=None
    assert (trial.x-s.fsi.x).abs().max().item()<1e-8
    assert max(v.abs().max().item() for v in trial.velocity)<2e-6
    assert abs(info['balanced']['inlet_flow_ml_s'])<1e-6
    assert abs(info['balanced']['mass_error_ml_s'])<1e-5
    assert abs(info['balanced']['pressure_core']['mean_mmhg'])<1e-6
    assert info['balanced']['cap_reference_mmhg']==8.


def test_reference_completed_joint_jacobian(mac_case):
    d=dynamic_driver(mac_case);s=d.initialize(x=d.mesh.X*1.002)
    e=d._fsi_stepper(d.maximum_ticks);st=e.transfer.prepare(s.fsi.x)
    rows=outlet_rows(e,s.fsi.x,label=1)
    control=BalancedState(s,MACEndocardialPressure(d.model).volume(s.fsi.x).item(),
        torch.zeros_like(rows.area),s.fsi.x.clone(),mode='filling',
        wall_pressure_mmhg=6.,cap_reference_mmhg=6.,filling_pressure_mmhg=0.)
    problem=BalancedDynamicProblem(e,s.fsi,st,e.flow.advect(s.fsi.velocity),control)
    z=torch.cat((torch.zeros_like(s.fsi.x).flatten(),control.cap_pressure/problem.cap_scale))
    z[:s.fsi.x.numel()]=2e-4*torch.sin(torch.arange(s.fsi.x.numel(),device=z.device,dtype=z.dtype))
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype));v/=v.norm();v[-1]=.02
    action=problem.linearization(z);eps=2e-6
    fd=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),fd,rtol=6e-4,atol=8e-6)
    problem.accepted(z);x=problem.last_data[0]
    expected=d.model.force(x,problem.time)+problem.load.force(x,6.)+rows.reaction(torch.full_like(rows.area,6*C))
    torch.testing.assert_close(problem.last_data[3],expected,rtol=2e-10,atol=1e-8)


def test_cap_reference_profile_requires_direct_wall_load():
    from afsi_torch.ep.circulation_config import CardiacCirculationConfig
    with pytest.raises(ValueError,match='cap reference'):
        CardiacCirculationConfig(filling_cap_reference=True)
    with pytest.raises(ValueError,match='cap reference'):
        CardiacCirculationConfig(filling_pressure_location='endocardium',filling_cap_reference=1)
