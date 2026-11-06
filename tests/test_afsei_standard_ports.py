"""Independent derivatives and upstream S/J equivalence on the GPU."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import dynamic_driver
from afsi_torch.ep.mac_standard_ports import StandardPortTransfer, StandardPortProblem
from afsi_torch.ep.mac_balanced_dynamic import BalancedState, BalancedAFSEIStepper
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


@pytest.mark.parametrize('mode',['closed','filling','rcr'])
@pytest.mark.parametrize('cap_rule',['centroid','gauss7','gauss12'])
def test_standard_port_jacobian_and_unmodified_pair(mac_case,mode,cap_rule):
    d=dynamic_driver(mac_case); d.one_way=False
    d.cap_integration_rule=cap_rule
    ep=d.initialize(x=d.mesh.X*1.002)
    e=d._fsi_stepper(d.maximum_ticks)
    e.transfer=StandardPortTransfer(e,e.transfer.base)
    stencil=e.transfer.prepare(ep.fsi.x)
    rows=outlet_rows(e,ep.fsi.x,label=1 if mode=='filling' else 2) if mode!='closed' else stencil.rows
    control=BalancedState(ep,MACEndocardialPressure(d.model).volume(ep.fsi.x).item(),
        torch.full_like(rows.area,8*C),ep.fsi.x.clone(),mode=mode)
    problem=StandardPortProblem(e,ep.fsi,stencil,e.flow.advect(ep.fsi.velocity),control)
    z=torch.cat((torch.zeros_like(ep.fsi.x).flatten(),control.cap_pressure/problem.cap_scale))
    z[:ep.fsi.x.numel()]=2e-4*torch.sin(torch.arange(ep.fsi.x.numel(),device=z.device,dtype=z.dtype))
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype)); v/=v.norm(); v[-1]=.02
    action=problem.linearization(z); eps=2e-6
    expected=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),expected,rtol=6e-4,atol=8e-6)
    f=torch.sin(ep.fsi.x)
    e.transfer.reset_warm_start(); sf,_=e.transfer.spread(f,stencil)
    e.transfer.base.reset_warm_start(); base,_=e.transfer.base.spread(f,stencil.original)
    for a,b in zip(sf,base): torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    u=tuple(torch.cos(a) for a in sf)
    e.transfer.reset_warm_start(); j,_=e.transfer.interpolate(u,stencil)
    e.transfer.base.reset_warm_start(); jb,_=e.transfer.base.interpolate(u,stencil.original)
    torch.testing.assert_close(j,jb,rtol=1e-11,atol=1e-11)
    assert stencil.potential.item()==0


@pytest.mark.parametrize('fluid_solver',['be-projection','be-stokes-fgmres'])
@pytest.mark.parametrize('cap_rule',['centroid','gauss7','gauss12'])
def test_explicit_standard_method_survives_checkpoint(mac_case,tmp_path,fluid_solver,cap_rule):
    from afsi_torch.ep import mac_balanced_checkpoint as restart
    from afsi_torch.ep.mac_cap_coupled import closed_rows
    base=dynamic_driver(mac_case)
    d=BalancedAFSEIStepper(base.mesh,base.config,
        spatial_coupling='griffith-luo-unified',pressure_backend='multigrid',fluid_solver=fluid_solver,
        cap_integration_rule=cap_rule)
    d.port_geometry=base.port_geometry
    d.pressure_guard_cells=base.pressure_guard_cells
    ep=d.initialize()
    e=d._fsi_stepper(d.maximum_ticks)
    assert isinstance(e.transfer,StandardPortTransfer)
    assert 'dst' not in e.flow.pressure_solver.backend
    rows=closed_rows(e,ep.fsi.x)
    state=BalancedState(ep,MACEndocardialPressure(d.model).volume(ep.fsi.x).item(),
        torch.zeros_like(rows.area),ep.fsi.x.clone())
    state=replace(state,ep=replace(ep,fsi=replace(ep.fsi,force=d.force_for_state(state))))
    path=tmp_path/'standard.npz'; restart.save(path,d,state)
    restored,saved,_=restart.load(path)
    assert restored.spatial_coupling=='griffith-luo-unified'
    assert restored.pressure_backend=='multigrid'
    assert restored.volume_transfer=='griffith-luo-unified-p1-v1'
    assert restored.fluid_solver==fluid_solver
    assert restored.cap_integration_rule==cap_rule
    assert isinstance(restored._fsi_stepper(restored.maximum_ticks).transfer,StandardPortTransfer)
    torch.testing.assert_close(saved.fsi.x,state.fsi.x,rtol=0.,atol=0.)
