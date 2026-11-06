"""Independent waveform, complementarity, GPU derivative and restart checks."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.ep.gao_filling import (GaoFillingParameters, filling_fraction,
    filling_port_law, filling_port_accepted)
from afsi_torch.ep.circulation_config import CardiacCirculationConfig
from afsi_torch.ep.windkessel import WindkesselParameters, MMHG_TO_DYN_CM2 as C
from afsi_torch.ep.mac_cycle import CardiacStepper
from afsi_torch.ep.mac_balanced_dynamic import BalancedDynamicProblem, BalancedState
from afsi_torch.ep.mac_standard_ports import StandardPortTransfer, StandardPortProblem
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep import mac_balanced_checkpoint as restart
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import dynamic_driver, balanced_path
from test_afsei_driver import assert_same, physical_arrays


def test_published_waveforms_on_rescaled_filling_interval():
    p=GaoFillingParameters()
    for fraction,expected in [(0.,(0.,0.)),(.25,(2.,6.)),(.5,(4.,12.)),(.75,(6.,6.)),(1.,(8.,0.))]:
        assert p.pressures(fraction,8.)==expected
    assert p.pressures(2.,8.)==(8.,0.)
    state=SimpleNamespace(fill_start_tick=18000,fill_end_tick=26000)
    assert filling_fraction(state,22000)==.5
    with pytest.raises(ValueError): filling_fraction(SimpleNamespace(fill_start_tick=5,fill_end_tick=5),6)
    with pytest.raises(ValueError): GaoFillingParameters(wall_peak_fraction=1.)


def test_config_separates_supply_from_peripheral_reference(tmp_path):
    from afsi_torch.config import load_config
    path=tmp_path/'gao.json'
    path.write_text('{"filling_model":"gao2017","windkessel":{"venous_mmhg":3.0},"gao_filling":{"wall_peak_mmhg":12.0}}')
    c=load_config(path,CardiacCirculationConfig)
    assert c.windkessel.venous_mmhg==3. and c.fill_target_mmhg==8.
    assert c.gao_filling.wall_peak_mmhg==12. and c.gao_filling.inlet_one_way
    with pytest.raises(ValueError): CardiacCirculationConfig(filling_model='gao2017',filling_pressure_location='endocardium')
    with pytest.raises(ValueError): CardiacCirculationConfig(filling_model='gao2017',filling_cap_reference=True)


def test_inlet_complementarity_signs():
    # Below supply: an open inlet permits negative outward flow.
    residual,opened=filling_port_law(torch.tensor(-2.),torch.tensor(4.),4.,True)
    assert opened and residual==0
    assert filling_port_accepted(-2.,-2.,4.,4.,True,True)
    # Above supply: the inlet is closed, without post-hoc clipping of flow.
    residual,opened=filling_port_law(torch.tensor(0.),torch.tensor(7.),4.,True)
    assert not opened and residual==0
    assert filling_port_accepted(0.,0.,7.,4.,False,True)
    assert not filling_port_accepted(1.,1.,4.,4.,True,True)
    assert not filling_port_accepted(0.,0.,3.,4.,False,True)


@pytest.mark.parametrize('standard',[False,True])
@pytest.mark.parametrize('opened',[False,True])
def test_gpu_one_way_joint_jacobian_and_port_work(mac_case,standard,opened):
    d=dynamic_driver(mac_case); d.inlet_one_way=True
    ep=d.initialize(x=d.mesh.X*1.002)
    e=d._fsi_stepper(d.maximum_ticks)
    if standard: e.transfer=StandardPortTransfer(e,e.transfer.base)
    stencil=e.transfer.prepare(ep.fsi.x); rows=outlet_rows(e,ep.fsi.x,label=1)
    control=BalancedState(ep,MACEndocardialPressure(d.model).volume(ep.fsi.x).item(),
        torch.full_like(rows.area,8*C),ep.fsi.x.clone(),mode='filling',
        filling_pressure_mmhg=100. if opened else -100.,wall_pressure_mmhg=6.)
    cls=StandardPortProblem if standard else BalancedDynamicProblem
    problem=cls(e,ep.fsi,stencil,e.flow.advect(ep.fsi.velocity),control)
    z=torch.cat((torch.zeros_like(ep.fsi.x).flatten(),control.cap_pressure/problem.cap_scale))
    z[:ep.fsi.x.numel()]=2e-4*torch.sin(torch.arange(ep.fsi.x.numel(),device=z.device,dtype=z.dtype))
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype));v/=v.norm();v[-1]=.02
    action=problem.linearization(z);assert problem.linear_inlet_open is opened
    eps=2e-6;fd=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),fd,rtol=6e-4,atol=8e-6)
    # No extra cap-reference force is hidden in the Gao wall/supply combination.
    problem.accepted(z);x=problem.last_data[0]
    expected=d.model.force(x,problem.time)+problem.load.force(x,6.)+rows.reaction(problem.last_cap_pressure)
    torch.testing.assert_close(problem.last_data[3],expected,rtol=2e-10,atol=1e-8)
    from afsi_torch.mac.open_boundary import face_weights
    u,nodal=problem.last_data[4],problem.last_data[7];bp=problem.last_cap_pressure
    fluid=-sum((a*b*face_weights(e.grid,c,x)).sum() for c,(a,b) in enumerate(zip(u,rows.lift(bp))))
    solid=(rows.reaction(bp)*nodal).sum()
    torch.testing.assert_close(fluid+solid,-(bp*(rows.flow(u)-rows.material_flow(nodal))).sum(),rtol=1e-10,atol=1e-8)


def test_gpu_gao_step_restart_and_rejection(mac_case,balanced_path,tmp_path,monkeypatch):
    ref=dynamic_driver(mac_case)
    d=CardiacStepper(ref.mesh,ref.config,level=2,filling_model='gao2017')
    d.port_geometry=ref.port_geometry;d.pressure_guard_cells=1
    initial=d.initialize_preload(balanced_path[0])
    old=replace(initial,phase='FILLING',mode='filling',cap_mode='rcr',
        fill_start_tick=6000,fill_end_tick=14000,fill_start_pressure_mmhg=0.,
        ep=replace(initial.ep,cycle=replace(initial.cycle,mode='zero',closure_tick=8400)))
    new,info=d.step(old);r=info['balanced']
    fraction=(new.tick-6000)/8000
    assert r['supply_pressure_mmhg']==pytest.approx(8*fraction)
    assert r['wall_pressure_mmhg']==pytest.approx(24*(1-fraction))
    assert new.cap_reference_mmhg==0. and not new.outlet_open
    assert r['inlet_flow_ml_s']>=-1e-5 and abs(r['mass_error_ml_s'])<1e-5
    assert r['boundary_work_relative_error']<1e-7
    assert r['inlet_reservoir_power_erg_s']==pytest.approx(C*r['supply_pressure_mmhg']*r['inlet_flow_ml_s'])
    path=tmp_path/'gao.npz';restart.save(path,d,new)
    restored,copied,_=restart.load(path,cardiac=True)
    assert restored.filling_model=='gao2017' and restored.inlet_one_way
    assert restored.gao_filling==d.gao_filling
    assert_same(new,copied,exact=True)
    a,_=d.step(new);b,_=restored.step(copied)
    assert a.tick==b.tick and a.fsi.time==b.fsi.time
    for x,y in zip(physical_arrays(a),physical_arrays(b)):
        if x is a.fsi.pressure:
            # Compare pressure in physical mmHg, including values near zero.
            torch.testing.assert_close(x/C,y/C,rtol=3e-8,atol=1e-7)
        elif x is a.fsi.force:
            # Compare the force vector relative to its norm: individual entries
            # near zero suffer cancellation of wall, elastic and cap forces.
            assert (x-y).norm() <= 3e-8*max(x.norm().item(),y.norm().item(),1.)
        else:torch.testing.assert_close(x,y,rtol=3e-8,atol=2e-8)
    arrays=[v.clone() for v in physical_arrays(new)]
    def fail(*a,**k):raise RuntimeError('forced Gao flow failure')
    for n in range(1,d.maximum_ticks+1):monkeypatch.setattr(d._fsi_stepper(n).flow,'advance',fail)
    from afsi_torch.ep.driver import CoupledStepFailure
    with pytest.raises(CoupledStepFailure):d.step(new)
    assert d._boundary_context is None
    for a,b in zip(arrays,physical_arrays(new)):assert torch.equal(a,b)
