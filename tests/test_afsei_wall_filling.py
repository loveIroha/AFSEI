"""Direct endocardial follower load: exact tangent, work and accepted history."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import dynamic_driver, balanced_path
from test_afsei_mac_cycle import driver
from test_afsei_driver import assert_same
from afsi_torch.ep.mac_balanced_dynamic import BalancedDynamicProblem, BalancedState
from afsi_torch.ep.mac_standard_ports import StandardPortTransfer, StandardPortProblem
from afsi_torch.ep.mac_rcr_coupled import outlet_rows
from afsi_torch.ep.mac_preload import MACEndocardialPressure
from afsi_torch.ep.mac_cycle import filling_pressure
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.windkessel import MMHG_TO_DYN_CM2 as C


@pytest.mark.parametrize('standard',[False,True])
def test_wall_follower_joint_jacobian_and_single_loading(mac_case,standard):
    d=dynamic_driver(mac_case); ep=d.initialize(x=d.mesh.X*1.002)
    e=d._fsi_stepper(d.maximum_ticks)
    if standard: e.transfer=StandardPortTransfer(e,e.transfer.base)
    stencil=e.transfer.prepare(ep.fsi.x); rows=outlet_rows(e,ep.fsi.x,label=1)
    control=BalancedState(ep,MACEndocardialPressure(d.model).volume(ep.fsi.x).item(),
        torch.zeros_like(rows.area),ep.fsi.x.clone(),mode='filling',
        filling_pressure_mmhg=0.,wall_pressure_mmhg=7.)
    cls=StandardPortProblem if standard else BalancedDynamicProblem
    problem=cls(e,ep.fsi,stencil,e.flow.advect(ep.fsi.velocity),control)
    z=torch.cat((torch.zeros_like(ep.fsi.x).flatten(),control.cap_pressure/problem.cap_scale))
    z[:ep.fsi.x.numel()]=2e-4*torch.sin(torch.arange(ep.fsi.x.numel(),device=z.device,dtype=z.dtype))
    v=torch.cos(torch.arange(len(z),device=z.device,dtype=z.dtype)); v/=v.norm(); v[-1]=.02
    action=problem.linearization(z); eps=2e-6
    expected=(problem.residual(z+eps*v)-problem.residual(z-eps*v))/(2*eps)
    torch.testing.assert_close(action(v),expected,rtol=6e-4,atol=8e-6)
    problem.accepted(z); x=problem.last_data[0]
    # At zero inlet/closed-patch multipliers the force contains ONE wall load.
    expected_force=d.model.force(x,problem.time)+problem.load.force(x,7.)
    # The compiled material force can differ by a few 1e-9 dyn from its eager
    # reconstruction; the prescribed wall load here is many orders larger.
    torch.testing.assert_close(problem.last_data[3],expected_force,rtol=2e-10,atol=1e-8)
    assert control.filling_pressure_mmhg==0.
    velocity=torch.cos(x); load=problem.load.force(x,7.)
    area=.5*torch.linalg.cross(x[problem.load.faces][:,1]-x[problem.load.faces][:,0],
                              x[problem.load.faces][:,2]-x[problem.load.faces][:,0])
    surface_power=-7.*C*(area*velocity[problem.load.faces].mean(1)).sum()
    torch.testing.assert_close((load*velocity).sum(),surface_power,rtol=1e-12,atol=1e-9)


def test_wall_fill_event_history_restart_and_rollback(mac_case,balanced_path,tmp_path,monkeypatch):
    path,_=balanced_path; d=driver(mac_case)
    d.filling_pressure_location='endocardium'; d.filling_reservoir_mmhg=0.
    initial=d.initialize_preload(path)
    # The preceding solve used the 8 mmHg cap; the next solve uses a wall
    # follower load and an independent zero-pressure supply port.
    old=replace(initial,phase='FILLING',mode='filling',cap_mode='rcr',
        fill_start_tick=initial.tick,fill_end_tick=26000,fill_start_pressure_mmhg=8.,
        ep=replace(initial.ep,cycle=replace(initial.cycle,mode='zero',closure_tick=8400)))
    state,info=d.step(old)
    r=info['balanced']
    assert state.wall_pressure_mmhg==pytest.approx(filling_pressure(old,state.tick))
    assert r['boundary_pressure_mmhg']==pytest.approx(0.,abs=1e-7)
    assert abs(r['mass_error_ml_s'])<1e-5 and r['transfer_power_relative_error']<1e-7
    torch.testing.assert_close(d.force_for_state(state),state.fsi.force,rtol=1e-9,atol=1e-7)
    assert state.arterial_pc_mmhg<old.arterial_pc_mmhg
    save=tmp_path/'wall.npz'; restart.save(save,d,state)
    restored,copied,_=restart.load(save,cardiac=True)
    assert copied.wall_pressure_mmhg==state.wall_pressure_mmhg
    assert restored.filling_pressure_location=='endocardium' and restored.filling_reservoir_mmhg==0.
    a,_=d.step(state); b,_=restored.step(copied); assert_same(a,b)
    assert a.wall_pressure_mmhg==b.wall_pressure_mmhg
    # Publish pacing with an accepted filling load. The force history must
    # remain reconstructible even though the NEXT boundary mode is RCR.
    paced=replace(state,ep=replace(state.ep,tick=26000,
        fsi=replace(state.fsi,time=1.3,force_time=1.3,pressure_time=1.3)))
    event=dict(balanced=dict(r,inlet_flow_ml_s=0.,measured_relative_q_ml_s=0.,volume_ml=state.fsi.x.new_tensor(r['volume_ml']).item()),
               accepted_dt_s=0.)
    next_state,_=d.publish_phase(state,paced,event)
    assert next_state.mode=='rcr' and next_state.wall_pressure_mmhg==state.wall_pressure_mmhg
    torch.testing.assert_close(d.force_for_state(next_state),next_state.fsi.force,rtol=1e-9,atol=1e-7)
    saved_force=state.fsi.force.clone()
    from afsi_torch.ep.driver import CoupledStepFailure
    def fail(*args,**kwargs): raise RuntimeError('forced boundary-load failure')
    for n in range(1,d.maximum_ticks+1): monkeypatch.setattr(d._fsi_stepper(n).flow,'advance',fail)
    with pytest.raises(CoupledStepFailure): d.step(state)
    assert state.wall_pressure_mmhg==copied.wall_pressure_mmhg
    assert torch.equal(state.fsi.force,saved_force) and d._boundary_context is None


@pytest.mark.parametrize('cap_reference',[False,True])
def test_actual_wall_filling_to_pacing_handoff(mac_case,balanced_path,tmp_path,cap_reference):
    path,_=balanced_path; d=driver(mac_case)
    d.filling_pressure_location='endocardium'; d.filling_reservoir_mmhg=0.
    d.filling_cap_reference=cap_reference
    initial=d.initialize_preload(path)
    tick=26000-d.maximum_ticks; time=tick*d.protocol.dt_ep_s
    load=replace(initial.force_load,use_land=False,anchor=None,ramp_factor=0.)
    d.buffer.stage(load)
    old=replace(initial,phase='FILLING',mode='filling',cap_mode='rcr',
        fill_start_tick=25000,fill_end_tick=26000,fill_start_pressure_mmhg=7.,
        ep=replace(initial.ep,tick=tick,force_load=load,
            cycle=replace(initial.cycle,mode='zero',closure_tick=23000,freeze_tick=24000),
            fsi=replace(initial.fsi,time=time,force_time=time,pressure_time=time)))
    old=replace(old,ep=replace(old.ep,fsi=replace(old.fsi,force=d.force_for_state(old))))
    paced,info=d.step(old)
    assert paced.tick==26000 and paced.mode=='rcr' and paced.cycle.mode=='land'
    assert paced.wall_pressure_mmhg==8. and info['balanced']['boundary_pressure_mmhg']==pytest.approx(0.,abs=1e-7)
    assert paced.cap_reference_mmhg==(8. if cap_reference else 0.)
    path=tmp_path/'paced.npz'; restart.save(path,d,paced)
    restored,copied,_=restart.load(path,cardiac=True)
    assert restored.filling_cap_reference==cap_reference
    assert copied.cap_reference_mmhg==paced.cap_reference_mmhg
    resumed,info=restored.step(copied)
    assert resumed.wall_pressure_mmhg==0. and info['balanced']['wall_pressure_mmhg']==0.
    assert resumed.cap_reference_mmhg==0.
    assert info['cardiac']['solved_phase']=='ISOVOLUMIC_CONTRACTION'
    assert abs(info['balanced']['mass_error_ml_s'])<1e-5
    assert resumed.fsi.x.isfinite().all()
    assert d.protocol.relaxation_s==.080
    torch.testing.assert_close(restored.force_for_state(resumed),resumed.fsi.force,rtol=1e-9,atol=1e-7)
