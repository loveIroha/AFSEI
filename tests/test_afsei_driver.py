"""Physical state publication, rollback, compiled loads and exact EP clocks."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_real_lv import real_case
from test_paper_lv import config_for
from afsi_torch.paper_lv import imported_model,PaperLVConfig
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.ep.driver import AFSEIStepper,CoupledStepFailure
from afsi_torch.ep.tissue import initial_state,TissueState
from afsi_torch.ep.protocol import EPProtocol,CycleState,recovery_ready,ticks
from afsi_torch.ep.buffered_load import BufferedLandLoad,ActiveLoadState
from afsi_torch.ep.mapping import source_p1_fields
from afsi_torch.ep.checkpoint import save,load


def test_integer_protocol_and_spatial_recovery():
    p = EPProtocol()
    mask = torch.tensor([True,False]); like = torch.zeros(2,dtype=torch.float64)
    for pace in p.pulse_ticks:
        assert sum(bool(p.stimulus(i,mask,like)[0]) for i in range(pace-1,pace+41)) == 40
    cycle = CycleState(mode='relaxation',closure_tick=11000)
    assert cycle.factor(11000,p) == 1 and cycle.factor(12600,p) == 0
    assert p.limit(12599,2,cycle) == 1
    assert CycleState(freeze_tick=12000).ep_live(13000,p) is False
    state = initial_state(like)
    ready,_ = recovery_ready(state,state,state.voltage_v,state.gpb[:,37],.0001,p)
    assert ready
    gpb = state.gpb.clone(); gpb[-1,37] *= 1.2
    assert not recovery_ready(state,TissueState(state.voltage_v,gpb,state.land),
                             state.voltage_v,state.gpb[:,37],.0001,p)[0]
    with pytest.raises(ValueError): ticks(.000075,p.dt_ep_s)


@pytest.fixture
def case(real_case):
    if not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    fsi = config_for(real_case,solver='newton')
    # Exercise the production compiled force and graph fluid backends.
    fsi = replace(fsi,execution=PaperLVConfig().execution)
    mesh = imported_model(fsi,'cuda').mesh
    driver = AFSEIStepper(mesh,AFSEIConfig(fsi=fsi))
    tissue = initial_state(mesh.X[:,0])
    land = tissue.land.clone()
    land[:,0] = 1e-4; land[:,1] = 1e-4
    # A small seeded activation makes motion measurable on this short fixture.
    state = driver.initialize(tissue=TissueState(tissue.voltage_v,tissue.gpb,land))
    return driver,state


def physical_arrays(state):
    return [state.fsi.x,state.fsi.pressure,state.fsi.force,*state.fsi.velocity,
            state.tissue.voltage_v,state.tissue.gpb,state.tissue.land,
            state.strain.stretch,state.strain.zs,state.strain.zw,state.ionic_current,
            state.force_load.previous_x,state.force_load.xs,state.force_load.xw,
            state.force_load.zs_old,state.force_load.zw_old]


def assert_same(a,b,*,exact=False):
    assert a.tick == b.tick and a.fsi.time == b.fsi.time and a.fsi.step == b.fsi.step
    for x,y in zip(physical_arrays(a),physical_arrays(b)):
        torch.testing.assert_close(x,y,rtol=0 if exact else 3e-8,atol=0 if exact else 2e-8)


def test_source_owner_and_stimulus(case):
    driver,_ = case; mesh = driver.mesh
    f,s,mask,owner = source_p1_fields(mesh)
    expected = torch.full_like(owner,-1)
    for i,c in enumerate(mesh.cells): expected[c] = i
    assert torch.equal(owner,expected)
    endo = torch.zeros(len(mesh.cells),device=mesh.X.device,dtype=torch.bool)
    endo[mesh.boundary_cells[mesh.facet_tags==1]] = True
    assert torch.equal(mask,endo[owner])
    assert torch.equal(f,mesh.fiber[owner]) and torch.equal(s,mesh.sheet[owner])


def test_buffered_force_jvp_and_compiled_updates(case):
    driver,old = case
    buffer = driver.buffer; x = old.fsi.x*1.0002
    direction = torch.sin(x)
    compiled = torch.compile(buffer.force,fullgraph=True)
    identities = [id(v) for v in (buffer.fields,buffer.previous_F,buffer.dt,buffer.anchor,buffer.use_land,buffer.ramp)]
    anchor = buffer.closure_snapshot(x)
    trials = [old.force_load,
              replace(old.force_load,dt_s=5e-5,xs=old.force_load.xs*2),
              replace(old.force_load,use_land=False,anchor=anchor,ramp_factor=.7),
              replace(old.force_load,use_land=False,anchor=None,ramp_factor=0)]
    for trial in trials:
        buffer.stage(trial); reference = trial.materialize(driver.mesh)
        torch.testing.assert_close(buffer.force(x),reference.force(x),rtol=1e-11,atol=2e-9)
        torch.testing.assert_close(compiled(x),reference.force(x),rtol=1e-10,atol=2e-8)
        torch.testing.assert_close(buffer.force_jvp(x,direction),reference.force_jvp(x,direction),rtol=1e-10,atol=2e-7)
    assert identities == [id(v) for v in (buffer.fields,buffer.previous_F,buffer.dt,buffer.anchor,buffer.use_land,buffer.ramp)]


def test_unified_endpoint_and_force_history_checkpoint(case,tmp_path):
    driver,old = case
    copies = [v.clone() for v in physical_arrays(old)]
    new,info = driver.step(old)
    assert new.tick == old.tick+2 and info['ep']['steps'] == 2 and info['ep']['ode_substeps'] == 50
    assert new.fsi.force_time == new.fsi.pressure_time == new.fsi.time
    assert info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']
    assert info['divergence_l2'] < 1e-8 and info['power_error'] < 1e-6
    assert (new.fsi.x-old.fsi.x).abs().max() > 0
    assert max(u.abs().max().item() for u in new.fsi.velocity) > 0
    assert (new.strain.zs-old.strain.zs).abs().max() > 0
    for a,b in zip(physical_arrays(old),copies): torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(new.force_load.zs_old,old.strain.zs,rtol=0,atol=0)
    path = tmp_path/'coupled.npz'; save(path,driver,new,dict(ep_steps=2))
    restored,restart,progress = load(path)
    assert_same(new,restart,exact=True); assert progress['ep_steps'] == 2
    direct,_ = driver.step(new); resumed,_ = restored.step(restart)
    assert_same(direct,resumed)
    # Content integrity rejects silent edits before allocating the solver.
    with np.load(path,allow_pickle=False) as archive: data = dict(archive)
    data['voltage_v'][0] += .001
    with path.open('wb') as stream: np.savez(stream,**data)
    with pytest.raises(ValueError,match='checksum'): load(path)


def test_rejected_complete_trial_restores_ep_and_halves_dt(case,monkeypatch):
    driver,old = case; originals = [v.clone() for v in physical_arrays(old)]
    project = driver.ep.project_cell_field
    failures = []
    def fail_after_fsi(value):
        if not failures:
            failures.append(True)
            raise FloatingPointError('injected post-FSI feedback failure')
        return project(value)
    monkeypatch.setattr(driver.ep,'project_cell_field',fail_after_fsi)
    new,info = driver.step(old)
    assert failures and new.tick == old.tick+1
    assert info['ep']['steps'] == 1 and info['ep']['ode_substeps'] == 25
    assert len(info['rejected_attempts']) == new.rejected_attempts == 1
    assert new.fsi.previous_dt == 5e-5
    expected,_ = driver.attempt(old,1)
    assert_same(new,expected)
    for a,b in zip(physical_arrays(old),originals): torch.testing.assert_close(a,b,rtol=0,atol=0)
    def always_fail(value): raise FloatingPointError('injected permanent failure')
    monkeypatch.setattr(driver.ep,'project_cell_field',always_fail)
    with pytest.raises(CoupledStepFailure): driver.step(old)
    torch.testing.assert_close(driver.model.force(old.fsi.x,old.fsi.time),old.fsi.force,rtol=1e-10,atol=1e-7)
    for a,b in zip(physical_arrays(old),originals): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_accepted_closure_anchor_and_land_reset(case,tmp_path):
    driver,old = case
    closed,info = driver.attempt(old,2,closure=True)
    assert closed.cycle.mode == 'relaxation' and closed.cycle.closure_tick == closed.tick
    torch.testing.assert_close(closed.cycle.anchor,driver.buffer.tension(closed.fsi.x),rtol=0,atol=0)
    torch.testing.assert_close(closed.tissue.land,old.cycle.resting_land,rtol=0,atol=0)
    assert not torch.equal(closed.tissue.gpb,old.tissue.gpb)
    assert closed.strain.zs.abs().max() == closed.strain.zw.abs().max() == 0
    force = closed.fsi.force.clone()
    driver.buffer.stage(replace(closed.force_load,use_land=False,anchor=closed.cycle.anchor,ramp_factor=1.))
    torch.testing.assert_close(driver.model.force(closed.fsi.x,closed.fsi.time),force,rtol=1e-11,atol=1e-7)
    save(tmp_path/'closure.npz',driver,closed)
    restored,restart,_ = load(tmp_path/'closure.npz')
    next_state,next_info = restored.step(restart)
    assert next_info['ep']['steps'] == 2 and not next_info['ep']['land_advanced']
    torch.testing.assert_close(next_state.tissue.land,closed.tissue.land,rtol=0,atol=0)
    assert next_state.force_load.ramp_factor == pytest.approx(1-.0001/.080)
    with pytest.raises(ValueError,match='closure'): restored.attempt(next_state,2,closure=True)


def test_runner_restart_and_failed_run_publishes_only_accepted_state(case,tmp_path,monkeypatch):
    from afsi_torch.simulation.afsei import run
    driver,_ = case
    folder = tmp_path/'run'
    first = run(config=driver.config,output=folder,duration_s=.0001)
    assert first['accepted_steps'] == 1 and first['ep_steps'] == 2
    continued = run(resume=folder/'checkpoint.npz',duration_s=.00005)
    direct = run(config=driver.config,output=tmp_path/'direct',duration_s=.00015)
    assert continued['accepted_steps'] == direct['accepted_steps'] == 2
    assert continued['ep_steps'] == direct['ep_steps'] == 3
    _,a,_ = load(folder/'checkpoint.npz'); _,b,_ = load(tmp_path/'direct/checkpoint.npz')
    assert_same(a,b)
    before = [v.clone() for v in physical_arrays(a)]
    def fail(*args,**kwargs): raise CoupledStepFailure('injected failed interval',[{'ticks':1}])
    monkeypatch.setattr(AFSEIStepper,'step',fail)
    with pytest.raises(CoupledStepFailure): run(resume=folder/'checkpoint.npz',duration_s=.0001)
    _,after,progress = load(folder/'checkpoint.npz')
    assert_same(a,after,exact=True)
    assert progress['failure']['last_accepted_ep_tick'] == a.tick
    for x,y in zip(before,physical_arrays(after)): torch.testing.assert_close(x,y,rtol=0,atol=0)
