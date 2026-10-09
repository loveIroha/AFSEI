from dataclasses import replace
import numpy as np
import pytest
import torch
from test_real_lv import real_case
from test_paper_lv import config_for
from afsi_torch.paper_lv import imported_model
from afsi_torch.ep.config import AFSEIConfig
from afsi_torch.moving_cavity import from_solid,ALEConfig
from afsi_torch.circulation.geometry import PortGeometry
from afsi_torch.two_way import ThreeFieldStepper
from afsi_torch.two_way import checkpoint


@pytest.fixture
def physical_case(real_case,monkeypatch):
    if not torch.cuda.is_available(): pytest.skip('actual CUDA required')
    from afsi_torch.mac.paper_coupling import BEIBStepper
    def forbidden(*args,**kwargs): raise AssertionError('the exterior IB box must not be constructed')
    monkeypatch.setattr(BEIBStepper,'__init__',forbidden)
    cfg=AFSEIConfig.source(fsi=config_for(real_case,solver='newton'))
    mesh=imported_model(cfg.fsi,'cuda').mesh
    ports=PortGeometry(mv_center_cm=(.2,0),av_center_cm=(-.2,0),mv_radius_cm=.12,av_radius_cm=.12)
    fluid=from_solid(mesh,ports,cap_max_edge_cm=.05)
    d=ThreeFieldStepper(mesh,cfg,fluid,ALEConfig(start_time_s=.5,initial_pc_mmhg=85.))
    return d,d.initialize_three_fields()


def arrays(s):
    e=s.ep; f=e.fsi; b=s.blood
    return [f.x,f.velocity,f.force,e.tissue.voltage_v,e.tissue.gpb,e.tissue.land,e.strain.stretch,e.strain.zs,e.strain.zw,
        e.ionic_current,e.force_load.previous_x,e.force_load.xs,e.force_load.xw,e.force_load.zs_old,e.force_load.zw_old,
        b.solid_x,b.velocity,b.pressure,b.pc_mmhg,b.q_out_ml_s,b.port_pressure_cgs]


def test_three_field_clock_checkpoint_resume_and_integrity(physical_case,tmp_path):
    d,old=physical_case; copies=[x.clone() for x in arrays(old)]
    new,diag,force=d.step_three_fields(old)
    assert new.ep.tick==new.blood.tick==old.ep.tick+2
    assert diag['ep']['steps']==2 and diag['ep']['ode_substeps']==50
    assert diag['two_way_interface_feedback_implemented'] and not diag['original_box_fluid_used']
    assert diag['solid_momentum_residual_norm_dyn']<=diag['solid_momentum_tolerance_dyn']
    for x,y in zip(arrays(old),copies): torch.testing.assert_close(x,y,rtol=0,atol=0)
    path=tmp_path/'three_fields.npz'; checkpoint.save(path,d,new,{'preload_performed':False})
    restored,restart,progress=checkpoint.load(path)
    assert progress['preload_performed'] is False
    for x,y in zip(arrays(new),arrays(restart)): torch.testing.assert_close(x,y,rtol=0,atol=0)
    direct,_,_=d.step_three_fields(new); resumed,_,_=restored.step_three_fields(restart)
    for x,y in zip(arrays(direct),arrays(resumed)): torch.testing.assert_close(x,y,rtol=3e-7,atol=2e-7)
    with np.load(path,allow_pickle=False) as f: data={n:f[n] for n in f.files}
    data['ep__voltage_v'][0]+=.001; np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='checksum'): checkpoint.load(path)


def test_three_field_post_solve_failure_rolls_back_and_retries(physical_case,monkeypatch):
    d,old=physical_case; copies=[x.clone() for x in arrays(old)]
    project=d.ep.project_cell_field; fail=[]
    def injected(value):
        if not fail:
            fail.append(True); raise FloatingPointError('injected after two-way solve before strain publication')
        return project(value)
    monkeypatch.setattr(d.ep,'project_cell_field',injected)
    new,diag,force=d.step_three_fields(old)
    assert new.ep.tick==new.blood.tick==old.ep.tick+1
    assert diag['rejected_attempts']==new.ep.rejected_attempts==1
    assert diag['ep']['steps']==1 and diag['ep']['ode_substeps']==25
    for x,y in zip(arrays(old),copies): torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_seeded_land_closed_phase_and_real_constitutive_volume_constraint(physical_case):
    from afsi_torch.ep.tissue import TissueState
    inherited,unused=physical_case
    # The generic mesh fixture stores a uniform x-direction fiber. Use an
    # explicitly circumferential contractile fixture for isovolumic systole.
    mesh=inherited.mesh; center=mesh.X.new_tensor([7.5,7.5,7.5]); cell_center=mesh.X[mesh.cells].mean(1)-center
    fiber=torch.stack((-cell_center[:,1],cell_center[:,0],torch.zeros_like(cell_center[:,0])),1)
    fiber=fiber/fiber.norm(dim=1,keepdim=True)
    sheet=torch.zeros_like(fiber); sheet[:,2]=1.
    mesh=replace(mesh,fiber=fiber,sheet=sheet)
    d=ThreeFieldStepper(mesh,inherited.config,inherited.blood_driver.mesh,inherited.blood_driver.config)
    tissue=unused.ep.tissue; land=tissue.land.clone(); land[:,0]=.02; land[:,1]=.002
    old=d.initialize_three_fields(tissue=TissueState(tissue.voltage_v,tissue.gpb,land))
    new,diag,force=d.step_three_fields(old)
    assert new.blood.opened==(False,False),diag
    assert diag['closed_volume_constraint_active']
    assert abs(diag['volume_rate_ml_s'])<=1e-8
    assert diag['q_in_ml_s']==diag['q_out_ml_s']==0.
    assert 8.5<diag['pressure_mean_mmhg']<85.
    assert diag['solid_momentum_residual_norm_dyn']<=diag['solid_momentum_tolerance_dyn']
