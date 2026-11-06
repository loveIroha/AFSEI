from copy import deepcopy
import pytest
from afsi_torch.simulation.circular_tube_restart import restart_plan


def inputs():
    old = dict(time=dict(fsi_dt_s=5e-5,ep_dt_s=5e-5,ode_dt_ms=.002,end_time_s=.26),
               active=dict(tension_scale=1), output=dict(log_every=100))
    new = deepcopy(old)
    new['time'].update(fsi_dt_s=2e-5,ep_dt_s=2e-5,end_time_s=.3)
    saved = dict(step=5200,time_s=.26,ep_steps=5200)
    return old,new,saved


def test_refinement_preserves_update_ledger_and_absolute_clock():
    old,new,saved=inputs()
    p=restart_plan(old,new,saved,refine_time=True)
    assert p['solver_step_index']==13000
    assert p['accepted_ep_steps']==p['accepted_fsi_steps']==5200
    assert p['accepted_ode_substeps']==130000
    assert p['predictor_scale']==pytest.approx(.4)
    assert p['clock_s']==.26


def test_default_refuses_changed_dt():
    with pytest.raises(ValueError): restart_plan(*inputs())


def test_refined_checkpoint_can_resume_with_same_dt():
    _,new,_=inputs()
    saved=dict(step=15000,time_s=.3,ep_steps=7200,fsi_steps=7200,ode_substeps=150000)
    p=restart_plan(new,new,saved)
    assert p['solver_step_index']==15000
    assert p['accepted_fsi_steps']==7200
    assert p['accepted_ode_substeps']==150000


@pytest.mark.parametrize('change',['material','ode','off_grid','bad_clock','ledger'])
def test_invalid_resume_rejected(change):
    old,new,saved=inputs()
    if change=='material': new['active']['tension_scale']=2
    if change=='ode': new['time']['ode_dt_ms']=.001
    if change=='off_grid': new['time'].update(fsi_dt_s=18e-6,ep_dt_s=18e-6)
    if change=='bad_clock': saved['time_s']=.26001
    if change=='ledger': saved['fsi_steps']=5199
    with pytest.raises(ValueError): restart_plan(old,new,saved,refine_time=True)


def test_zero_advance_restart_keeps_predictor_interval():
    _,new,_=inputs()
    saved=dict(step=13000,time_s=.26,ep_steps=5200,fsi_steps=5200,ode_substeps=130000,previous_dt_s=5e-5)
    plan=restart_plan(new,new,saved)
    assert plan['predictor_scale']==pytest.approx(.4)
