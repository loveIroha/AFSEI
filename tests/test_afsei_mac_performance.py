"""Independent implicit solves, joint roots and fresh-tangent failure recovery."""
from dataclasses import replace
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import balanced_path,dynamic_driver
from test_afsei_driver import assert_same
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.mac_balanced_dynamic import BalancedDynamicProblem
from afsi_torch.mac.backward_euler import BackwardEulerFlow,BEFlowOptions
from afsi_torch.mac.grid import MACGrid


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('backend',['reference','graph'])
def test_initial_sweep_schedule_keeps_implicit_solution_and_owned_outputs(device,backend):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    grid=MACGrid((8,10,12),(1.3,1.9,2.1))
    options=BEFlowOptions(helmholtz_backend=backend,convection=False)
    reference=BackwardEulerFlow(grid,dt=1e-4,device=device,options=options)
    trial=BackwardEulerFlow(grid,dt=1e-4,device=device,options=replace(options,initial_sweeps=4))
    rhs=tuple(torch.sin(torch.arange(v.numel(),device=device,dtype=v.dtype)).reshape_as(v)
              for v in grid.zeros(device=device))
    original=tuple(v.clone() for v in rhs)
    expected,q,_=reference.solve_rhs(rhs)
    actual,p,info=trial.solve_rhs(rhs)
    for a,b in zip((*actual,p),(*expected,q)):
        torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-12)
    assert all(r<=t for r,t in zip(info['helmholtz_residuals'],info['helmholtz_tolerances']))
    retained=tuple(v.clone() for v in (*actual,p))
    zero,_,_=trial.solve_rhs(tuple(torch.zeros_like(v) for v in rhs))
    assert all(v.count_nonzero()==0 for v in zero)
    for a,b in zip((*actual,p),retained): torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(rhs,original): torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_joint_roots_and_execution_restart_match_fresh_newton(mac_case,balanced_path,tmp_path):
    path,_=balanced_path
    for mode in ('closed','rcr','filling'):
        d=dynamic_driver(mac_case)
        original_mode='rcr' if mode=='rcr' else 'closed'
        initial=d.initialize_preload(path,mode=original_mode,arterial_pc_mmhg=7.9)
        initial=replace(initial,mode=mode,cap_mode=original_mode,filling_pressure_mmhg=8.01)
        saved=tmp_path/f'{mode}.npz'; restart.save(saved,d,initial)
        reference,old,_=restart.load(saved); reference.reuse_step_tangent=False
        assert d.reuse_step_tangent
        a,ia=d.step(initial); b,ib=reference.step(old)
        assert_same(a,b)
        torch.testing.assert_close(a.cap_pressure,b.cap_pressure,rtol=3e-8,atol=2e-7)
        for info in (ia,ib):
            assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
            assert abs(info['balanced']['mass_error_ml_s'])<1e-5
            assert info['balanced']['boundary_work_relative_error']<1e-7
        restart.save(saved,d,a)
        restored,copy,_=restart.load(saved)
        assert restored.reuse_step_tangent==d.reuse_step_tangent
        assert restored.balanced_tangent_chunk_size==d.balanced_tangent_chunk_size
        assert restored.helmholtz_initial_sweeps==d.helmholtz_initial_sweeps
        replay,_=restored.step(copy); continued,_=d.step(a); assert_same(replay,continued)


def test_failed_reused_linearization_recovers_with_fresh_newton(mac_case,balanced_path,monkeypatch):
    path,_=balanced_path; d=dynamic_driver(mac_case); old=d.initialize_preload(path)
    linearization=BalancedDynamicProblem.linearization
    attempts=[]
    def injected(problem,z,*,reuse_tangent=False):
        attempts.append(reuse_tangent)
        if reuse_tangent: raise RuntimeError('injected reused tangent failure')
        return linearization(problem,z)
    monkeypatch.setattr(BalancedDynamicProblem,'linearization',injected)
    new,info=d.step(old)
    assert True in attempts and False in attempts
    assert info['nonlinear']['tangent_refresh_fallback']
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    assert abs(info['balanced']['mass_error_ml_s'])<1e-5
    assert new.tick>old.tick and not info['rejected_attempts']


def test_half_step_filling_and_return_to_full_step(mac_case,balanced_path,tmp_path):
    path,_=balanced_path; d=dynamic_driver(mac_case)
    old=d.initialize_preload(path)
    old=replace(old,mode='filling',cap_mode='closed',filling_pressure_mmhg=8.01)
    saved=tmp_path/'half-step.npz'; restart.save(saved,d,old)
    reference,copy,_=restart.load(saved); reference.reuse_step_tangent=False
    a,ia=d.step(old,target_tick=old.tick+1)
    b,ib=reference.step(copy,target_tick=copy.tick+1)
    assert_same(a,b)
    assert ia['accepted_dt_s']==ib['accepted_dt_s']==5e-5
    assert abs(ia['balanced']['mass_error_ml_s'])<1e-5
    restart.save(saved,d,a)
    restored,copy,_=restart.load(saved)
    continued,info=d.step(a); replay,_=restored.step(copy)
    assert_same(continued,replay)
    assert info['accepted_dt_s']==1e-4 and not info['rejected_attempts']
