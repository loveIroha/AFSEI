"""Execution-only optimizations: reference solves, ownership and bad inputs."""
import pytest
import torch
from test_real_lv import real_case
from test_afsei_mac_preload import mac_case
from test_afsei_mac_balanced_dynamic import dynamic_driver
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.backward_euler import BackwardEulerFlow,BEFlowOptions
from afsi_torch.ep.mac_cap_coupled import closed_rows
from afsi_torch.ep.mac_compact_cap import CompactCapRows,add_cap_lift
from afsi_torch.ep.mac_pressure_mode import PressureModeTransfer
from afsi_torch.ep.mac_pressure_compatibility import surface_stencil,_surface_stencil_values
from afsi_torch.mac.mass_graph import GraphMassSolver


DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA required'))]


@pytest.mark.parametrize('device',DEVICES)
def test_zero_mass_initialization_cannot_hide_bad_operator_or_guess(device):
    mass=torch.eye(5,device=device,dtype=torch.float64).to_sparse_csr()
    diagonal=torch.ones((5,3),device=device,dtype=torch.float64)
    solver=GraphMassSolver(mass,diagonal)
    zero=torch.zeros_like(diagonal)
    result,info=solver.solve(zero)
    assert info.iterations==0 and result.count_nonzero()==0
    with pytest.raises(ValueError,match='finite'):
        solver.solve(zero,torch.full_like(zero,float('nan')))
    bad=mass.clone(); bad.values()[0]=float('nan')
    with pytest.raises(ValueError,match='finite constant mass'):
        GraphMassSolver(bad,diagonal)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('backend',['workspace','graph'])
def test_direct_forcing_matches_original_rhs_and_keeps_owned_outputs(device,backend):
    grid=MACGrid((8,10,12),(1.3,1.9,2.1))
    opt=BEFlowOptions(convection=False,helmholtz_backend=backend,initial_sweeps=4)
    reference=BackwardEulerFlow(grid,dt=1e-4,device=device,options=opt)
    trial=BackwardEulerFlow(grid,dt=1e-4,device=device,options=opt)
    density=tuple(torch.sin(torch.arange(u.numel(),device=device,dtype=u.dtype)).reshape_as(u)
                  for u in grid.zeros(device=device))
    departure=tuple(.002*torch.cos(u) for u in density)
    saved=tuple(u.clone() for u in (*density,*departure))
    expected=reference.solve_rhs(tuple(reference.dt/reference.rho*f for f in density))[0]
    actual=trial.response(density)
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-12)
    owned=tuple(u.clone() for u in actual)
    expected,p,_=reference.solve_rhs(tuple(a+reference.dt/reference.rho*f for a,f in zip(departure,density)))
    result,q,info=trial.advance(departure,density)
    for a,b in zip((*result,q),(*expected,p)):
        torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-12)
    assert all(r<=t for r,t in zip(info['helmholtz_residuals'],info['helmholtz_tolerances']))
    for a,b in zip(actual,owned): torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip((*density,*departure),saved): torch.testing.assert_close(a,b,rtol=0,atol=0)
    invalid=list(density); invalid[0]=torch.full_like(invalid[0],float('nan'))
    with pytest.raises(FloatingPointError,match='nonfinite'): trial.response(tuple(invalid))
    zero=trial.response(grid.zeros(device=device))
    assert all(u.count_nonzero()==0 for u in zero)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kernel',['peskin4','bs4-bs3'])
def test_fused_surface_stencil_matches_eager_oracle_and_rejects_bad_points(device,kernel):
    grid=MACGrid((16,18,20),(4.,4.5,5.))
    points=torch.tensor([[1.,1.2,1.7],[2.01,2.32,2.85],[1.12,2.17,1.77]],
                        device=device,dtype=torch.float64)
    for actual,expected in zip(surface_stencil(grid,points,kernel=kernel),
                               _surface_stencil_values(grid,points,kernel)):
        assert torch.equal(actual[0],expected[0])
        torch.testing.assert_close(actual[1],expected[1],rtol=2e-13,atol=2e-15)
    with pytest.raises(ValueError,match='finite'):
        surface_stencil(grid,points*float('nan'),kernel=kernel)
    with pytest.raises(ValueError,match='outer box'):
        surface_stencil(grid,points*10,kernel=kernel)


def test_compact_inplace_lift_matches_full_lift_and_preserves_public_validation(mac_case):
    d=dynamic_driver(mac_case); e=d._fsi_stepper(d.maximum_ticks)
    rows=closed_rows(e,d.mesh.X*1.002)
    assert isinstance(rows,CompactCapRows)
    p=3+torch.sin(torch.arange(rows.count,device=rows.area.device,dtype=rows.area.dtype))
    density=tuple(torch.cos(e.grid.coordinates(c,device=p.device)[...,c]) for c in range(3))
    saved=tuple(v.clone() for v in density)
    expected=tuple(a+b for a,b in zip(saved,rows.lift(p)))
    pointers=[v.data_ptr() for v in density]
    result=add_cap_lift(rows,p,density)
    assert [v.data_ptr() for v in result]==pointers
    for a,b in zip(result,expected): torch.testing.assert_close(a,b,rtol=2e-12,atol=1e-10)
    for operation in (rows.lift,rows.reaction,lambda p:rows.add_lift_into(p,density)):
        with pytest.raises(ValueError,match='finite'):
            operation(torch.full_like(p,float('nan')))
    force=rows.reaction(torch.full_like(p,float('nan')),check_finite=False)
    with pytest.raises(ValueError,match='finite'):
        e.transfer.base.solve_mass(force,None)


def test_cached_geometry_matches_fresh_prepare_and_keeps_old_stencil(mac_case):
    d=dynamic_driver(mac_case); e=d._fsi_stepper(d.maximum_ticks); transfer=e.transfer
    x=d.mesh.X*1.002
    old=transfer.prepare(x)
    owned=old.potential.clone()
    audit=transfer.audit; rule=audit.rule(transfer.level)
    changed=x+1e-4*torch.sin(x)
    cached=transfer.prepare(changed)
    assert transfer.audit is audit and audit.rule(transfer.level).data_ptr()==rule.data_ptr()
    e.transfer=transfer.base
    try: fresh=PressureModeTransfer(e,level=transfer.level).prepare(changed)
    finally: e.transfer=transfer
    for name in ('mode','dual','potential','force_potential'):
        torch.testing.assert_close(getattr(cached,name),getattr(fresh,name),rtol=2e-11,atol=2e-8)
    for name in ('correction','gradient','target'):
        for a,b in zip(getattr(cached,name),getattr(fresh,name)):
            torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-8)
    torch.testing.assert_close(old.potential,owned,rtol=0,atol=0)
    with pytest.raises(ValueError,match='finite'):
        audit.validate_geometry(changed*float('nan'))
