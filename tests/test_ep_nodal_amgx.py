"""Independent vertex-quadrature references and actual CUDA backend checks."""
from dataclasses import asdict
import os
import pytest
import torch
from afsi_torch.ep import Monodomain, MonodomainNumerics
from afsi_torch.ep.config import AFSEIConfig, decode_checkpoint_config
from afsi_torch.fluid.solvers import SolverOptions
from test_ep_monodomain import mesh


def reference(X, cells, fiber, sheet):
    n = len(X)
    M, K, Mc = (X.new_zeros((n, n)) for _ in range(3))
    for cell in cells:
        nodes = X[cell]
        affine = torch.cat((X.new_ones((4, 1)), nodes), dim=1)
        grad = torch.linalg.inv(affine)[1:].T
        volume = torch.linalg.det((nodes[1:]-nodes[:1]).T).abs()/6
        f, s = fiber[cell], sheet[cell]
        normal = torch.linalg.cross(f, s)
        D = sum(g*(v[:, :, None]*v[:, None, :]) for g, v in
                zip((.117, .039, .039), (f, s, normal)))
        M[cell, cell] += volume/4
        Mc[cell[:, None], cell[None, :]] += volume/20*(
            X.new_ones((4, 4))+torch.eye(4, device=X.device, dtype=X.dtype))
        K[cell[:, None], cell[None, :]] += volume*grad@D.mean(0)@grad.T
    return M, K, Mc


def test_vertex_rule_variable_tensor_dense_solution_and_cached_dt(mesh):
    X, cells, fiber, sheet = mesh
    fiber = fiber.clone(); fiber[:, 0] += .3*X[:, 1]
    sheet = sheet.clone(); sheet[:, 1] += .2*X[:, 2]
    p = Monodomain(X, cells, fiber, sheet, numerics=MonodomainNumerics(
        backend='torch_pcg_jacobi'), solver_options=SolverOptions(rtol=1e-13, atol=1e-14))
    M, K, Mc = reference(X, cells, fiber, sheet)
    torch.testing.assert_close(p.mass.to_dense(), M, rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(p.stiffness.to_dense(), K, rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(p.projection_mass.to_dense(), Mc, rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(K@X.new_ones(len(X)), X.new_zeros(len(X)), atol=1e-14, rtol=0)
    assert torch.linalg.eigvalsh(M*2000+K).min() > 0
    V, I, S = X[:, 0]*.05-.08, X[:, 1]*3+.2, X[:, 2]*25
    for dt in (5e-5, 5e-5, 1e-4):
        theta=p.numerics.theta
        expected = torch.linalg.solve(.1/dt*M+theta*K, M@(.1/dt*V+.1*(S-I))-(1-theta)*K@V)
        actual, info = p.step(V, I, S, dt)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-13)
        assert info.residual_norm <= info.tolerance
    assert len(p._systems) == 2


def test_nodal_conservation_units_and_unchanged_nonuniform_land_projection(mesh):
    X, cells, f, s = mesh
    p = Monodomain(X, cells, f, s, numerics=MonodomainNumerics(backend='torch_pcg_jacobi'),
                   solver_options=SolverOptions(rtol=1e-13,atol=1e-14))
    old = Monodomain(X, cells, f, s, numerics=MonodomainNumerics.legacy())
    v, z = X.new_full((len(X),), -.081), X.new_zeros(len(X))
    result, _ = p.step(v, z+2, z+25, 5e-5)
    torch.testing.assert_close(result, v+5e-5*23, rtol=1e-11, atol=1e-13)
    nonuniform = v+X[:, 0]*.05
    result, _ = p.step(nonuniform, z, z, .02)
    torch.testing.assert_close((p.mass_diagonal*result).sum(),
                              (p.mass_diagonal*nonuniform).sum(), rtol=1e-10, atol=1e-13)
    cell_value = X.new_tensor([.9, 1.2])
    torch.testing.assert_close(p.project_cell_field(cell_value),
                              old.project_cell_field(cell_value), rtol=1e-10, atol=1e-12)


def test_new_defaults_and_legacy_restart_are_distinct():
    new = AFSEIConfig()
    assert new.ep_numerics == MonodomainNumerics()
    data = asdict(new); data.pop('ep_numerics')
    assert decode_checkpoint_config(data).ep_numerics == MonodomainNumerics.legacy()
    assert decode_checkpoint_config(asdict(new)) == new
    data=asdict(new);data['ep_numerics'].pop('time_scheme')
    old_nodal=decode_checkpoint_config(data)
    assert old_nodal.ep_numerics.discretization=='nodal_p1'
    assert old_nodal.ep_numerics.time_scheme=='backward_euler'
    assert new.ep_numerics.time_scheme=='crank_nicolson'
    with pytest.raises(ValueError):
        MonodomainNumerics(backend='auto')


@pytest.mark.parametrize('discretization', ['nodal_p1', 'consistent_p1'])
def test_actual_amgx_matches_same_equation_and_reuses_hierarchy(discretization):
    if not torch.cuda.is_available():
        pytest.skip('actual CUDA required')
    from afsi_torch.ep.amgx import _library_path
    try:
        _library_path()
    except RuntimeError:
        if os.environ.get('AFSEI_REQUIRE_AMGX') == '1':
            raise
        pytest.skip('optional AmgX not installed')
    X = torch.tensor([[0.,0.,0.], [.03,0.,0.], [0.,.03,0.], [0.,0.,.03],
                      [.03,.03,.03]], dtype=torch.float64, device='cuda')
    cells = torch.tensor([[0,1,2,3], [4,1,2,3]], device='cuda')
    f = X.new_tensor([1.,0.,0.]).expand(5,-1).clone(); f[:,0] += X[:,1]*2
    s = X.new_tensor([0.,1.,0.]).expand(5,-1)
    ep = Monodomain(X, cells, f, s, numerics=MonodomainNumerics(discretization))
    M, K = ep.mass.to_dense(), ep.stiffness.to_dense()
    V, I, S = X[:,0]-.08, X[:,1]*3+.2, X[:,2]*250
    first = None
    for _ in range(3):
        expected = torch.linalg.solve(2000*M+.5*K, M@(2000*V+.1*(S-I))-.5*K@V)
        result, info = ep.step(V, I, S, 5e-5)
        torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-12)
        assert info.residual_norm <= info.tolerance
        if first is None:
            first = result.clone()
            retained = result
        V = result+X[:,0]*.01
    torch.testing.assert_close(retained, first, rtol=0, atol=0)
    backend = ep._systems[5e-5][2]
    assert backend.setup_count == 1 and backend.solve_count == 3
    # A restored driver and its original may coexist. Closing either must
    # leave the other's native matrices and memory pool usable.
    restored = Monodomain(X, cells, f, s, numerics=MonodomainNumerics(discretization))
    restored.prepare(5e-5)
    ep.close()
    result, info = restored.step(V, I, S, 5e-5)
    expected = torch.linalg.solve(2000*M+.5*K, M@(2000*V+.1*(S-I))-.5*K@V)
    torch.testing.assert_close(result, expected, rtol=1e-9, atol=1e-12)
    assert info.residual_norm <= info.tolerance
    restored.close()
