"""Compare the optimized derivative to unchanged force AD and finite differences."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.p1 import prepare_p1
from afsi_torch.ep.buffered_load import BufferedLandLoad, ActiveLoadState
from afsi_torch.ep.active_tangent import active_local_jacobian
from afsi_torch.mac.execution import tensor_kernel


@pytest.fixture(params=['cpu', 'cuda'])
def case(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    X = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                     device=request.param, dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2, 3]], device=request.param)
    load = BufferedLandLoad(prepare_p1(X, cells, 5), X.new_tensor([[1., 0., 0.]]))
    # Earlier fiber length .8 keeps Ta positive around the .87 length cutoff.
    state = ActiveLoadState(X*X.new_tensor([.8, 1., 1.]), X.new_tensor([.03, .07, .11, .13]),
        X.new_tensor([.06, .04, .13, .17]), X.new_tensor([.01, -.02, .03, -.04]),
        X.new_tensor([.03, -.04, .02, -.01]), .0001)
    load.stage(state)
    return X, load, state, tensor_kernel(active_local_jacobian, X.device)


def evaluate(load, kernel, x):
    return kernel(*load._args(x), load.dt, load.use_land, load.ramp)


def oracle(load, x):
    return torch.vmap(torch.func.jacrev(load._element_force, argnums=0))(*load._args(x))


def test_force_derivative_including_clamp_subgradients(case):
    X, load, state, kernel = case
    # Norm floor, zero Ta, both sides of length cutoffs, extra PK1 cutoff,
    # and exact kink values. At kinks AD is the intended one-sided convention.
    for stretch in (0., 1e-8, .5, .869, .87, .871, .999, 1., 1.001, 1.199, 1.2, 1.201, 1.4):
        x = X.clone(); x[:, 0] *= stretch
        actual = evaluate(load, kernel, x)
        torch.testing.assert_close(actual, oracle(load, x), rtol=5e-13, atol=2e-8)
    # Check the installed clamp convention when tension is zero or negative.
    for xs, zs in ((0., 0.), (.1, -1.), (.1, -2.)):
        load.stage(replace(state, xs=torch.full_like(state.xs, xs),
            xw=torch.zeros_like(state.xw), zs_old=torch.full_like(state.zs_old, zs),
            previous_x=X))
        torch.testing.assert_close(evaluate(load, kernel, X), oracle(load, X), rtol=5e-13, atol=2e-8)


def test_shear_nonunit_fiber_and_independent_difference(case):
    X, load, state, kernel = case
    load.fiber.copy_(X.new_tensor([[1., .17, -.09]]))
    x = X @ X.new_tensor([[1.09, .04, .02], [.01, .96, .03], [0., .02, 1.01]]).T
    d = X.new_tensor([[.03, -.02, .01], [.01, .04, -.03], [-.02, .01, .02], [.04, .03, -.01]])
    actual = evaluate(load, kernel, x)
    torch.testing.assert_close(actual, oracle(load, x), rtol=5e-13, atol=2e-8)
    action = torch.einsum('eaibk,ebk->eai', actual, d[load.geometry.cells])[0]
    eps = 2e-6
    fd = (load.force(x+eps*d)-load.force(x-eps*d))/(2*eps)
    torch.testing.assert_close(action, fd, rtol=3e-8, atol=2e-3)
    torch.testing.assert_close(actual.sum(1), torch.zeros_like(actual.sum(1)), rtol=0, atol=2e-8)


def test_frozen_ramp_and_stable_trial_buffers(case):
    X, load, state, kernel = case
    x = X.clone(); x[:, 0] *= 1.09
    anchor = load.closure_snapshot(x)
    identities = [id(t) for t in (load.fields, load.dt, load.use_land, load.ramp)]
    copies = [t.clone() for t in (load.previous_F, load.fields, load.anchor)]
    evaluate(load, kernel, x)
    for a, b in zip((load.previous_F, load.fields, load.anchor), copies):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for trial in (replace(state, dt_s=.00005),
                  replace(state, use_land=False, anchor=anchor, ramp_factor=.6),
                  replace(state, use_land=False, anchor=None, ramp_factor=0.), state):
        load.stage(trial)
        for scale in (.9, 1., 1.09):
            x = X.clone(); x[:, 0] *= scale
            torch.testing.assert_close(evaluate(load, kernel, x), oracle(load, x), rtol=5e-13, atol=2e-8)
    assert identities == [id(t) for t in (load.fields, load.dt, load.use_land, load.ramp)]
