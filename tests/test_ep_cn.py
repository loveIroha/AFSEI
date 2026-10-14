"""Independent CN equation, temporal order and closed-domain conservation."""
import pytest
import torch
from afsi_torch.ep import Monodomain,MonodomainNumerics
from afsi_torch.fluid.solvers import SolverOptions
from test_ep_monodomain import mesh


@pytest.mark.parametrize('discretization',['consistent_p1','nodal_p1'])
@pytest.mark.parametrize('scheme',['backward_euler','crank_nicolson'])
def test_theta_step_matches_dense_equation_and_conserves_mass(mesh,discretization,scheme):
    X,cells,f,s=mesh
    ep=Monodomain(X,cells,f,s,numerics=MonodomainNumerics(discretization,'torch_pcg_jacobi',
        SolverOptions(rtol=1e-13,atol=1e-15),time_scheme=scheme))
    M,K=ep.mass.to_dense(),ep.stiffness.to_dense();theta=ep.numerics.theta
    V=-.08+.02*X[:,0]-.01*X[:,1];I=X[:,2]+.2;S=X[:,1]*3
    dt=.01;alpha=.1/dt
    expected=torch.linalg.solve(alpha*M+theta*K,M@(alpha*V+.1*(S-I))-(1-theta)*K@V)
    result,info=ep.step(V,I,S,dt)
    torch.testing.assert_close(result,expected,rtol=1e-11,atol=1e-13)
    assert info.residual_norm<=info.tolerance
    zero=torch.zeros_like(V)
    diffused,_=ep.step(V,zero,zero,dt)
    torch.testing.assert_close((M@diffused).sum(),(M@V).sum(),rtol=1e-12,atol=1e-14)
    uniform=torch.full_like(V,-.081)
    uniform_result,_=ep.step(uniform,zero+2,zero+25,5e-5)
    torch.testing.assert_close(uniform_result,uniform+5e-5*23,rtol=1e-11,atol=1e-13)


@pytest.mark.parametrize('discretization',['consistent_p1','nodal_p1'])
def test_cn_second_order_on_exact_semidiscrete_diffusion_eigenmode(mesh,discretization):
    X,cells,f,s=mesh
    ep=Monodomain(X,cells,f,s,numerics=MonodomainNumerics(discretization,'torch_pcg_jacobi',
        SolverOptions(rtol=1e-13,atol=1e-15),time_scheme='crank_nicolson'))
    M,K=ep.mass.to_dense(),ep.stiffness.to_dense()
    inverse=torch.linalg.inv(torch.linalg.cholesky(M))
    rates,modes=torch.linalg.eigh(inverse@K@inverse.T/.1)
    lam=rates[-1];mode=inverse.T@modes[:,-1]
    mode=mode/mode.abs().max()*.01
    duration=.01;initial=mode-.08;zero=torch.zeros_like(initial)
    expected=mode*torch.exp(-lam*duration)-.08
    errors=[]
    for steps in (2,4,8):
        state=initial.clone()
        for _ in range(steps):state,_=ep.step(state,zero,zero,duration/steps)
        errors.append(torch.linalg.vector_norm(state-expected).item())
    assert 3.8<errors[0]/errors[1]<4.2
    assert 3.8<errors[1]/errors[2]<4.2
    assert len(ep._systems)==3


def test_invalid_time_scheme_is_rejected():
    with pytest.raises(ValueError,match='time_scheme'):
        MonodomainNumerics(time_scheme='automatic')
