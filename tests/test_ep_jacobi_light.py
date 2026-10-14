import pytest
import torch
from afsi_torch.ep import Monodomain,MonodomainNumerics
from afsi_torch.ep.jacobi import JacobiPCG
from afsi_torch.fluid.solvers import SolverOptions


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('recompute_every',[1,40])
def test_light_pcg_independent_spd(device,recompute_every):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    A=torch.tensor([[4.,-1.,0.],[-1.,3.,-.5],[0.,-.5,2.]],device=device,dtype=torch.float64)
    b=A.new_tensor([1.,2.,-.3]);initial=A.new_tensor([.2,-.1,.1]);saved=initial.clone()
    backend=JacobiPCG(A.to_sparse_csr(),A.diag(),SolverOptions(rtol=1e-12,atol=1e-14,max_iterations=200,recompute_every=recompute_every))
    x,info=backend.solve(b,initial=initial)
    torch.testing.assert_close(x,torch.linalg.solve(A,b),rtol=1e-11,atol=1e-12)
    assert torch.linalg.vector_norm(b-A@x).item()<=info.tolerance
    assert torch.equal(initial,saved)
    exact,zero=backend.solve(A@saved,initial=saved)
    assert zero.iterations==0 and torch.equal(exact,saved)
    with pytest.raises((RuntimeError,torch.jit.Error)):
        JacobiPCG(A.to_sparse_csr(),A.diag(),SolverOptions(rtol=1e-15,atol=1e-16,max_iterations=1)).solve(b,initial=initial)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_nodal_cn_step_and_cache(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    X=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],device=device,dtype=torch.float64)
    cells=torch.tensor([[0,1,2,3]],device=device)
    f=X.new_tensor([1.,0.,0.]).expand(4,-1);s=X.new_tensor([0.,1.,0.]).expand(4,-1)
    options=SolverOptions(rtol=1e-12,atol=1e-14)
    ep=Monodomain(X,cells,f,s,numerics=MonodomainNumerics(backend='torch_pcg_jacobi_light',solver=options))
    dt=5e-5;V=X.new_tensor([-.085,-.07,-.06,-.08]);I=X.new_zeros(4);S=X.new_tensor([35.,0.,0.,0.])
    ep.prepare(dt);alpha,A,backend=ep._systems[dt]
    rhs=ep.mass_diagonal*(alpha*V+ep.parameters.surface_to_volume_per_cm*S)-.5*(ep.stiffness@V)
    x,info=ep.step(V,I,S,dt)
    torch.testing.assert_close(x,torch.linalg.solve(A.to_dense(),rhs),rtol=1e-11,atol=1e-12)
    ep.step(V,I,S,dt)
    assert ep._systems[dt][2] is backend and info.residual_norm<=info.tolerance
    ep.close()
