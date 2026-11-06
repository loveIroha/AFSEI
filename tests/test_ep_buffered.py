"""Independent dense references and publication/failure contracts for buffered EP."""
import pytest
import torch
from afsi_torch.ep.jacobi import JacobiPCG
from afsi_torch.fluid.solvers import SolverOptions

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_spd_true_residual_and_owned_solution(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    g=torch.Generator().manual_seed(162)
    q=torch.randn(19,19,generator=g,dtype=torch.float64).to(device)
    dense=q.T@q+torch.eye(19,device=device,dtype=q.dtype)*15
    A=dense.to_sparse_csr();b=torch.linspace(-1,1,19,device=device,dtype=q.dtype)
    solver=JacobiPCG(A,dense.diag(),SolverOptions(rtol=1e-11,atol=1e-13,
        max_iterations=101,recompute_every=17,check_every=5))
    x,info=solver.solve(b,initial=torch.zeros_like(b));saved=x.clone()
    assert torch.linalg.vector_norm(b-dense@x)<=info.tolerance
    torch.testing.assert_close(x,torch.linalg.solve(dense,b),rtol=1e-10,atol=1e-12)
    solver.solve(b*2,initial=torch.zeros_like(b));assert torch.equal(x,saved)
    x0,info0=solver.solve(torch.zeros_like(b),initial=torch.zeros_like(b))
    assert info0.iterations==0 and torch.equal(x0,torch.zeros_like(b))
    with pytest.raises((RuntimeError,torch.jit.Error)):solver.solve(b*float('nan'),initial=b)
    solver.close()

def test_gpu_rhs_matches_dense_and_preserves_inputs():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from afsi_torch.ep.gpu_kernels import BufferedRHS
    dev='cuda';dtype=torch.float64
    M=torch.tensor([[2.,1.,0.],[1.,3.,1.],[0.,1.,2.]],device=dev,dtype=dtype)
    K=torch.tensor([[1.,-1.,0.],[-1.,2.,-1.],[0.,-1.,1.]],device=dev,dtype=dtype)
    v=torch.tensor([-.08,.03,-.06],device=dev,dtype=dtype);i=v*4;s=v+1
    for nodal in (True,False):
        mass=M.diag().diag() if nodal else M
        rhs=BufferedRHS(mass.to_sparse_csr(),mass.diag(),K.to_sparse_csr(),7.,.1,.5,nodal)
        expected=mass@(7*v+.1*(s-i))-.5*K@v
        torch.testing.assert_close(rhs(v,i,s),expected,rtol=1e-14,atol=1e-15)
        torch.testing.assert_close(v,torch.tensor([-.08,.03,-.06],device=dev,dtype=dtype))
