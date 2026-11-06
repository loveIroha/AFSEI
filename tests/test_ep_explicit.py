import pytest
import torch
from afsi_torch.ep.monodomain import Monodomain
from afsi_torch.ep.numerics import MonodomainNumerics
from afsi_torch.ep.explicit import ExplicitNodal


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_explicit_dense_reference_cfl_and_conservation(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    X=torch.tensor([[0.,0.],[.025,0.],[.025,.025],[0.,.025]],device=device,dtype=torch.float64)
    cells=torch.tensor([[0,1,2],[0,2,3]],device=device)
    f=X.new_tensor([1.,0.]).expand(4,-1);s=X.new_tensor([0.,1.]).expand(4,-1)
    ep=Monodomain(X,cells,f,s,numerics=MonodomainNumerics(backend='torch_pcg_jacobi'))
    ex=ExplicitNodal(ep,.001)
    mass=ep.mass_diagonal*.1;K=ep.stiffness.to_dense()
    B=K/torch.sqrt(mass[:,None]*mass[None,:])
    assert ex.sub_dt*torch.linalg.eigvalsh(B).max().item()<2
    V=X[:,0]+.5*X[:,1];z=torch.zeros_like(V)
    expected=V.clone()
    for _ in range(ex.substeps):expected=expected-ex.sub_dt*(K@expected)/mass
    out,_=ex.step(V,z,z,.001)
    torch.testing.assert_close(out,expected,rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(mass@out,mass@V,rtol=1e-13,atol=1e-14)
    out,_=ex.step(z+1,z+2,z+5,.001)
    torch.testing.assert_close(out,z+1.003,rtol=1e-13,atol=1e-14)
